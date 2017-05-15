// compile with:
//    -O3 `pkg-config --cflags --libs opencv` -pthread

// C++ standard libs
#include <iostream>
#include <fstream>
#include <cmath>
#include <thread>
// OpenCV libs
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Constants
const int steps = 720;
const double pi = std::acos(-1);

// Global items
double *cosTable, *sinTable;
unsigned long **accumulator;
int maxR = 0;

// Prototypes
inline void checkCmd(int argc, char **argv);
void initTask(cv::Mat& gradient);
void getFirstDerivative(cv::Mat& image /*type: CV_8SC1*/, cv::Mat& result);
//void negate(cv::Mat& m);
void houghTransform(cv::Mat& image, unsigned long** accumulator, const int& threshold);
void searchAndDraw(cv::Mat& image, unsigned long** accumulator, const int& line_n);
void drawLine(cv::Mat gradient, int aa, int rr);

/*----------------------------------------------------------------------------*/
int main( int argc, char** argv ){
    
    checkCmd(argc, argv);
    
    cv::Mat image, gradient;
    int threshold = std::atoi(argv[2]), 
        line_n = std::atoi(argv[3]);
    
    image = cv::imread( argv[1], cv::IMREAD_GRAYSCALE );
    if( !image.data ){
        std::cerr << " No image data." << std::endl;
        std::exit(-1);
    }
    
    std::thread init(initTask, std::ref(image));

    image.convertTo(image, CV_8U);
    getFirstDerivative(image, gradient); // TODO: use thresold here
    init.join();
    houghTransform(gradient, accumulator, threshold);
    searchAndDraw(gradient, accumulator, line_n);

    cv::imwrite("result.jpg", gradient);
       
    for (int i = 0; i< steps; i++) delete[] accumulator[i];
    delete[] accumulator, sinTable, cosTable;
       
    return (0);
}
/*----------------------------------------------------------------------------*/



inline void checkCmd(int argc, char **argv){
    if( argc != 4 ){
        std::cerr << "Usage: " 
                  << argv[0] << " <image path> <intensity_threshold> <number of lines>" 
                  << std::endl;
        std::exit(-1);
    }
    if (std::atoi(argv[2]) > 255){
        std::cerr << "Out of family threshold value. Max: 255" << std::endl; 
        std::exit(-2);
    }
}


void initTask(cv::Mat& gradient){
    // Maximum distance of a straight line from the corner at (0,0)    
    maxR = std::sqrt(std::pow(gradient.rows-1,2) + std::pow(gradient.cols-1,2)) + 1;
    
    // Lookup tables
    // PI is splitted in steps, trigonometric functions are evaluated in each step.
    sinTable = new double[steps];
    cosTable = new double[steps];
    for (int i = 0; i<steps; i++){
        double angle = i * pi / steps;
        cosTable[i] = std::cos(angle);
        sinTable[i] = std::sin(angle);
    }
    
    // Accumulator required for Hough Transform
    // Spatial field: [0, PI]*[-maxR, maxR]
    accumulator = new unsigned long*[steps];
    for (int i = 0; i< steps; i++) accumulator[i] = new unsigned long[maxR*2];
    for (int i = 0; i< steps; i++) for (int j=0; j<maxR*2; j++) accumulator[i][j] = 0;
}

void getFirstDerivative(cv::Mat& image /*type: CV_8SC1*/, cv::Mat& result /*type: not initialized*/){
    CV_Assert(image.channels()==1);
    CV_Assert(result.channels()==1);
    int nRows = image.rows,
        nCols = image.cols;

    cv::Mat dx, dy; 
    dx.create(nRows-1, nCols-1, CV_8SC1);
    dy.create(nRows-1, nCols-1, CV_8SC1);
    result.create(nRows-1, nCols-1, CV_8UC1);
    
    // Calculate derivative from left to right
    for(int i = 0; i < nRows-1; ++i)
        for (int j = 0; j < nCols-1; ++j)
            dx.at<schar>(i,j) = (image.at<uchar>(i,j) - image.at<uchar>(i,j+1))/2;
    
    // Calculate derivative from top to bottom
    for(int i = 0; i < nRows-1; ++i)
        for (int j = 0; j < nCols-1; ++j)
            dy.at<schar>(i,j) = (image.at<uchar>(i,j) - image.at<uchar>(i+1,j))/2;
    
    // Calculate gradient
    uchar max = 0;
    for(int i = 0; i < nRows-1; ++i){
        for (int j = 0; j < nCols-1; ++j){
            uchar& p = result.at<uchar>(i,j);
            p = std::sqrt(std::pow(dx.at<schar>(i,j),2) + std::pow(dy.at<schar>(i,j),2));
            if (p>max) max = p;
        }
    }

    // Saturate gradient by expanding it on the full grayscale
    for(int i = 0; i < nRows-1; ++i)
        for (int j = 0; j < nCols-1; ++j)
            result.at<uchar>(i,j) =255 - cv::saturate_cast<uchar>(result.at<uchar>(i,j) * 255 / max);
}

//void negate(cv::Mat& m){
//    for(int i = 0; i < m.rows; ++i)
//        for (int j = 0; j < m.cols; ++j)
//            m.at<uchar>(i,j) = 255 - m.at<uchar>(i,j);
//}


void houghTransform(cv::Mat& image, unsigned long** accumulator, const int& threshold){
    for(int y = 0; y < image.rows; y++){
        for (int x = 0; x < image.cols; x++){
            if (image.at<uchar>(y,x) <= threshold){
                for (int angleStep = 0; angleStep < steps; angleStep++){
                    int distance = x*cosTable[angleStep] + y*sinTable[angleStep];
                    assert(distance + maxR >= 0);
                    accumulator[angleStep][distance + maxR] += 1;
                }
            }
        }
    }
}

void searchAndDraw(cv::Mat& image, unsigned long** accumulator, const int& line_n){
    std::ofstream f("data.csv", std::ios::trunc);
    f << "Angle;Distance;Counter;" << std::endl;
    for (int i =0; i< line_n; i++){
        //TODO: relative max values research
        int aa = 0,
            rr = 0;
        unsigned long max = 0;
        for(int a = 0; a<steps; a++)
            for (int r = 0; r < maxR*2; r++)
                if (accumulator[a][r] >= max)          
                    max=accumulator[a][r], aa=a, rr=r-maxR;
        accumulator[aa][rr+maxR] = 0;
        f << aa*180/steps << ';' << rr << ';' << max << ';' << std::endl;  
        // TODO: thread spawn with fixed number of thread active concurrently
        drawLine(image, aa, rr);
    }
    f.close();
}

void drawLine(cv::Mat gradient, int aa, int rr){
    // Try to determine intersection of a line with the borders of an image. 
    // A straight line crosses the borders of a rectangle either 0 or 2 times.
    // The line parameters are relative to the Hesse normal form.
    // Several assumptions and shorcuts have been used in order to optimize the procedure.
    if(!aa==0){
        std::pair<cv::Point, cv::Point> pp;
        int rows = gradient.rows;
        int cols = gradient.cols;
        char count = 0;
        int t;
        
        // x=0
        t = rr/sinTable[aa];
        if ( t >= 0  &&  t <= rows )
            pp.first = cv::Point(0,t), count++;
        
        // x = cols 
        t = (rr - cols*cosTable[aa])/sinTable[aa];
        if ( t >= 0  &&  t <= rows )
            switch(count){
                case 0: pp.first = cv::Point(cols,t), count++; break;
                case 1: pp.second = cv::Point(cols,t), count++; goto DRAW;
            }
        
        // y = 0
        t = rr/cosTable[aa];
        if ( t >= 0  &&  t <= cols )
            switch(count){
                case 0: pp.first = cv::Point(t,0), count++; break;
                case 1: pp.second = cv::Point(t,0), count++; goto DRAW;
            }

        // y = rows
        t = (rr - rows*sinTable[aa])/cosTable[aa];
        if ( t >= 0  &&  t <= cols )
            pp.second = cv::Point(t, rows); goto DRAW;
        
        // no intersection found with borders
        assert(count==0);
        pp.first = cv::Point(-1,-1), pp.second = cv::Point(-1,-1);
        
        DRAW:
        cv::line(gradient, 
                 pp.first, 
                 pp.second, 
                 cv::Scalar(100,100,100,100));
        }
    else {
        cv::line(gradient, 
                 cv::Point(rr/cosTable[aa],0), 
                 cv::Point(rr/cosTable[aa],gradient.rows), 
                 cv::Scalar(100,100,100,100));
    }
}
