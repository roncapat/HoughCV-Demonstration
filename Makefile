CPPFLAGS=-O3 -std=c++17
OCVFLAGS=`pkg-config --cflags --libs opencv`
LDFLAGS=-pthread
CC=g++

all: build/houghCV

build/houghCV: src/main.cpp
	$(CC) $(CPPFLAGS) $(OCVFLAGS) $(LDFLAGS) $< -o $@

clean:
	rm build/*
