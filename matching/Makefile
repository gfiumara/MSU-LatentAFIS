CXX=g++
# CFLAGS will be the options passed to the compiler.
INCLUDE=-I/usr/include/eigen3
CXXFLAGS=-c -O3 -fopenmp -std=gnu++11 $(INCLUDE)

all: match

match:main.o matcher.o
	$(CXX) -fopenmp main.o matcher.o -o match -lboost_system -lboost_filesystem

main.o:main.cpp
	$(CXX) $(CXXFLAGS) main.cpp

matcher_int.o:matcher.cpp
	$(CXX) $(CXXFLAGS) matcher.cpp

clean:
	rm match main.o matcher.o
