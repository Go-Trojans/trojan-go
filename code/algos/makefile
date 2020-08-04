CXX=g++
CFLAGS=-Wall -std=c++17
EXEC=run

all: gohelper.o
	$(CXX) $(CFLAGS) godomain.cpp gohelper.o -o $(EXEC)

gohelper.o: gohelper.cpp
	$(CXX) -c $(CFLAGS) gohelper.cpp

clean:
	rm $(EXEC) *.o