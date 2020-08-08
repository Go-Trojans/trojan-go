CXX=g++
INCLUDES = /Users/pujakumari/Desktop/CPPTROJAN/trojan-go/code/include
CPPFLAGS = -g -Wall -std=c++17 
EXEC=run
OBJS = trojanPlane.o gohelper.o godomain.o

encode: $(OBJS)
	$(CXX) $(CPPFLAGS) $(OBJS) -o encode

trojanPlane.o:
	$(CXX) -c $(CPPFLAGS) trojanPlane.cpp

gohelper.o:
	$(CXX) -c $(CPPFLAGS) $(INCLUDES)/gohelper.cpp

godomain.o:
	$(CXX) -c $(CPPFLAGS) ../godomain.cpp

clean:
	rm encode $(OBJS)