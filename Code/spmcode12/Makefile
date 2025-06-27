FF_ROOT            = ~/fastflow
CXX                = mpicxx -std=c++20
OPTFLAGS	   = -O3
CXXFLAGS          += -Wall 
INCLUDES	   = -I. -I./include
LIBS               = 
SOURCES            = $(wildcard *.cpp)
TARGET             = $(SOURCES:.cpp=)

.PHONY: all clean cleanall 

%: %.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTFLAGS) -o $@ $< $(LIBS)

all: $(TARGET)

summa_omp:summa_omp.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTFLAGS) -o $@ $< $(LIBS) -fopenmp

trapezoid_mpi+omp: trapezoid_mpi+omp.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTFLAGS) -o $@ $< $(LIBS) -fopenmp

trapezoid_mpi+ff: trapezoid_mpi+ff.cpp
	$(CXX) $(INCLUDES) -I $(FF_ROOT) $(CXXFLAGS) $(OPTFLAGS) -o $@ $< $(LIBS) -pthread

Farm+ff_farm: Farm+ff_farm.cpp
	$(CXX) $(INCLUDES) -I $(FF_ROOT) $(CXXFLAGS) $(OPTFLAGS) -o $@ $< $(LIBS) -pthread



clean: 
	-rm -fr *.o *~
cleanall: clean
	-rm -fr $(TARGET)
	-rm -f *.txt


