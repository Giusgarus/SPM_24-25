CXX                = nvcc -std=c++17 -x cu
OPTFLAGS	   = -O3
CXXFLAGS          += 
INCLUDES	   = -I . 
SOURCES            = $(wildcard *.cpp)
TARGET             = $(SOURCES:.cpp=)

.PHONY: all clean cleanall 

%: %.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTFLAGS) -o $@ $< $(LIBS)

# For files with _avx in their name
%_avx: CXX = g++ -std=c++17
%_avx: CXXFLAGS += -march=native


all: $(TARGET)

clean: 
	-rm -fr *.o *~
cleanall: clean
	-rm -fr $(TARGET)
