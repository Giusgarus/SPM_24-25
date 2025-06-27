CXX                = g++ -std=c++17
OPTFLAGS	   = -O3
AUTOFLAGS          = -march=native -ffast-math -mavx2
CXXFLAGS          += -Wall 
INCLUDES	   = -I. -I./include
LIBS               = #-pthread
SOURCES            = $(wildcard *.cpp)
TARGET             = $(SOURCES:.cpp=)

.PHONY: all clean cleanall 

%: %.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTFLAGS) -o $@ $< $(LIBS)

# For files with _avx in their name, append -mavx2 -mfma to CXXFLAGS
%_avx: CXXFLAGS += -mavx2 -mfma

all: $(TARGET)

clean: 
	-rm -fr *.o *~
cleanall: clean
	-rm -fr $(TARGET)
