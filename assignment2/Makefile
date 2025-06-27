CXX                = g++ -std=c++17
OPTFLAGS	       = -O3
CXXFLAGS          += -Wall
PARFLAGS           = 
INCLUDES	   	   = -I. -I./include
LIBS               = -pthread
SOURCES            = $(wildcard *.cpp)
TARGET             = $(SOURCES:.cpp=)

.PHONY: all clean cleanall 

%: %.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTFLAGS) -o $@ $< $(LIBS)

# For files with _par in their name, append flags to CXXFLAGS
%_par: CXXFLAGS += ${PARFLAGS}

all: $(TARGET)

clean: 
	-rm -fr *.o *~

cleanall: clean
	-rm -fr $(TARGET)
