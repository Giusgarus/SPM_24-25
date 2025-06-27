CXX                = mpicxx -std=c++20
OPTFLAGS	   = -O3
CXXFLAGS          += -Wall 
INCLUDES	   = -I. -I./include
LIBS               = 
SOURCES            = $(wildcard *.cpp)
TARGET             = $(SOURCES:.cpp=)
FFTW_INC           = `pkgconf --cflags /opt/ohpc/pub/libs/gnu12/openmpi4/fftw/3.3.10/lib/pkgconfig/fftw3.pc`
FFTW_LIB           = `pkgconf --libs /opt/ohpc/pub/libs/gnu12/openmpi4/fftw/3.3.10/lib/pkgconfig/fftw3.pc`
FFTW_LIB_DIR       = `pkgconf --variable=libdir /opt/ohpc/pub/libs/gnu12/openmpi4/fftw/3.3.10/lib/pkgconfig/fftw3.pc`

.PHONY: all clean cleanall 

%: %.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTFLAGS) -o $@ $< $(LIBS)

all: $(TARGET)

# compile on the spmcluster
fft1d: fft1d.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(FFTW_INC) $(OPTFLAGS) -o $@ $< $(LIBS) $(FFTW_LIB) -Wl,-rpath,$(FFTW_LIB_DIR)

clean: 
	-rm -fr *.o *~
cleanall: clean
	-rm -fr $(TARGET)



