CXX         = g++ -std=c++20
INCLUDES    = -I . -I ./miniz -I ./include
CXXFLAGS    += -Wall 

# Flags separati per i due target
SEQ_OPTFLAGS    = -O3
PAR_OPTFLAGS    = -O3
PAR_LDFLAGS     = -pthread -fopenmp

TARGETS     = miniz_plain miniz_parallel

.PHONY: all clean cleanall
.SUFFIXES: .cpp

all: $(TARGETS)

miniz_plain: miniz_plain.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SEQ_OPTFLAGS) -o $@ $< ./miniz/miniz.c 

miniz_parallel: miniz_parallel.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(PAR_OPTFLAGS) -o $@ $< ./miniz/miniz.c $(PAR_LDFLAGS)





clean:
	rm -f $(TARGETS)

cleanall: clean
	rm -f *.o *~