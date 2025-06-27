# 
# FF_ROOT     points to the FastFlow root directory (i.e.
#             the one containing the ff directory).
ifndef FF_ROOT 
FF_ROOT		= ./ #${HOME}/fastflow
endif

CXX                = mpic++ -std=c++17
MAINOPT:= 			-DRPAYLOAD=$(RPAYLOAD)
OPTFLAGS	   = -O3 -march=native
CXXFLAGS          +=  -Wall #-DNO_DEFAULT_MAPPING #-DBLOCKING_MODE
                      #-DFF_BOUNDED_BUFFER
INCLUDES	   = -I$(FF_ROOT)
LIBS               = -pthread -lX11
MAIN_TARGETS       = par_merge plain_sort
PSRS_TARGETS      = par_merge_psrs
GEN_TARGET         = gen_dataset

.PHONY: all clean cleanall clean-%

all: $(MAIN_TARGETS)

psrs: $(PSRS_TARGETS)

gen: $(GEN_TARGET)

$(MAIN_TARGETS): %: %.cpp
	$(CXX) $(INCLUDES) $(MAINOPT) $(CXXFLAGS) $(OPTFLAGS) -o $@ $< $(LIBS)

$(PSRS_TARGETS): %: %.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTFLAGS) -o $@ $< $(LIBS)

$(GEN_TARGET): %: %.cpp
	mpic++ -std=c++17 -O3 -march=native -Wall -o $@ $< 

nodisplay: CXXFLAGS += -DNO_DISPLAY
nodisplay: cleanall $(MAIN_TARGETS)

clean-%:
	-rm -f $*
cleanall: clean
	-rm -f $(MAIN_TARGETS) $(GEN_TARGET) $(PSRS_TARGETS)
