CXX=g++
MKDIR=mkdir -p
RM=rm -rf

BIN_DIR=../bin

EXE=$(BIN_DIR)/gauss_anticausal_direct $(BIN_DIR)/gauss_anticausal_cascaded

SPLITLIB_DIR=../../lib/
SPLITLIB_INC=$(SPLITLIB_DIR)/recfilter.h
SPLITLIB_LIB=$(SPLITLIB_DIR)/libSplit.a

HALIDE_LIB=$(HALIDE_DIR)/bin/libHalide.a

SRC := $(wildcard *.cpp)
INC := $(wildcard *.h)

all: $(EXE)

$(BIN_DIR)/gauss_anticausal_direct: gauss_anticausal_direct.cpp $(SPLITLIB_INC) $(HALIDE_LIB) $(SPLITLIB_LIB)
	@$(MKDIR) $(BIN_DIR)
	$(CXX) gauss_anticausal_direct.cpp -fno-rtti -I$(SPLITLIB_DIR) -I$(HALIDE_DIR)/include/ -lz -lpthread -ldl -g -Wall $(HALIDE_LIB) $(SPLITLIB_LIB) -o $(BIN_DIR)/gauss_anticausal_direct

$(BIN_DIR)/gauss_anticausal_cascaded: gauss_anticausal_cascaded.cpp $(SPLITLIB_INC) $(HALIDE_LIB) $(SPLITLIB_LIB)
	@$(MKDIR) $(BIN_DIR)
	$(CXX) gauss_anticausal_cascaded.cpp -fno-rtti -I$(SPLITLIB_DIR) -I$(HALIDE_DIR)/include/ -lz -lpthread -ldl -g -Wall $(HALIDE_LIB) $(SPLITLIB_LIB) -o $(BIN_DIR)/gauss_anticausal_cascaded

$(SPLITLIB_LIB): $(SPLITLIB_INC)
	make -C $(SPLITLIB_DIR)

clean:
	$(RM) $(EXE)
