CXX=g++
MKDIR=mkdir -p
RM=rm -rf

BIN_DIR=../bin

EXE=$(BIN_DIR)/gauss_cognex_overlapped $(BIN_DIR)/gauss_cognex_direct $(BIN_DIR)/gauss_cognex_cascaded

SPLITLIB_DIR=../../lib/
SPLITLIB_INC=$(SPLITLIB_DIR)/split.h
SPLITLIB_LIB=$(SPLITLIB_DIR)/libSplit.a

HALIDE_LIB=$(HALIDE_DIR)/bin/libHalide.a

SRC := $(wildcard *.cpp)
INC := $(wildcard *.h)

all: $(EXE)

$(BIN_DIR)/gauss_cognex_overlapped: gauss_cognex_overlapped.cpp $(SPLITLIB_INC) $(HALIDE_LIB) $(SPLITLIB_LIB)
	@$(MKDIR) $(BIN_DIR)
	$(CXX) gauss_cognex_overlapped.cpp -fno-rtti -I$(SPLITLIB_DIR) -I$(HALIDE_DIR)/include/ -lz -lpthread -ldl -g -Wall $(HALIDE_LIB) $(SPLITLIB_LIB) -o $(BIN_DIR)/gauss_cognex_overlapped

$(BIN_DIR)/gauss_cognex_direct: gauss_cognex_direct.cpp $(SPLITLIB_INC) $(HALIDE_LIB) $(SPLITLIB_LIB)
	@$(MKDIR) $(BIN_DIR)
	$(CXX) gauss_cognex_direct.cpp -fno-rtti -I$(SPLITLIB_DIR) -I$(HALIDE_DIR)/include/ -lz -lpthread -ldl -g -Wall $(HALIDE_LIB) $(SPLITLIB_LIB) -o $(BIN_DIR)/gauss_cognex_direct

$(BIN_DIR)/gauss_cognex_cascaded: gauss_cognex_cascaded.cpp $(SPLITLIB_INC) $(HALIDE_LIB) $(SPLITLIB_LIB)
	@$(MKDIR) $(BIN_DIR)
	$(CXX) gauss_cognex_cascaded.cpp -fno-rtti -I$(SPLITLIB_DIR) -I$(HALIDE_DIR)/include/ -lz -lpthread -ldl -g -Wall $(HALIDE_LIB) $(SPLITLIB_LIB) -o $(BIN_DIR)/gauss_cognex_cascaded

$(SPLITLIB_LIB): $(SPLITLIB_INC)
	make -C $(SPLITLIB_DIR)

clean:
	$(RM) $(EXE)
