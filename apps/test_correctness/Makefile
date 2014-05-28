CXX=g++
MKDIR=mkdir -p
RM=rm -rf

BIN_DIR=../bin

SPLITLIB_DIR=../../lib/
SPLITLIB_INC=$(SPLITLIB_DIR)/split.h
SPLITLIB_LIB=$(SPLITLIB_DIR)/libSplit.a

HALIDE_LIB=$(HALIDE_DIR)/bin/libHalide.a

TESTS=$(BIN_DIR)/test_repeated_causal

all: $(TESTS)

$(BIN_DIR)/test_repeated_causal: test_repeated_causal.cpp $(INC) $(SPLITLIB_INC)
	@$(MKDIR) $(BIN_DIR)
	$(CXX) test_repeated_causal.cpp -fno-rtti -I$(SPLITLIB_DIR) -I$(HALIDE_DIR)/include/ -lz -lpthread -ldl -g -Wall $(HALIDE_LIB) $(SPLITLIB_LIB) -o $(BIN_DIR)/test_repeated_causal

$(SPLITLIB_LIB):
	make -C $(SPLITLIB_DIR)

clean:
	$(RM) $(TESTS)
