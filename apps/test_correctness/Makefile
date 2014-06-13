CXX=g++
MKDIR=mkdir -p
RM=rm -rf

BIN_DIR=../bin

SPLITLIB_DIR=../../lib/
SPLITLIB_INC=$(SPLITLIB_DIR)/split.h
SPLITLIB_LIB=$(SPLITLIB_DIR)/libSplit.a

HALIDE_LIB=$(HALIDE_DIR)/bin/libHalide.a

TESTS=$(BIN_DIR)/test_repeated_causal $(BIN_DIR)/test_repeated_anticausal $(BIN_DIR)/test_causal_anticausal $(BIN_DIR)/test_causal_xy $(BIN_DIR)/test_causal_anticausal_xy $(BIN_DIR)/test_generic_xy $(BIN_DIR)/test_generic_xyz

all: $(TESTS)

$(BIN_DIR)/test_repeated_causal: test_repeated_causal.cpp $(SPLITLIB_INC) $(HALIDE_LIB) $(SPLITLIB_LIB)
	@$(MKDIR) $(BIN_DIR)
	$(CXX) test_repeated_causal.cpp -fno-rtti -I$(SPLITLIB_DIR) -I$(HALIDE_DIR)/include/ -lz -lpthread -ldl -g -Wall $(HALIDE_LIB) $(SPLITLIB_LIB) -o $(BIN_DIR)/test_repeated_causal

$(BIN_DIR)/test_repeated_anticausal: test_repeated_anticausal.cpp $(SPLITLIB_INC) $(HALIDE_LIB) $(SPLITLIB_LIB)
	@$(MKDIR) $(BIN_DIR)
	$(CXX) test_repeated_anticausal.cpp -fno-rtti -I$(SPLITLIB_DIR) -I$(HALIDE_DIR)/include/ -lz -lpthread -ldl -g -Wall $(HALIDE_LIB) $(SPLITLIB_LIB) -o $(BIN_DIR)/test_repeated_anticausal

$(BIN_DIR)/test_causal_anticausal: test_causal_anticausal.cpp $(SPLITLIB_INC) $(HALIDE_LIB) $(SPLITLIB_LIB)
	@$(MKDIR) $(BIN_DIR)
	$(CXX) test_causal_anticausal.cpp -fno-rtti -I$(SPLITLIB_DIR) -I$(HALIDE_DIR)/include/ -lz -lpthread -ldl -g -Wall $(HALIDE_LIB) $(SPLITLIB_LIB) -o $(BIN_DIR)/test_causal_anticausal

$(BIN_DIR)/test_causal_xy: test_causal_xy.cpp $(SPLITLIB_INC) $(HALIDE_LIB) $(SPLITLIB_LIB)
	@$(MKDIR) $(BIN_DIR)
	$(CXX) test_causal_xy.cpp -fno-rtti -I$(SPLITLIB_DIR) -I$(HALIDE_DIR)/include/ -lz -lpthread -ldl -g -Wall $(HALIDE_LIB) $(SPLITLIB_LIB) -o $(BIN_DIR)/test_causal_xy

$(BIN_DIR)/test_causal_anticausal_xy: test_causal_anticausal_xy.cpp $(SPLITLIB_INC) $(HALIDE_LIB) $(SPLITLIB_LIB)
	@$(MKDIR) $(BIN_DIR)
	$(CXX) test_causal_anticausal_xy.cpp -fno-rtti -I$(SPLITLIB_DIR) -I$(HALIDE_DIR)/include/ -lz -lpthread -ldl -g -Wall $(HALIDE_LIB) $(SPLITLIB_LIB) -o $(BIN_DIR)/test_causal_anticausal_xy

$(BIN_DIR)/test_generic_xy: test_generic_xy.cpp $(SPLITLIB_INC) $(HALIDE_LIB) $(SPLITLIB_LIB)
	@$(MKDIR) $(BIN_DIR)
	$(CXX) test_generic_xy.cpp -fno-rtti -I$(SPLITLIB_DIR) -I$(HALIDE_DIR)/include/ -lz -lpthread -ldl -g -Wall $(HALIDE_LIB) $(SPLITLIB_LIB) -o $(BIN_DIR)/test_generic_xy

$(BIN_DIR)/test_generic_xyz: test_generic_xyz.cpp $(SPLITLIB_INC) $(HALIDE_LIB) $(SPLITLIB_LIB)
	@$(MKDIR) $(BIN_DIR)
	$(CXX) test_generic_xyz.cpp -fno-rtti -I$(SPLITLIB_DIR) -I$(HALIDE_DIR)/include/ -lz -lpthread -ldl -g -Wall $(HALIDE_LIB) $(SPLITLIB_LIB) -o $(BIN_DIR)/test_generic_xyz

$(SPLITLIB_LIB): $(SPLITLIB_INC)
	make -C $(SPLITLIB_DIR)

clean:
	$(RM) $(TESTS)
