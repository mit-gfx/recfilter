APP_NAME=recursive_filter

CXX=g++
MKDIR=mkdir -p
RM=rm -rf

BUILD_DIR=_build
BIN_DIR=../bin
EXE=$(BIN_DIR)/$(APP_NAME)

SPLITLIB_DIR=../../lib/
SPLITLIB_INC=$(SPLITLIB_DIR)/split.h
SPLITLIB_LIB=$(SPLITLIB_DIR)/libSplit.a

HALIDE_LIB=$(HALIDE_DIR)/bin/libHalide.a

SRC := $(wildcard *.cpp)
OBJ := $(addprefix $(BUILD_DIR)/,$(notdir $(SRC:.cpp=.o)))
INC := $(wildcard *.h)

all: $(EXE)

$(BUILD_DIR)/%.o: %.cpp $(INC) $(SPLITLIB_INC)
	@$(MKDIR) $(BUILD_DIR)
	$(CXX) -c $< -o $@ -fno-rtti -I$(SPLITLIB_DIR) -I$(HALIDE_DIR)/include/

$(EXE): $(OBJ) $(SPLITLIB_LIB) $(HALIDE_LIB)
	@$(MKDIR) $(BIN_DIR)
	$(CXX) -lz -lpthread -ldl -g -Wall $(OBJ) $(HALIDE_LIB) $(SPLITLIB_LIB) -o $(EXE)

$(SPLITLIB_LIB): $(SPLITLIB_INC)
	make -C $(SPLITLIB_DIR)

clean:
	$(RM) $(BUILD_DIR)
	$(RM) $(EXE)
