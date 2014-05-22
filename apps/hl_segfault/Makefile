PROGRAM_NAME=hl_segfault

CXX=g++
MKDIR=mkdir -p
RM=rm -rf

BUILD_DIR=_build
EXE=$(PROGRAM_NAME)

HALIDE_LIB=$(HALIDE_DIR)/bin/libHalide.a

SRC := $(wildcard *.cpp)
OBJ := $(addprefix $(BUILD_DIR)/,$(notdir $(SRC:.cpp=.o)))
INC := $(wildcard *.h)

all: $(EXE)

$(BUILD_DIR)/%.o: %.cpp $(INC)
	@$(MKDIR) $(BUILD_DIR)
	$(CXX) -c $< -o $@ -fno-rtti -I$(HALIDE_DIR)/include/

$(EXE): $(OBJ) $(HALIDE_LIB)
	$(CXX) -lz -lpthread -ldl -g -Wall $(OBJ) $(HALIDE_LIB) -o $(EXE)

clean:
	$(RM) $(BUILD_DIR)
	$(RM) $(EXE)
