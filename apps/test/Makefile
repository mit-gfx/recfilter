CXX=g++
MKDIR=mkdir -p
RM=rm -f

BIN_DIR=../bin

APP_NAME=test

EXE=$(BIN_DIR)/$(APP_NAME)

HALIDE_LIB=$(HALIDE_DIR)/bin/libHalide.a

CPP := $(wildcard *.cpp)

OBJ := $(addprefix _build/,$(notdir $(CPP:.cpp=.o)))

all: $(OBJ) $(EXE)

_build/%.o: %.cpp
	@$(MKDIR) _build
	$(CXX) -c $< -o $@ -I$(HALIDE_DIR)/include/ -I../

$(EXE): $(OBJ) $(HALIDE_LIB)
	@$(MKDIR) $(BIN_DIR)
	$(CXX) $(OBJ) $(HALIDE_LIB) -o $(EXE) -lz -lpthread -ldl -g -Wall

clean:
	$(RM) $(OBJ) $(EXE)
