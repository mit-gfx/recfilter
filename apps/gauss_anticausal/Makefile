CXX=g++
MKDIR=mkdir -p
RM=rm -rf

BIN_DIR=../bin

SPLITLIB_DIR=../../lib/
SPLITLIB_INC=$(SPLITLIB_DIR)/recfilter.h
SPLITLIB_LIB=$(SPLITLIB_DIR)/libSplit.a

HALIDE_LIB=$(HALIDE_DIR)/bin/libHalide.a

SRC  := $(wildcard *.cpp)
APPS := $(addprefix $(BIN_DIR)/,$(notdir $(SRC:.cpp=.app)))

all: $(APPS)

$(BIN_DIR)/%.app: %.cpp  $(SPLITLIB_LIB) $(HALIDE_LIB) $(SPLITLIB_INC)
	@$(MKDIR) $(BIN_DIR)
	$(CXX) $< -o $@ -fno-rtti -I$(SPLITLIB_DIR) -I$(HALIDE_DIR)/include/ -lz -lpthread -ldl -g -Wall $(HALIDE_LIB) $(SPLITLIB_LIB)

$(SPLITLIB_LIB): $(SPLITLIB_INC)
	make -C $(SPLITLIB_DIR)

clean:
	$(RM) $(APPS)
