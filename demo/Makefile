MKDIR	:= mkdir -p
RM		:= rm -f
CXX		:= g++ -std=c++11 -O3

BIN_DIR		  := ../bin/

CUDA_PATH	  := /usr/local/cuda

HALIDE_DIR 	  := ../halide/
HALIDE_LIB 	  := $(HALIDE_DIR)/bin/libHalide.a

RECFILTER_DIR := ../lib/
RECFILTER_LIB := $(RECFILTER_DIR)/libRecFilter.a

SRC  := $(wildcard *.cpp)
INC  := $(wildcard *.h)
APPS := $(addprefix $(BIN_DIR)/,$(notdir $(SRC:.cpp=)))

all: $(APPS)

CXXFLAGS := -I$(RECFILTER_DIR) -I$(HALIDE_DIR)/include/ -g -Wall
LDFLAGS  := -L$(CUDA_PATH)/nvvm/lib64 -lz -lpthread -ldl -lncurses -lnvvm -lpng

$(BIN_DIR)/%: %.cpp $(INC) $(RECFILTER_LIB) $(HALIDE_LIB)
	@$(MKDIR) $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< $(RECFILTER_LIB) $(HALIDE_LIB) $(LDFLAGS) -o $@

$(RECFILTER_LIB): $(RECFILTER_INC)
	make -C $(RECFILTER_DIR) -j8

clean:
	$(RM) $(APPS)
