CXX=g++ -std=c++11
MKDIR=mkdir -p
RM=rm -rf

BIN_DIR=../../bin/gauss_cognex/

LIBRECFILTER_DIR=../../lib/
LIBRECFILTER_INC=$(LIBRECFILTER_DIR)/recfilter.h
LIBRECFILTER_LIB=$(LIBRECFILTER_DIR)/libRecFilter.a

HALIDE_LIB=$(HALIDE_DIR)/bin/libHalide.a

SRC  := $(wildcard *.cpp)
APPS := $(addprefix $(BIN_DIR)/,$(notdir $(SRC:.cpp=.app)))

all: $(APPS)

$(BIN_DIR)/%.app: %.cpp  $(LIBRECFILTER_LIB) $(HALIDE_LIB) $(LIBRECFILTER_INC)
	@$(MKDIR) $(BIN_DIR)
	$(CXX) -I$(LIBRECFILTER_DIR) -I$(HALIDE_DIR)/include/ -lz -lpthread -ldl -lncurses -g -Wall $< $(HALIDE_LIB) $(LIBRECFILTER_LIB) -o $@

$(LIBRECFILTER_LIB): $(LIBRECFILTER_INC)
	make -C $(LIBRECFILTER_DIR) -j8

clean:
	$(RM) $(APPS)
