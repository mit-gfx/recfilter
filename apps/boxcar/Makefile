CXX=g++ -std=c++11
MKDIR=mkdir -p
RM=rm -rf

BIN_DIR=../bin/boxcar/

LIBRECFILTER_DIR=../../lib/
LIBRECFILTER_INC=$(LIBRECFILTER_DIR)/recfilter.h
LIBRECFILTER_LIB=$(LIBRECFILTER_DIR)/libRecfilter.so

HALIDE_LIB=$(HALIDE_DIR)/bin/libHalide.so

SRC  := $(wildcard *.cpp)
APPS := $(addprefix $(BIN_DIR)/,$(notdir $(SRC:.cpp=.app)))

all: $(APPS)

$(BIN_DIR)/%.app: %.cpp  $(LIBRECFILTER_LIB) $(HALIDE_LIB) $(LIBRECFILTER_INC)
	@$(MKDIR) $(BIN_DIR)
	$(CXX) $< -o $@ -I$(LIBRECFILTER_DIR) -I$(HALIDE_DIR)/include/ -L$(LIBRECFILTER_DIR) -L$(HALIDE_DIR)/bin/ -lz -lpthread -ldl -g -lncurses -Wall -lHalide -lRecfilter

$(LIBRECFILTER_LIB): $(LIBRECFILTER_INC)
	make -C $(LIBRECFILTER_DIR)

clean:
	$(RM) $(APPS)
