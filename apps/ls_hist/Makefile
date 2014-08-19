CXX=g++
MKDIR=mkdir -p
RM=rm -rf

BIN_DIR=../bin/local_smooth_hist

LIBRECFILTER_DIR=../../lib/
LIBRECFILTER_INC=$(LIBRECFILTER_DIR)/recfilter.h
LIBRECFILTER_LIB=$(LIBRECFILTER_DIR)/libRecfilter.a

HALIDE_LIB=$(HALIDE_DIR)/bin/libHalide.a

SRC  := $(wildcard *.cpp)
APPS := $(addprefix $(BIN_DIR)/,$(notdir $(SRC:.cpp=.app)))

all: $(APPS)

$(BIN_DIR)/%.app: %.cpp  $(LIBRECFILTER_LIB) $(HALIDE_LIB) $(LIBRECFILTER_INC)
	@$(MKDIR) $(BIN_DIR)
	$(CXX) $< -o $@ -fno-rtti -I$(LIBRECFILTER_DIR) -I$(HALIDE_DIR)/include/ -lz -lpthread -ldl -g -Wall $(HALIDE_LIB) $(LIBRECFILTER_LIB)

$(LIBRECFILTER_LIB): $(LIBRECFILTER_INC)
	make -C $(LIBRECFILTER_DIR)

clean:
	$(RM) $(APPS)
