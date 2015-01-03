CXX=g++ -std=c++11
MKDIR=mkdir -p
RM=rm -rf

BUILD_DIR	  := _build

HALIDE_DIR	  := ../halide
HALIDE_INC 	  := $(HALIDE_DIR)/include/Halide.h
HALIDE_MAKE   := $(HALIDE_DIR)/Makefile

RECFILTER_DIR := ../lib
RECFILTER_LIB := libRecFilter.a
RECFILTER_SRC := $(wildcard *.cpp)
RECFILTER_OBJ := $(addprefix $(BUILD_DIR)/,$(notdir $(RECFILTER_SRC:.cpp=.o)))
RECFILTER_INC := $(wildcard *.h)

all: $(RECFILTER_LIB)

$(RECFILTER_LIB): $(RECFILTER_OBJ)
	$(RM) $(RECFILTER_LIB)
	ar q $(RECFILTER_LIB) $(RECFILTER_OBJ)
	ranlib $(RECFILTER_LIB)

$(HALIDE_INC): $(HALIDE_SRC)
	make -C $(HALIDE_DIR) -j12

$(HALIDE_SRC): halide_clone

halide_clone:
	cd $(HALIDE_DIR)
	git submodule init
	git submodule update
	cd $(RECFILTER_DIR)

$(BUILD_DIR)/%.o: %.cpp $(RECFILTER_INC) $(HALIDE_INC)
	@$(MKDIR) $(BUILD_DIR)
	$(CXX) -fPIC -fno-rtti -I$(HALIDE_DIR)/include/ -c $< -o $@

clean:
	$(RM) $(BUILD_DIR)
	$(RM) $(RECFILTER_LIB)
