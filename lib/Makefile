CXX=g++
MKDIR=mkdir -p
RM=rm -rf

BUILD_DIR=_build
LIB=libRecfilter.a

HALIDE_LIB=$(HALIDE_DIR)/bin/libHalide.a

SRC := $(wildcard *.cpp)
OBJ := $(addprefix $(BUILD_DIR)/,$(notdir $(SRC:.cpp=.o)))
INC := $(wildcard *.h)

all: $(LIB)

$(BUILD_DIR)/%.o: %.cpp $(INC)
	@$(MKDIR) $(BUILD_DIR)
	$(CXX) -c $< -o $@ -fno-rtti -I$(HALIDE_DIR)/include/

$(LIB): $(OBJ)
	$(RM) $(LIB)
	ar q $(LIB) $(OBJ)
	ranlib $(LIB)

clean:
	$(RM) $(BUILD_DIR)
	$(RM) $(LIB)
