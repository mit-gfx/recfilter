CXX=g++ -std=c++11
MKDIR=mkdir -p
RM=rm -rf

BUILD_DIR=_build
LIB=libRecFilter.a

SRC := $(wildcard *.cpp)
OBJ := $(addprefix $(BUILD_DIR)/,$(notdir $(SRC:.cpp=.o)))
INC := $(wildcard *.h)

all: $(LIB)

$(BUILD_DIR)/%.o: %.cpp $(INC)
	@$(MKDIR) $(BUILD_DIR)
	$(CXX) -fPIC -fno-rtti -I$(HALIDE_DIR)/include/ -c $< -o $@

$(LIB): $(OBJ)
	$(RM) $(LIB)
	ar q $(LIB) $(OBJ)
	ranlib $(LIB)

clean:
	$(RM) $(BUILD_DIR)
	$(RM) $(LIB)
