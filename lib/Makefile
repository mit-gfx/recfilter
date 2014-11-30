CXX=g++ -std=c++11
MKDIR=mkdir -p
RM=rm -rf

BUILD_DIR=_build
LIB=libRecfilter.so

SRC := $(wildcard *.cpp)
OBJ := $(addprefix $(BUILD_DIR)/,$(notdir $(SRC:.cpp=.o)))
INC := $(wildcard *.h)

all: $(LIB)

$(BUILD_DIR)/%.o: %.cpp $(INC)
	@$(MKDIR) $(BUILD_DIR)
	$(CXX) -c $< -o $@ -fPIC -fno-rtti -I$(HALIDE_DIR)/include/

$(LIB): $(OBJ)
	$(CXX) -shared $(OBJ) -L$(HALIDE_DIR)/bin/ -lHalide -ldl -lz -lpthread -o $(LIB)

clean:
	$(RM) $(BUILD_DIR)
	$(RM) $(LIB)
