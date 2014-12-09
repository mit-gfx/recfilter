CXX=g++ -std=c++11
NVCC=nvcc
MKDIR=mkdir -p
RM=rm -rf

BIN_DIR=../../../bin/gpu/
APP=$(BIN_DIR)/nvSummedTable

all: $(APP)

$(APP): summedTableThrust.cu
	@$(MKDIR) $(BIN_DIR)
	$(NVCC) $< -o $@

clean:
	$(RM) $(APP)
