CXX=g++
NVCC=nvcc
MKDIR=mkdir -p
RM=rm -rf

BIN_DIR=../../../bin/benchmarks/
APP=$(BIN_DIR)/nv_summed_area_table.app

all: $(APP)

$(APP): summed_area_table.cu
	@$(MKDIR) $(BIN_DIR)
	$(NVCC) summed_area_table.cu -o $(APP)

clean:
	$(RM) $(APP)
