MKDIR=mkdir -p
BIN_DIR=bin/

all: scripts
	make -C ../lib -j8
	make -C test_correctness -j8
	make -C boxcar
	make -C gauss_anticausal -j8
	make -C gauss_cognex -j8
	make -C benchmarks -j8
	make -C ls_hist

clean:
	make -C ../lib clean
	make -C boxcar clean
	make -C ls_hist clean
	make -C gauss_anticausal clean
	make -C gauss_cognex clean
	make -C test_correctness clean
	make -C benchmarks clean
	rm -f $(BIN_DIR)/opencl_profile.sh $(BIN_DIR)/cuda_profile.sh

scripts: $(BIN_DIR)/opencl_profile.sh $(BIN_DIR)/cuda_profile.sh

$(BIN_DIR)/opencl_profile.sh: opencl_profile.sh
	@$(MKDIR) $(BIN_DIR)
	cp opencl_profile.sh $(BIN_DIR)/

$(BIN_DIR)/cuda_profile.sh: cuda_profile.sh
	@$(MKDIR) $(BIN_DIR)
	cp cuda_profile.sh $(BIN_DIR)/
