MKDIR=mkdir -p
BIN_DIR=../bin/
RM=rm -f
CP=cp -f
MAKE=make -C

all:
	$(MKDIR) $(BIN_DIR)
	$(MAKE) ../lib -j8
	$(MAKE) tests -j8
	$(MAKE) boxcar
	$(MAKE) gaussian_box -j8
	$(MAKE) gaussian_vyv -j8
	$(MAKE) benchmarks -j8
	$(MAKE) ls_hist
	$(MAKE) audio_filter
	$(MAKE) cpu
	$(MAKE) gpu
	$(CP) opencl_profile.sh $(BIN_DIR)/
	$(CP) cuda_profile.sh $(BIN_DIR)/

clean:
	$(MAKE) ../lib clean
	$(MAKE) tests clean
	$(MAKE) boxcar clean
	$(MAKE) ls_hist clean
	$(MAKE) gaussian_box clean
	$(MAKE) gaussian_vyv clean
	$(MAKE) benchmarks clean
	$(MAKE) audio_filter clean
	$(MAKE) cpu clean
	$(MAKE) gpu clean
	$(RM) $(BIN_DIR)/opencl_profile.sh $(BIN_DIR)/cuda_profile.sh
