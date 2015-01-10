MKDIR	:= mkdir -p
BIN_DIR	:= ../bin
RM		:= rm -f
CP		:= cp -f
MAKE	:= make -j8 -C

SCRIPTS 	:= $(wildcard *.sh)
BIN_SCRIPTS := $(addprefix $(BIN_DIR)/,$(SCRIPTS))

all: scripts
	$(MKDIR) $(BIN_DIR)
	$(MAKE) ../lib
	$(MAKE) tests
	$(MAKE) box_filter
	$(MAKE) bspline_interp
	$(MAKE) gaussian_vyv
	$(MAKE) benchmarks
	$(MAKE) ls_hist
	$(MAKE) audio_filter
	$(MAKE) cpu
	$(MAKE) gpu

clean: clean_scripts
	$(MAKE) ../lib 		 clean
	$(MAKE) tests 		 clean
	$(MAKE) box_filter 	 clean
	$(MAKE) bspline_interp clean
	$(MAKE) ls_hist 	 clean
	$(MAKE) gaussian_vyv clean
	$(MAKE) benchmarks 	 clean
	$(MAKE) audio_filter clean
	$(MAKE) cpu 		 clean
	$(MAKE) gpu 		 clean

$(BIN_DIR)/%.sh: %.sh
	$(MKDIR) $(BIN_DIR)
	$(CP) $< $@

scripts: $(BIN_SCRIPTS)

clean_scripts:
	$(RM) $(BIN_SCRIPTS)
