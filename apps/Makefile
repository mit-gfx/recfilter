MKDIR	:= mkdir -p
BIN_DIR	:= ../bin
RM		:= rm -f
CP		:= cp -f
MAKE	:= make -C

SCRIPTS 	:= $(wildcard *.sh)
BIN_SCRIPTS := $(addprefix $(BIN_DIR)/,$(SCRIPTS))

all: scripts
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

clean: clean_scripts
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

$(BIN_DIR)/%.sh: %.sh
	$(CP) $< $@

scripts: $(BIN_SCRIPTS)

clean_scripts:
	$(RM) $(BIN_SCRIPTS)
