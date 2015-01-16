MKDIR	:= mkdir -p
BIN_DIR	:= ../bin
RM		:= rm -f
CP		:= cp -f

SCRIPTS 	:= $(wildcard *.sh)
BIN_SCRIPTS := $(addprefix $(BIN_DIR)/,$(SCRIPTS))

$(BIN_DIR)/%.sh: %.sh
	$(MKDIR) $(BIN_DIR)
	$(CP) $< $@

all: $(BIN_SCRIPTS)

clean:
	$(RM) $(BIN_SCRIPTS)
