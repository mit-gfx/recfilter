MAKE := make -j8 -C

all:
	$(MAKE) lib
	$(MAKE) apps
	$(MAKE) scripts

doc: lib Doxyfile
	doxygen Doxyfile

clean:
	$(MAKE) lib clean
	$(MAKE) apps clean
	$(MAKE) scripts clean
