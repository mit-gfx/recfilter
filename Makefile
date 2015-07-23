MAKE := make -j8 -C

all:
	$(MAKE) lib
	$(MAKE) apps
	$(MAKE) tests
	$(MAKE) scripts

doc: lib Doxyfile
	doxygen Doxyfile

clean:
	$(MAKE) lib 	clean
	$(MAKE) apps 	clean
	$(MAKE) tests	clean
	$(MAKE) scripts clean
