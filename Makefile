all:
	make -C lib -j8
	make -C apps

doc: lib Doxyfile
	doxygen Doxyfile

clean:
	make -C lib clean
	make -C apps clean
