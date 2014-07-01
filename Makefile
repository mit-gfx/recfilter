all:
	make -C lib -j8
	make -C apps

clean:
	make -C lib clean
	make -C apps clean
