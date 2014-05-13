all:
	make -C lib
	make -C apps

clean:
	make -C lib clean
	make -C apps clean

