all:
	make -C lib -j12
	make -C apps -j12

clean:
	make -C lib clean
	make -C apps clean

