all:
	make -C ../lib
	make -C test_correctness -j8
	make -C boxcar
	make -C summed_table
	make -C gauss_anticausal -j8
	make -C gauss_cognex -j8
	make -C ls_hist


clean:
	make -C ../lib clean
	make -C summed_table clean
	make -C boxcar clean
	make -C ls_hist clean
	make -C gauss_anticausal clean
	make -C gauss_cognex clean
	make -C test_correctness clean
