MAKE := make -j8 -C

all:
	$(MAKE) ../lib
	$(MAKE) tests
	$(MAKE) box
	$(MAKE) bspline
	$(MAKE) gaussian
	$(MAKE) summed_table
	$(MAKE) DoG
	$(MAKE) audio

clean:
	$(MAKE) ../lib 		 clean
	$(MAKE) tests 		 clean
	$(MAKE) box 	 	 clean
	$(MAKE) bspline 	 clean
	$(MAKE) gaussian 	 clean
	$(MAKE) DoG			 clean
	$(MAKE) summed_table clean
	$(MAKE) audio  		 clean
