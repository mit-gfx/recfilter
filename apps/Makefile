MAKE := make -j8 -C

all:
	$(MAKE) ../lib
	$(MAKE) tests
	$(MAKE) box
	$(MAKE) bspline
	$(MAKE) gaussian
	$(MAKE) summed_table
	$(MAKE) ls_hist
	$(MAKE) audio
	$(MAKE) cpu
	$(MAKE) gpu

clean:
	$(MAKE) ../lib 		 clean
	$(MAKE) tests 		 clean
	$(MAKE) box 	 	 clean
	$(MAKE) bspline 	 clean
	$(MAKE) ls_hist 	 clean
	$(MAKE) gaussian 	 clean
	$(MAKE) summed_table clean
	$(MAKE) audio  		 clean
	$(MAKE) cpu 		 clean
	$(MAKE) gpu 		 clean
