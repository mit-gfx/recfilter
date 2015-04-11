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
	$(MAKE) cpu
	$(MAKE) gpu
	$(MAKE) ls_hist

clean:
	$(MAKE) ../lib 		 clean
	$(MAKE) tests 		 clean
	$(MAKE) box 	 	 clean
	$(MAKE) bspline 	 clean
	$(MAKE) gaussian 	 clean
	$(MAKE) DoG			 clean
	$(MAKE) summed_table clean
	$(MAKE) audio  		 clean
	$(MAKE) cpu 		 clean
	$(MAKE) gpu 		 clean
	$(MAKE) ls_hist 	 clean
