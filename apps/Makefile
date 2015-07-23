MAKE := make -j8 -C

all:
	$(MAKE) ../lib
	$(MAKE) box
	$(MAKE) bspline
	$(MAKE) gaussian
	$(MAKE) summed_table
	$(MAKE) DoG
	$(MAKE) usm
	$(MAKE) audio
	$(MAKE) cpu
	$(MAKE) gpu

clean:
	$(MAKE) ../lib 		 clean
	$(MAKE) box 	 	 clean
	$(MAKE) bspline 	 clean
	$(MAKE) gaussian 	 clean
	$(MAKE) DoG			 clean
	$(MAKE) usm 		 clean
	$(MAKE) summed_table clean
	$(MAKE) audio  		 clean
	$(MAKE) cpu 		 clean
	$(MAKE) gpu 		 clean
