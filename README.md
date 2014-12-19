# Recursive filtering using Halide

## Compiling and running
You can obtain the Halide version by forking the modified [Halide](https://github.com/gchauras/Halide) repo.

This modified version of the original [Halide](https://github.com/halide/Halide) repo is synced with the original Halide repo once a week. So you are not missing any cool Halide features.

Environment variable <code>HALIDE_DIR</code> must point to the base of Halide directory that contains include and lib directories. The makefile in <code>code/</code> should build everything and place the executables in <code>code/bin</code>.

## Directory structure
```
$(RECFILTER_DIR)
 |- lib/              (RecFilter library)
 |- apps/
     |- cpu/          (CPU benchmarks in C/C++ with OpenMP)
     |- opencl/       (GPU benchmarks in OpenCL)
     |- gpu/          (GPU benchmarks in NVIDIA CUDA and NVIDIA Thrust)
     |- summed_table/ (RecFilter implementation of summed area table)
     |- boxcar/       (RecFilter implementation of box filter)
     |- gaussian_vyv/ (RecFilter implementation of Vliet-Young-Verbeek approxmiation of Gaussian blur)
     |- gaussian_box/ (RecFilter implementation of box filter approximation for Gaussian blur)
     |- ls_hist/      (RecFilter implementation of median filter using smoothed local histograms)
     |- audio_filter/ (RecFilter implementation of very high order 1D audio filters)
```
