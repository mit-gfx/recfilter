# Recursive filtering using Halide

## Compiling and running
The [Halide](https://github.com/gchauras/Halide) version required for compiling this library is a submodule and it should download automatically.

The makefile in <code>code/</code> should build everything and place the executables in <code>code/bin</code>.

## Directory structure
```
$(RECFILTER_DIR)
 |- halide/           (Halide submodule)
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
