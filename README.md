# Recursive filtering using Halide

## Requirements
- tested on Ubuntu 14.04 and Fedora 20, partially tested on Mac OSX
- our spiked version of [Halide](https://github.com/gchauras/Halide)
    - included as submodule, should download and build automatically upon running `make`
    - uses NVIDIA's nvvm backend for CUDA ptx generation rather than the open source ptx backend
    - other minor modifications
- ususal requirements for Halide: [llvm](http://llvm.org/), [clang](http://clang.llvm.org/) (see [Halide building instructions](https://github.com/halide/Halide))
- [NVIDIA CUDA toolkit 7](https://developer.nvidia.com/cuda-toolkit) (version 6.0 or 6.5 do not suffice)

## Compilation
The makefile in the base directory should build everything and place the executables in `bin`.

## Directory structure
```
$(RECFILTER_DIR)
 |- halide/           (Halide submodule - pulls automatically on running make)
 |- lib/              (RecFilter library)
 |- apps/             (benchmarking applications)
     |- gpu/          (CUDA benchmarks from NVIDIA toolkit and Thrust)
     |- summed_table/ (summed area table)
     |- box/          (iterated box filters)
     |- gaussian/     (Vliet-Young-Verbeek approxmiation of Gaussian blur)
     |- bspline/      (bicubic and biquintic b-spline filters)
     |- ls_hist/      (smoothed local histograms)
     |- audio_filter/ (high order 1D IIR filters used for audio processing)
 |-demos/             (application demos)
     |- gaussian/     (RGB Gaussian blur)
```
