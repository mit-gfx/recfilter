## Domain specific language for recursive filters

This a domain-specific language based on [Halide](http://halide-lang.org) that allows easy
implementation of recursive or IIR filters for n-dimensional data.

### Requirements
- Ubuntu 14.04, Fedora 20 and Mac OSX
- our spiked version of [Halide](https://github.com/gchauras/Halide)
    - included as submodule, it should download and build automatically upon running `make`
    - uses NVIDIA's nvvm backend for CUDA ptx generation rather than the open source ptx backend
    - other minor modifications
- Halide requirements: [llvm](http://llvm.org/), [clang](http://clang.llvm.org/) (see [Halide building instructions](https://github.com/halide/Halide))
- [NVIDIA CUDA toolkit 7](https://developer.nvidia.com/cuda-toolkit) (version 6.0 or 6.5 will not suffice)

The makefile in the base directory should build everything and place the executables in `bin`.

### Directory structure
```
$(RECFILTER_DIR)
 |- halide/     (Halide submodule - cloned automatically on running make)
 |- lib/        (RecFilter library)
 |- apps/       (benchmarking applications)
 |- tests/      (tests to check the correctness of tiling algebra)
 |- demos/      (application demos)
 |- gpu/        (CUDA benchmarks from NVIDIA toolkit and Thrust)
```

### See also:
- [Tests](https://github.com/mit-gfx/recfilter/tree/master/tests)
- [Benchmarking applications](https://github.com/mit-gfx/recfilter/tree/master/apps)
- [Demos](https://github.com/mit-gfx/recfilter/tree/master/demos)


### TODO list:
- Change number of threads to a global constant instead of specifying
  it for each auto GPU scheduling command.
- Change vectorization width to a global constant instead of specifying
  it for each auto CPU scheduling command.
- Allow initializing ``RecFilter`` from another ``RecFilter`` directly, i.e.
allow ``R(x,y) = S(x,y)``, currently this is done using ``R(x,y) = S.as_func()(x,y)``.
- Provide better interoperability between ``Halide::Var`` and ``RecFilter::RecFilterDim``.
- Provide better semantics for color images, color channels must be specified
as a ``Halide::Tuple`` e.g. ``R(x,y) = Tuple(red,green,blur)`` which is not intuitive.
- Allow support for tiling of ``RecFilter`` which do not have any scans, currently
this is done via a non-intuitive scheduling command.
- Change the ``RecFilter::compute_at`` interface to something cleaner for merging
a recursive filter with a downstream operation.
