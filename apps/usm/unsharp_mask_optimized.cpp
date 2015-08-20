/**
 * \file unsharp_mask_optimized.cpp
 *
 * Unsharp mask: uses Gaussian filter for blurring, can be replaced by
 * any low pass IIR filter
 *
 * UnsharpMask = (1+w)*Image - w*Blur(Image)
 *
 * This implementation is called optimized because it computes the unsharp
 * mask in the same CUDA kernel that computes the final result of blur, which
 * saves one write pass to global memory.
 */

#include <iostream>
#include <Halide.h>

#include <recfilter.h>
#include <iir_coeff.h>

using namespace Halide;

using std::vector;
using std::string;
using std::endl;
using std::cerr;

int main(int argc, char **argv) {
    Arguments args(argc, argv);

    int iter       = args.iterations;
    int tile_width = args.block;
    int width      = args.width;
    int height     = args.width;

    Image<float> image = generate_random_image<float>(width,height);

    float sigma  = 5.0f;
    float weight = 1.0f;
    vector<float> W3 = gaussian_weights(sigma,3);

    RecFilter USM("USM");
    RecFilter B  ("Blur");

    RecFilterDim x("x", width);
    RecFilterDim y("y", height);

    B.set_clamped_image_border();

    B(x,y) = image(x,y);
    B.add_filter(+x, W3);
    B.add_filter(-x, W3);
    B.add_filter(+y, W3);
    B.add_filter(-y, W3);

    vector<RecFilter> fc = B.cascade_by_dimension();

    fc[0].split_all_dimensions(tile_width);
    fc[1].split_all_dimensions(tile_width);

    // subtract the blurred image from original image
    USM(x,y) = (1.0f+weight)*image(x,y) - (weight)*fc[1](x,y);

    // merge the last stage of Gaussian blur computation with unsharp
    // mask computation -- perform this merge before GPU scheduling so
    // that appropriate can be generated
    fc[1].compute_at(USM.as_func(), Var::gpu_blocks());

    // set the bounds of USM
    // TODO: make this automatic
    USM.apply_bounds();

    // auto schedule for GPU
    RecFilter::set_max_threads_per_cuda_warp(128);
    fc[0].gpu_auto_schedule();
    fc[1].gpu_auto_schedule();
    USM  .gpu_auto_schedule(tile_width);

    USM.profile(iter);

    return EXIT_SUCCESS;
}
