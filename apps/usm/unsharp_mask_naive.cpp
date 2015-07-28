/**
 * \file unsharp_mask_naive.cpp
 *
 * Unsharp mask: uses Gaussian filter for blurring, can be replaced by
 * any low pass IIR filter
 *
 * UnsharpMask = (1+w)*Image - w*Blur(Image)
 *
 * This implementation is called naive because it computes the full result
 * of the blur and then computes unsharp mask in a subsequent pass, hence
 * it involves one extra write to global memory.
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

    // auto schedule for GPU
    fc[0].gpu_auto_schedule(128);
    fc[1].gpu_auto_schedule(128);
    USM  .gpu_auto_schedule(128, tile_width);

    USM.profile(iter);

    return EXIT_SUCCESS;
}
