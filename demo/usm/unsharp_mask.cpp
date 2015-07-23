/**
 * \file unsharp_mask.cpp
 *
 * Unsharp mask: uses Gaussian filter for blurring, can be replaced by
 * any low pass IIR filter
 *
 * UnsharpMask = Image + w*HighFreq;
 *             = Image + w*(Image - Blur(Image))
 *             = (1+w)*Image - w*Blur(Image)
 */

#include <iostream>
#include <Halide.h>

#include "iir_coeff.h"
#include "recfilter.h"

using namespace Halide;

using std::vector;

int main(int argc, char **argv) {
    Arguments args(argc, argv);

    int iter       = args.iterations;
    int tile_width = args.block;
    int in_w       = args.width;

    int width = in_w;
    int height= in_w;

    Image<float> image = generate_random_image<float>(width,height);

    RecFilterDim x("x", width);
    RecFilterDim y("y", height);

    float sigma  = 5.0f;
    float weight = 1.0f;
    vector<float> W3 = gaussian_weights(sigma,3);

    RecFilter USM ("UnsharpMask");
    RecFilter Blur("Blur");

    // some low pass blur
    {
        RecFilter B("GaussianBlur");
        B.set_clamped_image_border();

        B(x,y) = image(x,y);
        B.add_filter(+x, W3);
        B.add_filter(-x, W3);
        B.add_filter(+y, W3);
        B.add_filter(-y, W3);

        // tile and schedule blur
        vector<RecFilter> fc = B.cascade_by_dimension();
        for (int i=0; i<fc.size(); i++) {
            fc[i].split_all_dimensions(tile_width);
            fc[i].gpu_auto_schedule(128);
        }

        // extract the result of Blur
        Blur = fc[fc.size()-1];
    }

    // subtract the blurred image from original image
    {
        USM(x,y) = (1.0f+weight)*image(x,y) - (weight)*Blur(x,y);

        USM.gpu_auto_schedule(128, tile_width);

        // make sure that Blur result and USM happen in the same CUDA kernel
        Blur.compute_at(USM.as_func(), Var::gpu_blocks());

        std::cerr << USM.print_schedule() << Blur.print_schedule() << std::endl;
    }

    USM.compile_jit("USM.html");
    USM.profile(iter);

    return EXIT_SUCCESS;
}
