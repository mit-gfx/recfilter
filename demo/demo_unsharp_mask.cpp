/**
 * \file demo_unsharp_mask.cpp
 *
 * Unsharp mask: uses Gaussian filter for blurring, can be replaced by
 * any low pass IIR filter
 *
 * UnsharpMask = (1+w)*Image - w*Blur(Image)
 */

#include <iostream>
#include <Halide.h>

#include <recfilter.h>
#include <iir_coeff.h>

#include "image_io.h"

using namespace Halide;

using std::vector;
using std::string;
using std::endl;
using std::cerr;

int main(int argc, char **argv) {
//    string filename;
//    if (argc==2) {
//        filename = argv[1];
//    } else {
//        cerr << "Usage: unsharp_mask_demo [name of png file]" << endl;
//        return EXIT_FAILURE;
//    }
//
//    Image<uint8_t> input = load<uint8_t>(filename);
//
//    int width   = input.width();
//    int height  = input.height();
//    int channels= input.channels();

    float sigma  = 5.0f;
    float weight = 1.0f;
    vector<float> W3 = gaussian_weights(sigma,3);

    Arguments args(argc, argv);
    int iter       = args.iterations;
    int tile_width = args.block;
    int in_w       = args.width;
    int width      = in_w;
    int height     = in_w;
    Image<float> image = generate_random_image<float>(width,height);

    // some low pass blur
    {
        RecFilter USM("USM");
        RecFilter B  ("Blur");

        RecFilterDim x("x", width);
        RecFilterDim y("y", height);

//    Expr r = cast<float>(input(x,y,0)) / 255.0f;
//    Expr g = cast<float>(input(x,y,1)) / 255.0f;
//    Expr b = cast<float>(input(x,y,2)) / 255.0f;

        B.set_clamped_image_border();

        B(x,y) = image(x,y);
        B.add_filter(+x, W3);
        B.add_filter(-x, W3);
        B.add_filter(+y, W3);
        B.add_filter(-y, W3);

        // tile and schedule blur
        vector<RecFilter> fc = B.cascade_by_dimension();

        fc[0].split_all_dimensions(tile_width);
        fc[1].split_all_dimensions(tile_width);

        // subtract the blurred image from original image
        USM(x,y) = (1.0f+weight)*image(x,y) - (weight)*fc[1](x,y);

        // tile the dimensions of USM and schedule with 128 threads per tile
        // make sure that blur result and USM happen in the same CUDA kernel
        // make blur result to be computed at CUDA block level of USM result
        // set this before applying default GPU schedules so that the automatic
        // heuristics do not get confused
        fc[1].compute_at(USM.as_func(), Var::gpu_blocks());
        fc[0].gpu_auto_schedule(128);
        fc[1].gpu_auto_schedule(128);
        USM  .gpu_auto_schedule(128, tile_width);

        USM  .apply_bounds();
        USM  .profile(iter);
    }

//    // assemble channels again and save result
//    // should also be done on the GPU
//    {
//        Var i("i"), j("j"), c("c");
//
//        Result(i,j,c) = select(c==0, USM(i,j)[0], c==1, USM(i,j)[1], USM(i,j)[2]);
//
//        Buffer buff(type_of<float>(), width, height, channels);
//        Result.realize(buff);
//        Image<float> output(buff);
//
//        save(output, "out.png");
//    }

    return EXIT_SUCCESS;
}
