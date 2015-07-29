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
    string filename;
    if (argc==2) {
        filename = argv[1];
    } else {
        cerr << "Usage: unsharp_mask_demo [name of png file]" << endl;
        return EXIT_FAILURE;
    }

    Image<uint8_t> input = load<uint8_t>(filename);

    int width   = input.width();
    int height  = input.height();
    int channels= input.channels();

    float sigma  = 5.0f;
    float weight = 2.0f;
    vector<float> W3 = gaussian_weights(sigma,3);

    RecFilter USM("USM");
    RecFilter GaussianX;
    RecFilter GaussianY;

    // some low pass blur
    {
        RecFilter S("Gaussian");
        RecFilterDim x("x", width);
        RecFilterDim y("y", height);

        Expr r    = cast<float>(input(x,y,0)) / 255.0f;    // RGB to YUV conversion
        Expr g    = cast<float>(input(x,y,1)) / 255.0f;
        Expr b    = cast<float>(input(x,y,2)) / 255.0f;
        Expr y_in =  0.299f*r + 0.587f*g + 0.114f*b;
        Expr u_in = -0.147f*r - 0.288f*g + 0.436f*b;
        Expr v_in =  0.615f*r - 0.515f*g - 0.100f*b;

        S.set_clamped_image_border();

        S(x,y) = y_in;
        S.add_filter(+x, W3);
        S.add_filter(-x, W3);
        S.add_filter(+y, W3);
        S.add_filter(-y, W3);

        vector<RecFilter> fc = S.cascade_by_dimension();

        GaussianX = fc[0];
        GaussianY = fc[1];

        GaussianX.split_all_dimensions(32);
        GaussianY.split_all_dimensions(32);

        // subtract the blurred luminance from original
        Expr y_final = clamp(y_in + weight*(y_in-GaussianY(x,y)), 0.0f, 1.0f);

        // grayscale only for now
        // YUV to RGB conversion
        Expr r_final = y_final; // + 0.000f*u_in + 1.139f*v_in;
        Expr g_final = y_final; // - 0.394f*u_in - 0.580f*v_in;
        Expr b_final = y_final; // + 2.032f*u_in + 0.000f*v_in;

        USM(x,y) = Tuple(r_final, g_final, b_final);

        // merge the last stage of Gaussian blur computation with unsharp
        // mask computation -- perform this merge before GPU scheduling so
        // that appropriate can be generated
        GaussianY.compute_at(USM.as_func(), Var::gpu_blocks());

        // set the bounds of USM
        // TODO: make this automatic
        USM.apply_bounds();

        // auto schedule for GPU
        GaussianX.gpu_auto_schedule(128);
        GaussianY.gpu_auto_schedule(128);
        USM      .gpu_auto_schedule(128, 32);
    }

    // assemble channels again and save result
    // TODO: should also be done on the GPU or avoided
    {
        Var i, j, c;

        Func Result;
        Result(i,j,c) = select(c==0, USM(i,j)[0],
                               c==1, USM(i,j)[1],
                                     USM(i,j)[2]);

        Buffer buff(type_of<float>(), width, height, channels);
        Result.realize(buff);
        Image<float> output(buff);

        save(output, "out.png");
    }

    return EXIT_SUCCESS;
}
