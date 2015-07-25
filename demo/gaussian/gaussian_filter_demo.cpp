/**
 * \file gaussian_filter_demo.cpp
 *
 * Gaussian blur demo using IIR filters: cascade of 3rd order x and 3rd order y
 */

#include <iostream>
#include <Halide.h>

#include <recfilter.h>
#include <iir_coeff.h>

#include "../image_io.h"

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
        cerr << "Usage: gaussian_filter_demo [name of png file]" << endl;
        return EXIT_FAILURE;
    }

    Image<uint8_t> input = load<uint8_t>(filename);

    int width   = input.width();
    int height  = input.height();
    int channels= input.channels();

    float sigma = 10.0;
    vector<float> W3 = gaussian_weights(sigma,3);

    Func Result;
    Func Gaussian;

    // perform the Gaussian blur
    {
        RecFilter S("GaussianDemo");
        RecFilterDim x("x", width);
        RecFilterDim y("y", height);

        Expr r = cast<float>(input(x,y,0)) / 255.0f;
        Expr g = cast<float>(input(x,y,1)) / 255.0f;
        Expr b = cast<float>(input(x,y,2)) / 255.0f;

        S.set_clamped_image_border();

        S(x,y) = Tuple(r,g,b);
        S.add_filter(+x, W3);
        S.add_filter(-x, W3);
        S.add_filter(+y, W3);
        S.add_filter(-y, W3);

        vector<RecFilter> fc = S.cascade_by_dimension();

        fc[0].split_all_dimensions(32);
        fc[1].split_all_dimensions(32);

        fc[0].gpu_auto_schedule(128);
        fc[1].gpu_auto_schedule(128);

        Gaussian = fc[1].as_func();
    }

    // assemble channels again and save result
    {
        Var i("i"), j("j"), c("c");

        Result(i,j,c) = select(c==0, Gaussian(i,j)[0], c==1, Gaussian(i,j)[1], Gaussian(i,j)[2]);

        Buffer buff(type_of<float>(), width, height, channels);
        Result.realize(buff);
        Image<float> output(buff);

        save(output, "out.png");
    }

    return EXIT_SUCCESS;
}
