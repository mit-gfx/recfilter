#include <iostream>
#include <cstdio>
#include <cmath>

#include <Halide.h>

#include "gaussian_weights.h"
#include "recfilter.h"

using namespace Halide;

using std::vector;
using std::string;
using std::cerr;
using std::endl;

#define VECTORIZE_WIDTH 8

int main(int argc, char **argv) {
    Arguments args(argc, argv);

    bool nocheck    = true; //args.nocheck;
    int  width      = args.width;
    int  height     = args.width;
    int  tile_width = args.block;
    int  iterations = args.iterations;

    Image<float> image = generate_random_image<float>(width,height);
    Buffer out(type_of<float>(), width, height);

    // ----------------------------------------------------------------------------------------------

    double time  = 0.0;
    double sigma = 5.0;
    int    box   = gaussian_box_filter(3, sigma); // approx Gaussian with 3 box filters
    double norm  = std::pow(box, 3*2);            // normalizing factor

    // ----------------------------------------------------------------------------------------------

    Func I("I");
    Func S("S");

    RecFilterDim x("x", width);
    RecFilterDim y("y", height);

    I(x,y) = image(clamp(x,0,image.width()-1), clamp(y,0,image.height()-1));

    // convolve image with third derivative of three box filters
    S(x,y) =
            ( 1.0 * I(x+0*box,y+0*box) +
             -3.0 * I(x+1*box,y+0*box) +
              3.0 * I(x+2*box,y+0*box) +
             -1.0 * I(x+3*box,y+0*box) +
             -3.0 * I(x+0*box,y+1*box) +
              9.0 * I(x+1*box,y+1*box) +
             -9.0 * I(x+2*box,y+1*box) +
              3.0 * I(x+3*box,y+1*box) +
              3.0 * I(x+0*box,y+2*box) +
             -9.0 * I(x+1*box,y+2*box) +
              9.0 * I(x+2*box,y+2*box) +
             -3.0 * I(x+3*box,y+2*box) +
             -1.0 * I(x+0*box,y+3*box) +
              3.0 * I(x+1*box,y+3*box) +
             -3.0 * I(x+2*box,y+3*box) +
              1.0 * I(x+3*box,y+3*box)) / norm;

    // ----------------------------------------------------------------------------------------------

    // non-tiled version of the same filter
    RecFilter F_non_tiled;
    F_non_tiled(x, y) = S(x,y);
    F_non_tiled.add_filter(+x, {1.0, 3.0, -3.0, 1.0});
    F_non_tiled.add_filter(+y, {1.0, 3.0, -3.0, 1.0});

    // tiled version of the same filter
    RecFilter F_tiled(F_non_tiled);
    F_tiled(x, y) = S(x,y);
    F_tiled.add_filter(+x, {1.0, 3.0, -3.0, 1.0});
    F_tiled.add_filter(+y, {1.0, 3.0, -3.0, 1.0});
    F_tiled.split(x, tile_width, y, tile_width);

    // ----------------------------------------------------------------------------------------------

    if (F_tiled.target().has_gpu_feature() ||
        F_non_tiled.target().has_gpu_feature())
    {
    } else {
        Var xi;

        S.compute_root()
            .split(x.var(), x.var(), xi,VECTORIZE_WIDTH)
            .vectorize(xi).parallel(x.var()).parallel(y.var());

        F_non_tiled.intra_schedule().compute_globally()
            .reorder(F_non_tiled.inner_scan(), F_non_tiled.full())
            .split(F_non_tiled.full(0), VECTORIZE_WIDTH)
            .vectorize(F_non_tiled.full(0).split_var())
            .parallel(F_non_tiled.full(0));

//        F_tiled.intra_schedule().compute_globally()
//            .reorder  (F_tiled.inner_scan(), F_tiled.full())
//            .split    (F_tiled.full(0), VECTORIZE_WIDTH)
//            .vectorize(F_tiled.full(0).split_var())
//            .parallel (F_tiled.full(0));
//
//        F_tiled.inter_schedule().compute_globally()
//            .reorder  (F_tiled.outer_scan(), F_tiled.full())
//            .split    (F_tiled.full(0), VECTORIZE_WIDTH)
//            .vectorize(F_tiled.full(0).split_var())
//            .parallel (F_tiled.full(0));
//
//        time = F_tiled.realize(out, iterations);
        time = F_non_tiled.realize(out, iterations);
        cerr << "non_tiled_direct\t" << width << "\t" << time << endl;
        cerr << "tiled_direct\t" << width << "\t" << time << endl;
    }

    // ----------------------------------------------------------------------------------------------

    if (!nocheck) {
        cerr << "\nChecking difference ...\n" << endl;
        Image<float> hl_out(out);
        Image<float> ref = reference_gaussian<float>(image, sigma);
        cerr << "Difference with true Gaussian \n" << CheckResultVerbose<float>(ref,hl_out) << endl;
    }

    return 0;
}
