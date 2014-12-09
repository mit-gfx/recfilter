#include <iostream>
#include <cstdio>
#include <cmath>

#include <Halide.h>

#include "gaussian_weights.h"
#include "recfilter.h"

#define MAX_THREAD 192

using namespace Halide;

using std::vector;
using std::string;
using std::cerr;
using std::endl;

int main(int argc, char **argv) {
    Arguments args(argc, argv);

    bool nocheck = args.nocheck;
    int  width  = args.width;
    int  height = args.width;
    int  tile_width = args.block;

    Image<float> random_image = generate_random_image<float>(width,height);

    ImageParam image(type_of<float>(), 2);
    image.set(random_image);

    // ----------------------------------------------------------------------------------------------

    double sigma = 16.0;
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

    RecFilter filter;

    filter(x, y) = Expr(S(x,y));

    filter.add_filter(+x, {1.0, 1.0});
    filter.add_filter(+x, {1.0, 1.0});
    filter.add_filter(+y, {1.0, 2.0, -1.0});
    filter.add_filter(+y, {1.0, 2.0, -1.0});

    // cascade the scans
    vector<RecFilter> cascaded_filters = filter.cascade({0,1}, {2,3});
    RecFilter filter1 = cascaded_filters[0];
    RecFilter filter2 = cascaded_filters[1];

    filter1.split(x, tile_width);
    filter2.split(y, tile_width);

    cerr << filter2 << endl;

    // ----------------------------------------------------------------------------------------------

    {
    }

    // ----------------------------------------------------------------------------------------------

    cerr << "\nJIT compilation ... " << endl;
    filter.func().compile_jit();

    Buffer hl_out_buff(type_of<float>(), width,height);
    {
        filter.func().realize(hl_out_buff);
    }
    hl_out_buff.copy_to_host();
    hl_out_buff.free_dev_buffer();

    // ----------------------------------------------------------------------------------------------

    if (!nocheck) {
        cerr << "\nChecking difference ...\n" << endl;
        Image<float> hl_out(hl_out_buff);
        Image<float> ref = reference_gaussian<float>(random_image, sigma);
        cerr << "Difference with true Gaussian \n" << CheckResultVerbose<float>(ref,hl_out) << endl;
    }

    return 0;
}
