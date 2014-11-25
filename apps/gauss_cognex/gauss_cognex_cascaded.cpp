#include <iostream>
#include <cstdio>
#include <cmath>

#include <Halide.h>

#include "../../lib/gaussian_weights.h"
#include "../../lib/recfilter.h"

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

    float sigma = 16.0f;
    int   box   = gaussian_box_filter(3, sigma); // approx Gaussian with 3 box filters
    float norm  = std::pow(box, 3*2);            // normalizing factor

    // ----------------------------------------------------------------------------------------------
    vector<float> W1, W2;
    W1.push_back( 1.0f);
    W2.push_back( 2.0f);
    W2.push_back(-1.0f);

    Func I("I");
    Func S("S");

    Var x("x");
    Var y("y");

    I(x,y) = image(clamp(x,0,image.width()-1), clamp(y,0,image.height()-1));

    // convolve image with third derivative of three box filters
    S(x,y) =
            ( 1.0f * I(x+0*box,y+0*box) +
             -3.0f * I(x+1*box,y+0*box) +
              3.0f * I(x+2*box,y+0*box) +
             -1.0f * I(x+3*box,y+0*box) +
             -3.0f * I(x+0*box,y+1*box) +
              9.0f * I(x+1*box,y+1*box) +
             -9.0f * I(x+2*box,y+1*box) +
              3.0f * I(x+3*box,y+1*box) +
              3.0f * I(x+0*box,y+2*box) +
             -9.0f * I(x+1*box,y+2*box) +
              9.0f * I(x+2*box,y+2*box) +
             -3.0f * I(x+3*box,y+2*box) +
             -1.0f * I(x+0*box,y+3*box) +
              3.0f * I(x+1*box,y+3*box) +
             -3.0f * I(x+2*box,y+3*box) +
              1.0f * I(x+3*box,y+3*box)) / norm;

    RecFilter filter;
    filter.set_args(x, y, width, height);
    filter.define(Expr(S(x, y)));
    filter.add_filter(x, 1.0f, W1, RecFilter::CAUSAL);
    filter.add_filter(x, 1.0f, W1, RecFilter::CAUSAL);
    filter.add_filter(y, 1.0f, W2, RecFilter::CAUSAL);
    filter.add_filter(y, 1.0f, W2, RecFilter::CAUSAL);

    // cascade the scans
    vector<RecFilter> cascaded_filters = filter.cascade(
            Internal::vec(0,1),
            Internal::vec(2,3));
    RecFilter filter1 = cascaded_filters[0];
    RecFilter filter2 = cascaded_filters[1];
    filter1.split(x, tile_width);
    filter2.split(y, tile_width);

    cerr << filter2 << endl;

    // ----------------------------------------------------------------------------------------------

    {
    }

    // ----------------------------------------------------------------------------------------------

    Target target = get_jit_target_from_environment();

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
        Image<float> ref = reference_gaussian(random_image, sigma);
        cerr << "Difference with true Gaussian \n" << CheckResultVerbose(ref,hl_out) << endl;
    }

    return 0;
}
