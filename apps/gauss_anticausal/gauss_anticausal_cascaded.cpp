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
    int  width   = args.width;
    int  height  = args.width;
    int  tile_width = args.block;

    float sigma = 16.0f;

    Image<float> random_image = generate_random_image<float>(width,height);

    ImageParam image(type_of<float>(), 2);
    image.set(random_image);

    // ----------------------------------------------------------------------------------------------

    float         B1 = gaussian_weights(sigma, 1).first;
    float         B2 = gaussian_weights(sigma, 2).first;
    vector<float> W1 = gaussian_weights(sigma, 1).second;
    vector<float> W2 = gaussian_weights(sigma, 2).second;

    Var x("x");
    Var y("y");

    RecFilter filter("Gauss");

    filter.set_args(x, y, width, height);
    filter.set_clamped_image_border();
    filter.define(image(clamp(x,0,image.width()-1), clamp(y,0,image.height()-1)));

    filter.add_causal_filter    (x, B1, W1);
    filter.add_anticausal_filter(x, B1, W1);
    filter.add_causal_filter    (y, B1, W1);
    filter.add_anticausal_filter(y, B1, W1);
    filter.add_causal_filter    (x, B2, W2);
    filter.add_anticausal_filter(x, B2, W2);
    filter.add_causal_filter    (y, B2, W2);
    filter.add_anticausal_filter(y, B2, W2);

    // cascade the scans
    vector<RecFilter> cascaded_filters = filter.cascade(
            make_vec(0,1,2,3),
            make_vec(4,5,6,7));

    RecFilter filter1 = cascaded_filters[0];
    RecFilter filter2 = cascaded_filters[1];

    filter1.split(x, tile_width, y, tile_width);
    filter2.split(x, tile_width, y, tile_width);

    cerr << filter2 << endl;

    // ----------------------------------------------------------------------------------------------

    {
    }

    // ----------------------------------------------------------------------------------------------

    Target target = get_jit_target_from_environment();

    cerr << "\nJIT compilation ... " << endl;
    filter2.func().compile_jit();

    Buffer hl_out_buff(type_of<float>(), width,height);
    {
        filter2.func().realize(hl_out_buff);
    }
    hl_out_buff.copy_to_host();
    hl_out_buff.free_dev_buffer();

    // ----------------------------------------------------------------------------------------------

    if (!nocheck) {
        cerr << "\nChecking difference ...\n" << endl;
        Image<float> hl_out(hl_out_buff);
        Image<float> ref = reference_gaussian(random_image, sigma);
        cerr << "Difference with true Gaussian \n" << CheckResult(ref,hl_out) << endl;
    }

    return 0;
}
