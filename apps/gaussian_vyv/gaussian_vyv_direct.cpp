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
    int  width   = args.width;
    int  height  = args.width;
    int  tile_width = args.block;

    Image<float> random_image = generate_random_image<float>(width,height);

    ImageParam image(type_of<float>(), 2);
    image.set(random_image);

    // ----------------------------------------------------------------------------------------------

    double sigma = 16.0;
    vector<double> W3 = gaussian_weights(sigma, 3);

    RecFilterDim x("x", width);
    RecFilterDim y("y", height);

    RecFilter filter;

    filter.set_clamped_image_border();

    filter(x, y) = image(clamp(x,0,image.width()-1), clamp(y,0,image.height()-1));

    filter.add_filter(+x, W3);
    filter.add_filter(-x, W3);
    filter.add_filter(+y, W3);
    filter.add_filter(-y, W3);

    filter.split(x, tile_width, y, tile_width);

    cerr << filter << endl;

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
