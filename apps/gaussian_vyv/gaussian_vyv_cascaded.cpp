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

    bool nocheck = true; // args.nocheck;
    int  width   = args.width;
    int  height  = args.width;
    int  tile_width = args.block;
    int  iterations = args.iterations;

    Image<float> image = generate_random_image<float>(width,height);

    // ----------------------------------------------------------------------------------------------

    float sigma = 5.0;
    vector<float> W1 = gaussian_weights(sigma, 1);
    vector<float> W2 = gaussian_weights(sigma, 2);

    RecFilterDim x("x", width);
    RecFilterDim y("y", height);

    RecFilter F;
    F.set_clamped_image_border();

    F(x,y) = image(x,y);

    F.add_filter(+x, W1);
    F.add_filter(-x, W1);
    F.add_filter(+y, W1);
    F.add_filter(-y, W1);
    F.add_filter(+x, W2);
    F.add_filter(-x, W2);
    F.add_filter(+y, W2);
    F.add_filter(-y, W2);

    // cascade the scans
    vector<RecFilter> cascaded_filters = F.cascade({0,1,2,3}, {4,5,6,7});

    RecFilter F1 = cascaded_filters[0];
    RecFilter F2 = cascaded_filters[1];

    F1.split(x, tile_width, y, tile_width);
    F2.split(x, tile_width, y, tile_width);

    // ----------------------------------------------------------------------------------------------

    {
    }

    // ----------------------------------------------------------------------------------------------

    cerr << "width\ttime (ms)" << endl;

    Realization out = F2.realize();
    float time = F2.profile(iterations);
    cerr << width << "\t" << time << endl;

    // ----------------------------------------------------------------------------------------------

    if (!nocheck) {
        cerr << "\nChecking difference ...\n" << endl;
        Image<float> hl_out(out);
        Image<float> ref = reference_gaussian<float>(image, sigma);
        cerr << "Difference with true Gaussian \n" << CheckResult<float>(ref,hl_out) << endl;
    }

    return 0;
}
