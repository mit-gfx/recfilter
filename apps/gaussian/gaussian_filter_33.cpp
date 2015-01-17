/**
 * \file gaussian_filter_33.cpp
 *
 * Gaussian blur using IIR filters: 3rd order x and 3rd order y
 */

#include <iostream>
#include <Halide.h>

#include "recfilter.h"
#include "iir_coeff.h"

using namespace Halide;

using std::map;
using std::vector;
using std::string;
using std::cerr;
using std::cout;
using std::endl;

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

    float sigma = 5.0;
    vector<float> W3 = gaussian_weights(sigma,3);

    RecFilter G("Gaussian_33");

    G.set_clamped_image_border();

    G(x,y) = image(x,y);
    G.add_filter(+x, W3);
    G.add_filter(-x, W3);
    G.add_filter(+y, W3);
    G.add_filter(-y, W3);

    vector<RecFilter> fc = G.cascade_by_dimension();

    for (size_t i=0; i<fc.size(); i++) {
        RecFilter& F = fc[i];

        F.split_all_dimensions(tile_width);

        if (F.target().has_gpu_feature()) {
            int n_scans  = 4;
            int ws       = 32;
            int unroll_w = ws/4;
            int intra_tiles_per_warp = ws / (4*n_scans);
            int inter_tiles_per_warp = 4;

        }
    }

    fc[fc.size()-1].profile(iter);

    return 0;
}
