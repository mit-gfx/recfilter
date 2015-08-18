/**
 * \file gaussian_filter_1xy_2x_2y.cpp
 *
 * Gaussian blur using IIR filters: 1st x-y overlapped, 2nd order x and 2nd order y
 */

#include <iostream>
#include <Halide.h>

#include "recfilter.h"
#include "iir_coeff.h"

using namespace Halide;

using std::vector;

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
    vector<float> W1 = gaussian_weights(sigma, 1);
    vector<float> W2 = gaussian_weights(sigma, 2);

    RecFilter F("Gaussian_1xy_2x_2y");

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

    vector<RecFilter> fc = F.cascade({{0,1,2,3}, {4,5}, {6,7}});

    for (size_t k=0; k<fc.size(); k++) {
        RecFilter f = fc[k];
        RecFilter::set_max_threads_per_cuda_warp(128);
        f.split_all_dimensions(tile_width);
        f.gpu_auto_schedule();
    }

    fc[fc.size()-1].profile(iter);

    return EXIT_SUCCESS;
}
