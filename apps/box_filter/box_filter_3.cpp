/**
 * \file box_filter_3.cpp
 *
 * Three box filters computed using 3rd order IIR filter
 */

#include "box_filter.h"

using namespace Halide;

int main(int argc, char **argv) {
    const int box_filter_radius = 10;

    Arguments args(argc, argv);
    int iter       = args.iterations;
    int tile_width = args.block;
    int in_w       = args.width;

    Image<float> image = generate_random_image<float>(in_w,in_w);
    ImageParam I;
    I.set(image);

    RecFilter S = integral_image  (3, in_w, in_w, tile_width, I);
    RecFilter D = derivative_image(3, box_filter_radius, in_w, in_w, tile_width, S.as_func());

    D.profile(iter);

    return 0;
}
