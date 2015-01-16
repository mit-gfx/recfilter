/**
 * \file box_filter_6.cpp
 *
 * Six box filters computed using two cascaded 3rd order IIR filters
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

    RecFilter S1 = integral_image  (3, in_w, in_w, tile_width, I);
    RecFilter D1 = derivative_image(3, box_filter_radius, in_w, in_w, tile_width, S1.as_func());
    RecFilter S2 = integral_image  (3, in_w, in_w, tile_width, D1.as_func());
    RecFilter D2 = derivative_image(3, box_filter_radius, in_w, in_w, tile_width, S2.as_func());

    D2.profile(iter);

    return 0;
}
