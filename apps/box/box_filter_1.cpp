/**
 * \file box_filter_1.cpp
 *
 * Single box filter computed using summed area table
 */

#include "box_filter.h"

using namespace Halide;

int main(int argc, char **argv) {
    const int box_filter_radius = 5;

    Arguments args(argc, argv);
    int iter       = args.iterations;
    int tile_width = args.block;
    int in_w       = args.width;

    Image<float> I = generate_random_image<float>(in_w,in_w);

    RecFilter S = integral_image  (1, in_w, in_w, tile_width, I);
    RecFilter D = derivative_image(1, box_filter_radius, in_w, in_w, tile_width, S.as_func());

    Realization R = D.realize();
    Image<float> res(R);
    cerr << res << endl;

    D.profile(iter);
}
