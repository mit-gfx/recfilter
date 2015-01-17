/**
 * \file box_filter_3.cpp
 *
 * Three box filters computed using 3rd order IIR filter
 */

#include "box_filter.h"

using namespace Halide;

int main(int argc, char **argv) {
    const int box_filter_radius = 1;

    Arguments args(argc, argv);
    int iter       = args.iterations;
    int tile_width = args.block;
    int in_w       = args.width;

    Image<float> I = generate_random_image<float>(in_w,in_w);

    RecFilter S = integral_image   (1, in_w, in_w, tile_width, I);
    RecFilter D1 = derivative_image(1, box_filter_radius, in_w, in_w, tile_width, S.as_func());
    RecFilter D2 = derivative_image(1, box_filter_radius, in_w, in_w, tile_width, D1.as_func());
    RecFilter D3 = derivative_image(1, box_filter_radius, in_w, in_w, tile_width, D2.as_func());

    RecFilter D = D3;

    D.profile(iter);

    Realization R = D.realize();
    Image<float> res(R);
    cerr << res << endl;

    return 0;
}
