/**
 * \file sat_filter_3.cpp
 *
 * Three box filters computed using 1st order integral images (overlapped in x and y)
 * and a second order integral image cascaded along x and y
 *
 * \todo the assumption is that the image is padded by k pixels where
 * k = box_filter_radius * num_applications_of_filter + 1
 */

#include "box_filter.h"

int main(int argc, char **argv) {
    const int B = 5;    // box filter radius

    Arguments args(argc, argv);

    bool nosched   = args.noschedule;
    int iter       = args.iterations;
    int tile_width = args.block;
    int width      = args.width;
    int height     = args.width;

    Image<float> in_image = generate_random_image<float>(width,height);

    // pad the image with zeros
    int pad = 6*(B+1)+1;
    for (int i=0; i<in_image.width(); i++) {
        for (int j=0; j<in_image.height(); j++) {
            if (i<pad || i>width-pad || j<pad || j>height-pad) {
                in_image(i,j) = 0.0f;
            }
        }
    }

    Var x,y;
    Func b0;
    b0(x,y) = in_image(x,y);

    RecFilter::set_max_threads_per_cuda_warp(128);

    RecFilter b1 = box_filter_order_2(b0,           width, height, B, tile_width, !nosched);
    RecFilter b2 = box_filter_order_2(b1.as_func(), width, height, B, tile_width, !nosched);
    RecFilter b3 = box_filter_order_2(b2.as_func(), width, height, B, tile_width, !nosched);

    b3.profile(iter);

    return 0;
}
