/**
 * \file box_filter_1.cpp
 *
 * Single box filter computed using summed area table
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

    Image<float> I = generate_random_image<float>(width,height);

    // pad the image with zeros
    int pad = (B+1)+1;
    for (int i=0; i<I.width(); i++) {
        for (int j=0; j<I.height(); j++) {
            if (i<pad || i>width-pad || j<pad || j>height-pad) {
                I(i,j) = 0.0f;
            }
        }
    }

    RecFilter b1 = box_filter_order_1(I, width, height, B, tile_width, !nosched);

    b1.profile(iter);

    return 0;
}
