/**
 * \file box_filter_1.cpp
 *
 * Single box filter computed using summed area table
 */

#include <iostream>
#include <Halide.h>

#include "recfilter.h"

using namespace Halide;

int main(int argc, char **argv) {
    const int filter_radius = 5;

    Arguments args(argc, argv);
    int iter       = args.iterations;
    int tile_width = args.block;
    int width      = args.width;
    int height     = args.width;

    int B = filter_radius;

    Image<float> I = generate_random_image<float>(width,height);

    RecFilterDim x("x", width);
    RecFilterDim y("y", height);

    RecFilter F("SAT");
    RecFilter D("Box");

    // integral_image
    F(x,y) = I(x,y);
    F.add_filter(x, {1.0f, 1.0f});
    F.add_filter(y, {1.0f, 1.0f});
    F.split(x, tile_width, y, tile_width);

    // derivative_image, set border as input image
    Func f = F.as_func();
    D(x,y) = select(x<B+1 || y<B+1 || x>width-1-B || y>height-1-B, 0,
            (1.0f * f(clamp(x+B+0, 0, width-1), clamp(y+B+0, 0, height-1)) +
            -1.0f * f(clamp(x+B+0, 0, width-1), clamp(y-B-1, 0, height-1)) +
             1.0f * f(clamp(x-B-1, 0, width-1), clamp(y-B-1, 0, height-1)) +
            -1.0f * f(clamp(x-B-1, 0, width-1), clamp(y+B+0, 0, height-1))) / std::pow(2*B+1,2));

    if (D.target().has_gpu_feature() && F.target().has_gpu_feature()) {
        int order    = 1;
        int n_scans  = 2;
        int ws       = 32;
        int unroll_w = ws/4;
        int intra_tiles_per_warp = ws / (order*n_scans);
        int inter_tiles_per_warp = 4;

        Var xo("xo"), xi("xi"), xii("xii");
        Var yo("yo"), yi("yi"), yii("yii");

        Var u = x.var();
        Var v = y.var();

        D.compute_globally()
            .split  (u, xo, xi, tile_width)
            .split  (v, yo, yi, tile_width)
            .split  (yi,yi, yii,8)
            .unroll (yii)
            .reorder(yii,xi,yi,xo,yo)
            .gpu    (xo,yo,xi,yi);

        F.intra_schedule(1).compute_locally()
            .reorder_storage(F.inner(), F.outer())
            .unroll         (F.inner_scan())
            .split          (F.inner(1), unroll_w)
            .unroll         (F.inner(1).split_var())
            .reorder        (F.inner_scan(), F.inner(1).split_var(), F.inner(), F.outer())
            .gpu_threads    (F.inner(0), F.inner(1))
            .gpu_blocks     (F.outer(0), F.outer(1));

        F.intra_schedule(2).compute_locally()
            .unroll         (F.inner_scan())
            .split          (F.outer(0), intra_tiles_per_warp)
            .reorder        (F.inner(),  F.inner_scan(), F.tail(), F.outer(0).split_var(), F.outer())
            .fuse           (F.tail(), F.inner(0))
            .gpu_threads    (F.tail(), F.outer(0).split_var())
            .gpu_blocks     (F.outer(0), F.outer(1));

        F.inter_schedule().compute_globally()
            .reorder_storage(F.inner(), F.tail(), F.outer())
            .unroll         (F.outer_scan())
            .split          (F.outer(0), inter_tiles_per_warp)
            .reorder        (F.outer_scan(), F.tail(), F.outer(0).split_var(), F.inner(), F.outer())
            .gpu_threads    (F.inner(0), F.outer(0).split_var())
            .gpu_blocks     (F.outer(0));
    }

    D.profile(iter);

    return 0;
}
