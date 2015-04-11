/**
 * \file gaussian_overlapped_filter.cpp
 *
 * Gaussian blur using IIR filters: 3rd order x-y overlapped
 */

#include <iostream>
#include <Halide.h>

#include "recfilter.h"
#include "iir_coeff.h"

using namespace Halide;

using std::vector;
using std::cout;
using std::endl;

void manual_schedule(RecFilter& f);

int main(int argc, char **argv) {
    Arguments args(argc, argv);

    bool nosched   = args.noschedule;
    int iter       = args.iterations;
    int tile_width = args.block;
    int in_w       = args.width;

    int width = in_w;
    int height= in_w;

    Image<float> image = generate_random_image<float>(width,height);

    RecFilterDim x("x", width);
    RecFilterDim y("y", height);

    float sigma = 5.0;
    vector<float> W3 = gaussian_weights(sigma, 3);

    RecFilter F("Gaussian_3_overlapped");

    F.set_clamped_image_border();

    F(x,y) = image(x,y);
    F.add_filter(+x, W3);
    F.add_filter(-x, W3);
    F.add_filter(+y, W3);
    F.add_filter(-y, W3);

    F.split_all_dimensions(tile_width);

    if (nosched) {
        manual_schedule(F);
    } else {
        F.gpu_auto_schedule(128);
    }

    F.profile(iter);

    return EXIT_SUCCESS;
}

void manual_schedule(RecFilter& F) {
    int n_scans  = 4;
    int ws       = 32;
    int unroll_w = ws/4;
    int intra_tiles_per_warp = ws / (3*n_scans);
    int inter_tiles_per_warp = 4;

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
