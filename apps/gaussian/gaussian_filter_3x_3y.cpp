/**
 * \file gaussian_filter_3x_3y.cpp
 *
 * Gaussian blur using IIR filters: cascade of 3rd order x and 3rd order y
 */

#include <iostream>
#include <Halide.h>

#include "recfilter.h"
#include "iir_coeff.h"

using namespace Halide;

using std::vector;

void manual_schedule(RecFilter& fx, RecFilter& fy);

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
    vector<float> W3 = gaussian_weights(sigma,3);

    RecFilter G("Gaussian_3x_3y");

    G.set_clamped_image_border();

    G(x,y) = image(x,y);
    G.add_filter(+x, W3);
    G.add_filter(-x, W3);
    G.add_filter(+y, W3);
    G.add_filter(-y, W3);

    vector<RecFilter> fc = G.cascade_by_dimension();

    fc[0].split_all_dimensions(tile_width);
    fc[1].split_all_dimensions(tile_width);

    if (nosched) {
        manual_schedule(fc[0], fc[1]);
    } else {
        RecFilter::set_max_threads_per_cuda_warp(128);
        fc[0].gpu_auto_schedule();
        fc[1].gpu_auto_schedule();
    }

    fc[1].profile(iter);

    return EXIT_SUCCESS;
}

void manual_schedule(RecFilter& fx, RecFilter& fy) {
    int ws       = 32;
    int unroll_w = 8;
    int tiles_per_warp = 4;

    // x filter schedule
    {
        RecFilter f = fx;
        f.intra_schedule().compute_locally()
            .split          (f.full(0), ws, f.inner())
            .split          (f.inner(1), unroll_w)
            .unroll         (f.inner(1).split_var())
            .unroll         (f.inner_scan())
            .reorder        (f.inner_scan(), f.inner(1).split_var(), f.tail(), f.inner(), f.outer(), f.full())
            .gpu_threads    (f.inner(0), f.inner(1))
            .gpu_blocks     (f.outer(0), f.full(0));

        f.inter_schedule().compute_globally()
            .reorder_storage(f.full(0), f.tail(), f.outer(0))
            .split          (f.full(0), ws, f.inner())
            .unroll         (f.outer_scan())
            .split          (f.full(0), tiles_per_warp)
            .reorder        (f.outer_scan(), f.tail(), f.full(0).split_var(), f.inner(), f.full(0))
            .gpu_threads    (f.inner(0), f.full(0).split_var())
            .gpu_blocks     (f.full(0));
    }

    // y filter schedule: same schedule as x but very small subtle changes in intra_schedule
    {
        RecFilter f = fy;
        f.intra_schedule().compute_locally()
            .split      (f.full(0), ws)
            .split      (f.inner(0), unroll_w)
            .unroll     (f.inner(0).split_var())
            .unroll     (f.inner_scan())
            .reorder    (f.inner_scan(), f.inner(0).split_var(), f.inner(0), f.full(0).split_var(), f.full(0), f.outer(0))
            .gpu_threads(f.full(0).split_var(), f.inner(0))
            .gpu_blocks (f.full(0), f.outer(0));

        f.inter_schedule().compute_globally()
            .reorder_storage(f.full(0), f.tail(), f.outer(0))
            .split          (f.full(0), ws, f.inner())
            .unroll         (f.outer_scan())
            .split          (f.full(0), tiles_per_warp)
            .reorder        (f.outer_scan(), f.tail(), f.full(0).split_var(), f.inner(), f.full(0))
            .gpu_threads    (f.inner(0), f.full(0).split_var())
            .gpu_blocks     (f.full(0));
    }
}
