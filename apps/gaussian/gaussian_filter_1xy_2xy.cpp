/**
 * \file gaussian_filter_1xy_2xy.cpp
 *
 * Gaussian blur using IIR filters: 1st and 2nd order x-y overlapped
 */

#include <iostream>
#include <Halide.h>

#include "recfilter.h"
#include "iir_coeff.h"

using namespace Halide;

using std::vector;

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
    vector<float> W1 = gaussian_weights(sigma, 1);
    vector<float> W2 = gaussian_weights(sigma, 2);
    vector<float> W3 = gaussian_weights(sigma, 3);

    RecFilter F("Gaussian_1xy_2xy");

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

    vector<RecFilter> fc = F.cascade({0,1,2,3}, {4,5,6,7});

    for (size_t k=0; k<fc.size(); k++) {
        RecFilter f = fc[k];

        f.split_all_dimensions(tile_width);

        if (nosched) {
            manual_schedule(f);
        } else {
            f.gpu_auto_schedule(128);
        }
    }

    fc[fc.size()-1].profile(iter);

    return EXIT_SUCCESS;
}

void manual_schedule(RecFilter& f) {
    int n_scans  = 4;
    int ws       = 32;
    int unroll_w = ws/4;
    int intra_tiles_per_warp = ws / n_scans;
    int inter_tiles_per_warp = 4;

    f.intra_schedule(1).compute_locally()
        .reorder_storage(f.inner(), f.outer())
        .unroll         (f.inner_scan())
        .split          (f.inner(1), unroll_w)
        .unroll         (f.inner(1).split_var())
        .reorder        (f.inner_scan(), f.inner(1).split_var(), f.inner(), f.outer())
        .gpu_threads    (f.inner(0), f.inner(1))
        .gpu_blocks     (f.outer(0), f.outer(1));

    f.intra_schedule(2).compute_locally()
        .unroll         (f.inner_scan())
        .split          (f.outer(0), intra_tiles_per_warp)
        .reorder        (f.inner(),  f.inner_scan(), f.tail(), f.outer(0).split_var(), f.outer())
        .fuse           (f.tail(), f.inner(0))
        .gpu_threads    (f.tail(), f.outer(0).split_var())
        .gpu_blocks     (f.outer(0), f.outer(1));

    f.inter_schedule().compute_globally()
        .reorder_storage(f.inner(), f.tail(), f.outer())
        .unroll         (f.outer_scan())
        .split          (f.outer(0), inter_tiles_per_warp)
        .reorder        (f.outer_scan(), f.tail(), f.outer(0).split_var(), f.inner(), f.outer())
        .gpu_threads    (f.inner(0), f.outer(0).split_var())
        .gpu_blocks     (f.outer(0));
}
