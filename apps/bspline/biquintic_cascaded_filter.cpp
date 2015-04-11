/**
 * \file biquintic_cascaded_filter.cpp
 *
 * Bspline biquintic interpolation using IIR filters: causal-anticausal second
 * order filters in x followed by second order causal-anticausal filters in y
 *
 * FIR Biquintic Kernel Matrix
 *     [ 1/120  13/60  11/20 13/60  1/120 0;
 *      -1/24   -5/12    0    5/12  1/24  0;
 *       1/12    1/6   -1/2   1/6   1/12  0;
 *      -1/12    1/6     0   -1/6   1/12  0;
 *       1/24   -1/6    1/4  -1/6   1/24  0;
 *      -1/120   1/24  -1/12  1/12 -1/24 1/120];
 *
 * IIR filter weights may not be inccurate for the bicubic and biquintic filters,
 * these are just used for performance evaluation
 */

#include <iostream>
#include <Halide.h>

#include "recfilter.h"

using namespace Halide;

using std::map;
using std::vector;
using std::string;
using std::cout;
using std::cout;
using std::endl;

template<typename T>
void check(RecFilter F, vector<float> filter_coeff, Image<T> image);
void manual_schedule(RecFilter& fx, RecFilter& fy);

int main(int argc, char **argv) {
    Arguments args(argc, argv);

    bool nosched   = args.noschedule;
    bool nocheck   = args.nocheck;
    int iter       = args.iterations;
    int tile_width = args.block;
    int in_w       = args.width;

    // inaccurate coefficients, only for measuring performance
    const float a = 2.0f-std::sqrt(3.0f);
    vector<float> coeff = {1+a, -a, 0.1f};

    int width = in_w;
    int height= in_w;

    Image<float> image = generate_random_image<float>(width,height);

    RecFilterDim x("x", width);
    RecFilterDim y("y", height);

    RecFilter F("Biquintic_cascaded");

    F.set_clamped_image_border();

    F(x,y) = image(x,y);
    F.add_filter(+x, coeff);
    F.add_filter(-x, coeff);
    F.add_filter(+y, coeff);
    F.add_filter(-y, coeff);

    vector<RecFilter> fc = F.cascade_by_dimension();

    fc[0].split_all_dimensions(tile_width);
    fc[1].split_all_dimensions(tile_width);

    if (nosched) {
        manual_schedule(fc[0], fc[1]);
    } else {
        int max_threads = 128;
        fc[0].gpu_auto_schedule(max_threads);
        fc[1].gpu_auto_schedule(max_threads);
    }

    fc[1].profile(iter);

    if (!nocheck) {
        check<float>(fc[1], coeff, image);
    }

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
            .reorder_storage(f.inner(), f.full(0), f.outer())
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
            .reorder_storage(f.full(0), f.inner(), f.outer())
            .split          (f.full(0), ws)
            .split          (f.inner(0), unroll_w)
            .unroll         (f.inner(0).split_var())
            .unroll         (f.inner_scan())
            .reorder        (f.inner_scan(), f.inner(0).split_var(), f.inner(0), f.full(0).split_var(), f.full(0), f.outer(0))
            .gpu_threads    (f.full(0).split_var(), f.inner(0))
            .gpu_blocks     (f.full(0), f.outer(0));

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

template<typename T>
void check(RecFilter F, vector<float> filter_coeff, Image<T> image) {
    cout << "\nChecking difference ... " << endl;

    int width = image.width();
    int height = image.height();

    Realization out = F.realize();
    Image<float> hl_out(out);
    Image<float> ref(width,height);

    float b0 = (filter_coeff.size()>=0 ? filter_coeff[0] : 0.0f);
    float a1 = (filter_coeff.size()>=1 ? filter_coeff[1] : 0.0f);
    float a2 = (filter_coeff.size()>=2 ? filter_coeff[2] : 0.0f);

    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            ref(x,y) = image(x,y);
        }
    }
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            ref(x,y) = b0*ref(x,y)
                + a1*ref(std::max(x-1,0),y)
                + a2*ref(std::max(x-2,0),y);
        }
    }
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            ref(x,y) = b0*ref(x,y)
                + a1*ref(x,std::max(y-1,0))
                + a2*ref(x,std::max(y-2,0));
        }
    }
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            ref(width-1-x,y) = b0*ref(width-1-x,y)
                + a1*ref(width-1-std::max(x-1,0),y)
                + a2*ref(width-1-std::max(x-2,0),y);
        }
    }
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            ref(x,height-1-y) = b0*ref(x,height-1-y)
                + a1*ref(x,height-1-std::max(y-1,0))
                + a2*ref(x,height-1-std::max(y-2,0));
        }
    }
    cout << CheckResult<float>(ref,hl_out) << endl;
}
