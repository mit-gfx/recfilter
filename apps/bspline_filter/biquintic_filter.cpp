/**
 * \file biquintic_filter.cpp
 *
 * Bspline interpolation using IIR filters:
 * - biquintic using second order 2D causal-anticausal overlapped
 * - biquintic using two cascaded second order 1D causal-anticausal overlapped
 *
 * Biquintic Kernel Matrix
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
using std::cerr;
using std::cout;
using std::endl;

template<typename T>
void check(RecFilter F, vector<float> filter_coeff, Image<T> image);

int main(int argc, char **argv) {
    Arguments args(argc, argv);

    bool nocheck   = args.nocheck;
    int iter       = args.iterations;
    int tile_width = args.block;
    int in_w       = args.width;

    // possibly inaccurate coeff, only for measuring performance
    const float a = 2.0f-std::sqrt(3.0f);
    vector<float> coeff = {1+a, -a, 0.1f};

    int width = in_w;
    int height= in_w;

    Image<float> image = generate_random_image<float>(width,height);

    RecFilterDim x("x", width);
    RecFilterDim y("y", height);

    RecFilter F("Biquintic");

    F.set_clamped_image_border();

    F(x,y) = image(x,y);
    F.add_filter(+x, coeff);
    F.add_filter(-x, coeff);
    F.add_filter(+y, coeff);
    F.add_filter(-y, coeff);

#define OVERLAPPED 0
#if OVERLAPPED
    F.split_all_dimensions(tile_width);

    if (F.target().has_gpu_feature()) {
        int order    = coeff.size()-1;
        int n_scans  = 4;
        int ws       = 32;
        int unroll_w = ws/4;
        int intra_tiles_per_warp = ws / (order*n_scans);
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

        F.profile(iter);
    }
#else
    vector<RecFilter> fc = F.cascade_by_dimension();
    for (size_t i=0; i<fc.size(); i++) {
        RecFilter f = fc[i];

        f.split_all_dimensions(tile_width);

        if (f.target().has_gpu_feature()) {
            int order    = coeff.size()-1;
            int ws       = 32;
            int unroll_w = ws/4;
            int nthreads = 128;

            f.intra_schedule(1).compute_locally()
                .reorder_storage(f.inner(), f.outer(), f.full())
                .unroll         (f.inner_scan())
                .split          (f.inner(0), unroll_w)
                .split          (f.full(0),  ws)
                .unroll         (f.inner(0).split_var())
                .reorder        (f.inner_scan(), f.inner(0).split_var(), f.inner(0), f.full(0).split_var(), f.outer(0), f.full(0))
                .gpu_threads    (f.inner(0), f.full(0).split_var())
                .gpu_blocks     (f.outer(0), f.full(0));

            f.inter_schedule().compute_globally()
                .reorder_storage(f.tail(), F.outer(0), f.full(0))
                .unroll         (f.outer_scan())
                .split          (f.full(0), nthreads/order)
                .reorder        (f.outer_scan(), f.tail(), f.full(0))
                .gpu_threads    (f.tail(), f.full(0).split_var())
                .gpu_blocks     (f.full(0));

            cerr << f << endl;
        }

        fc[fc.size()-1].profile(iter);
    }
#endif

    if (!nocheck) {
        check<float>(F, coeff, image);
    }

    return 0;
}

template<typename T>
void check(RecFilter F, vector<float> filter_coeff, Image<T> image) {
    cerr << "\nChecking difference ... " << endl;

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
