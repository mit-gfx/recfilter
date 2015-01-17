/**
 * \file bicubic_filter.cpp
 *
 * Bspline interpolation using IIR filters:
 * - bicubic bspline interpolation using first order 2D causal-anticausal overlapped
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

    const float a = 2.0f-std::sqrt(3.0f);
    vector<float> coeff = {1+a, -a};

    int width = in_w;
    int height= in_w;

    Image<float> image = generate_random_image<float>(width,height);

    RecFilterDim x("x", width);
    RecFilterDim y("y", height);

    RecFilter F("Bicubic");

    F.set_clamped_image_border();

    F(x,y) = image(x,y);
    F.add_filter(+x, coeff);
    F.add_filter(-x, coeff);
    F.add_filter(+y, coeff);
    F.add_filter(-y, coeff);

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
    }

    F.profile(iter);

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
