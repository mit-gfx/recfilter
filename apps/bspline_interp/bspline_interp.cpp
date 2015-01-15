/**
 *
 * \file bspline_interp.cpp
 *
 * Bspline interpolation using IIR filters:
 * - bicubic bspline interpolation using first order 2D causal-anticausal overlapped
 * - biquintic bspline interpolation using second order 2D causal-anticausal overlapped
 * - biquintic bspline interpolation using two cascaded second order 1D causal-anticausal overlapped
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
#include "timing.h"

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
    int min_w      = args.min_width;
    int max_w      = args.max_width;
    int inc_w      = tile_width;

    // vector<string> filters = {"Bicubic", "Biquintic", "Biquintic_cascaded"};
    // vector<string> filters = {"Bicubic", "Biquintic"};
    vector<string> filters = {"Bicubic", };

    // possibly inaccurate coeff, only for measuring performance
    const float a = 2.0f-std::sqrt(3.0f);
    vector<vector<float> > bspline_coeff = {{1+a, -a}, {1+a, -a}, {1+a, -a, 0.1f}};

    Log log("bspline_filter.perflog");
    log << "Width";
    for (int j=0; j<filters.size(); j++) {
        log << "\t" << filters[j];
    }
    log << "\n";

    // Profile the filter for all image widths
    for (int in_w=min_w; in_w<=max_w; in_w+=inc_w) {
        int width = in_w;
        int height= in_w;

        Image<float> image = generate_random_image<float>(width,height);

        RecFilterDim x("x", width);
        RecFilterDim y("y", height);

        vector<float> runtime(3, 0.0f);

        // run all three filters separately and record the runtimes
        for (int j=0; j<filters.size(); j++) {
            RecFilter F(filters[j]);

            vector<float> filter_coeff = bspline_coeff[j];

            F.set_clamped_image_border();

            F(x,y) = image(x,y);
            F.add_filter(+x, filter_coeff);
            F.add_filter(-x, filter_coeff);
            F.add_filter(+y, filter_coeff);
            F.add_filter(-y, filter_coeff);

            // same schedule for bicubic and biquintic, different schedule for biquintic cascaded
            if (j==0 || j==1) {

                F.split(x, tile_width, y, tile_width);

                int order    = (j==0 ? 1 : 2);
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
                    .reorder_storage(F.tail(), F.inner(), F.outer())
                    .unroll         (F.inner_scan())
                    .split          (F.outer(0), intra_tiles_per_warp)
                    .reorder        (F.inner_scan(), F.inner(), F.outer(0).split_var(), F.tail(), F.outer())
                    .fuse           (F.inner(0), F.outer(0).split_var())
                    .fuse           (F.inner(0), F.tail())
                    .gpu_threads    (F.inner(0))
                    .gpu_blocks     (F.outer(0), F.outer(1));

                F.inter_schedule().compute_globally()
                    .reorder_storage(F.inner(), F.tail(), F.outer())
                    .unroll         (F.outer_scan())
                    .split          (F.outer(0), inter_tiles_per_warp)
                    .reorder        (F.outer_scan(), F.tail(), F.outer(0).split_var(), F.inner(), F.outer())
                    .gpu_threads    (F.inner(0), F.outer(0).split_var())
                    .gpu_blocks     (F.outer(0));

                cerr << F << endl;
                F.compile_jit("stmt.html");
            }
            else {
                //vector<RecFilter> fc = F.cascade_by_dimension();
            }

            runtime[j] = F.profile(iter);

            if (!nocheck) {
                check<float>(F, filter_coeff, image);
            }
        }

        cerr << width << "\t" << runtime[0] << "\t" << runtime[1] << "\t" << runtime[2] << endl;
        log  << width
            << "\t" << throughput(runtime[0], width*width)
            << "\t" << throughput(runtime[1], width*width)
            << "\t" << throughput(runtime[2], width*width) << endl;
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
