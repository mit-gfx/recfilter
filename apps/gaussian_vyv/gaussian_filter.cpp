/**
 *
 * \file gaussian_filter.cpp
 *
 * Gaussian blur using IIR filters:
 * - 3rd order x-y overlapped
 * - 1st and 2nd order x-y overlapped
 * - 3rd order x and 3rd order y
 * - 1st order x-y overlapped, 2nd order x and 2nd order y
 */



#include <iostream>
#include <Halide.h>

#include "recfilter.h"
#include "timing.h"
#include "iir_coeff.h"

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

    int iter       = args.iterations;
    int tile_width = args.block;
    int min_w      = args.min_width;
    int max_w      = args.max_width;
    int inc_w      = tile_width;

    vector<string> filter_names = {
        "Gaussian_3",
        "Gaussian_12",
        "Gaussian_3cascaded",
    };

    Log log("gaussian_filter.perflog");
    log << "Width";
    for (int j=0; j<filter_names.size(); j++) {
        log << "\t" << filter_names[j];
    }
    log << "\n";

    // Profile the filter for all image widths
    for (int in_w=min_w; in_w<=max_w; in_w+=inc_w) {
        int width = in_w;
        int height= in_w;

        Image<float> image = generate_random_image<float>(width,height);

        RecFilterDim x("x", width);
        RecFilterDim y("y", height);

        vector<float> runtime(filter_names.size(), 0.0f);

        // for (int j=0; j<filter_names.size(); j++) {
        for (int j=1; j<filter_names.size(); j++) {
            float sigma = 5.0;
            vector<float> W1 = gaussian_weights(sigma, 1);
            vector<float> W2 = gaussian_weights(sigma, 2);
            vector<float> W3 = gaussian_weights(sigma, 3);

            if (j==0)
            {
                RecFilter F(filter_names[j]);

                F.set_clamped_image_border();

                F(x,y) = image(x,y);
                F.add_filter(+x, W3);
                F.add_filter(-x, W3);
                F.add_filter(+y, W3);
                F.add_filter(-y, W3);

                F.split_all_dimensions(tile_width);

                if (F.target().has_gpu_feature()) {
                    int n_scans  = 4;
                    int ws       = 32;
                    int unroll_w = ws/4;
                    int intra_tiles_per_warp = ws / (4*n_scans);
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

                runtime[j] = F.profile(iter);
            }

            if (j==1)
            {
                RecFilter F(filter_names[j]);

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

                for (int k=0; k<fc.size(); k++) {
                    RecFilter f = fc[k];

                    f.split_all_dimensions(tile_width);

                    if (f.target().has_gpu_feature()) {
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
                }

                runtime[j] = fc[fc.size()-1].profile(iter);
            }

            if (j==2)
            {
            }
        }

        cerr << width;
        log  << width;
        for (int j=0; j<filter_names.size(); j++) {
            cerr << "\t" << runtime[j];
            log << "\t" << throughput(runtime[j], width*width);
        }
        cerr << endl;
        log << " " << endl;
    }

    return 0;
}

