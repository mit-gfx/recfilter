/**
 * \file summed_table.cpp
 *
 * Summed area table using overlapped computation
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


int main(int argc, char **argv) {
    Arguments args(argc, argv);

    bool nocheck   = args.nocheck;
    int iter       = args.iterations;
    int tile_width = args.block;
    int min_w      = args.min_width;
    int max_w      = args.max_width;
    int inc_w      = tile_width;

    Log log("summed_table.perflog");
    log << "Width\tOurs" << endl;

    // Profile the filter for all image widths
    for (int in_w=min_w; in_w<=max_w; in_w+=inc_w) {
        int width = in_w;
        int height= in_w;

        Image<float> image = generate_random_image<float>(width,height);

        RecFilterDim x("x", width);
        RecFilterDim y("y", height);

        RecFilter F;

        F(x,y) = image(x,y);

        F.add_filter(+x, {1.0, 1.0});
        F.add_filter(+y, {1.0, 1.0});

        F.split(x, tile_width, y, tile_width);

        // ---------------------------------------------------------------------

        int tiles_per_warp = 2;
        int unroll_w       = 8;

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
            .split          (F.outer(0), tiles_per_warp)
            .reorder        (F.inner_scan(), F.tail(), F.outer(0).split_var(), F.inner(), F.outer())
            .gpu_threads    (F.inner(0), F.outer(0).split_var())
            .gpu_blocks     (F.outer(0), F.outer(1));

        F.inter_schedule().compute_globally()
            .reorder_storage(F.inner(), F.tail(), F.outer())
            .unroll         (F.outer_scan())
            .split          (F.outer(0), tiles_per_warp)
            .reorder        (F.outer_scan(), F.tail(), F.outer(0).split_var(), F.inner(), F.outer())
            .gpu_threads    (F.inner(0), F.outer(0).split_var())
            .gpu_blocks     (F.outer(0));

        // cerr << F << endl;
        F.compile_jit("stmt.html");

        float time = F.profile(iter);

        cerr << width << "\t" << time << " ms" << endl;
        log  << width << "\t" << RecFilter::throughput(time,width*width) << endl;

        // ---------------------------------------------------------------------

        if (!nocheck) {
            cerr << "\nChecking difference ... " << endl;
            Realization out = F.realize();

            Image<float> hl_out(out);
            Image<float> ref(width,height);

            for (int y=0; y<height; y++) {
                for (int x=0; x<width; x++) {
                    ref(x,y) = image(x,y);
                }
            }
            for (int y=0; y<height; y++) {
                for (int x=1; x<width; x++) {
                    ref(x,y) += ref(x-1,y);
                }
            }
            for (int y=1; y<height; y++) {
                for (int x=0; x<width; x++) {
                    ref(x,y) += ref(x,y-1);
                }
            }

            cout << CheckResult<float>(ref,hl_out) << endl;
        }
    }

    return 0;
}
