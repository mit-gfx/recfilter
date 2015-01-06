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

    bool  nocheck = args.nocheck;
    int   width   = args.width;
    int   height  = args.width;
    int   tile_width = args.block;
    int   iter    = args.iterations;

    Image<float> random_image = generate_random_image<float>(width,height);

    ImageParam image(type_of<float>(), 2);
    image.set(random_image);

    // ----------------------------------------------------------------------------------------------

    RecFilterDim x("x", width);
    RecFilterDim y("y", height);

    RecFilter F;

    F(x,y) = image(x,y);

    F.add_filter(+x, {1.0, 1.0});
    F.add_filter(+y, {1.0, 1.0});

    F.split(x, tile_width, y, tile_width);

    // ----------------------------------------------------------------------------------------------

    int tiles_per_warp_inter = 4;
    int tiles_per_warp_intra = 4;
    int unroll_w       = 8;
    int vectorize      = 4;

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
        .split          (F.outer(0), tiles_per_warp_intra)
        .reorder        (F.inner_scan(), F.inner(), F.tail(), F.outer(0).split_var(), F.outer())
        .gpu_threads    (F.inner(0), F.outer(0).split_var())
        .gpu_blocks     (F.outer(0), F.outer(1));

    F.inter_schedule().compute_globally()
        .reorder_storage(F.inner(), F.tail(), F.outer())
        .unroll         (F.outer_scan())
        .split          (F.outer(0), tiles_per_warp_inter)
        .reorder        (F.outer_scan(), F.inner(), F.tail(), F.outer(0).split_var(), F.outer())
        .gpu_threads    (F.inner(0), F.outer(0).split_var())
        .gpu_blocks     (F.outer(0));

    cerr << F << endl;

    cerr << "\nJIT compilation ... " << endl;
    F.compile_jit("stmt.html");

    cerr << "\nRunning ... " << endl;
    Buffer out(type_of<float>(), width, height);
    float time = F.realize(out, iter);
    cerr << width << "\t" << time << endl;

    // ----------------------------------------------------------------------------------------------

    if (!nocheck) {
        cerr << "\nChecking difference ... " << endl;
        Image<float> hl_out(out);
        Image<float> ref(width,height);

        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {
                ref(x,y) = random_image(x,y);
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

    return 0;
}
