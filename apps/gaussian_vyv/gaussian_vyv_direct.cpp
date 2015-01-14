#include <iostream>
#include <cstdio>
#include <cmath>

#include <Halide.h>

#include "iir_coeff.h"
#include "recfilter.h"

using namespace Halide;

using std::vector;
using std::string;
using std::cerr;
using std::endl;

#define VECTORIZE_WIDTH 8

int main(int argc, char **argv) {
    Arguments args(argc, argv);

    int  width   = args.width;
    int  height  = args.width;
    int  tile_width = args.block;
    int  iterations = args.iterations;

    Image<float> image = generate_random_image<float>(width,height);

    // ----------------------------------------------------------------------------------------------

    float sigma = 5.0;
    vector<float> W3 = gaussian_weights(sigma, 3);

    RecFilterDim x("x", width);
    RecFilterDim y("y", height);

    RecFilter F;

    F.set_clamped_image_border();

    F(x,y) = image(x,y);

    F.add_filter(+x, W3);
    F.add_filter(-x, W3);
    F.add_filter(+y, W3);
    F.add_filter(-y, W3);

    vector<RecFilter> cascaded_filters = F.cascade({0,1}, {2,3});

#define SPLIT 1
#if SPLIT
    cascaded_filters[0].split(x, tile_width);
    cascaded_filters[1].split(y, tile_width);
#endif

    // ----------------------------------------------------------------------------------------------

    F.intra_schedule().compute_globally()
        .split    (F.full(0), VECTORIZE_WIDTH)
        .vectorize(F.full(0).split_var())
        .parallel (F.full(0));

    for (int i=0; i<cascaded_filters.size(); i++) {
        RecFilter f = cascaded_filters[i];
        if (f.target().has_gpu_feature()) {
        } else {

#if SPLIT
            f.intra_schedule().compute_globally()
                .reorder_storage(f.full(), f.inner(), f.outer())
                .reorder  (f.inner_scan(), f.full(), f.outer())
                .split    (f.full(0), VECTORIZE_WIDTH)
                .vectorize(f.full(0).split_var())
                .parallel (f.full(0))
                .parallel (f.outer(0))
                ;

            f.inter_schedule().compute_globally()
                .reorder_storage(f.full(), f.tail(), f.outer())
                .reorder  (f.outer_scan(), f.tail(), f.full())
                .split    (f.full(0), VECTORIZE_WIDTH)
                .vectorize(f.full(0).split_var())
                .parallel (f.full(0))
                ;
#else
            f.intra_schedule().compute_globally()
                .split    (f.full(0), VECTORIZE_WIDTH)
                .vectorize(f.full(0).split_var())
                .parallel (f.full(0))
                ;
#endif
        }

        cerr << f << endl;
    }


    // ----------------------------------------------------------------------------------------------
    float time = 0.0;

    RecFilter f = cascaded_filters[cascaded_filters.size()-1];

    f.compile_jit("tiled.html");
    F.compile_jit("nontiled.html");

    time = F.profile(iterations);
    cerr << "non_tiled_direct\t" << width << "\t" << time << endl;

    time = f.profile(iterations);
    cerr << "tiled_xy_cascaded\t" << width << "\t" << time << endl;

    return 0;
}
