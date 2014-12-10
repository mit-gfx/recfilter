#include <iostream>
#include <cstdio>
#include <cmath>

#include <Halide.h>

#include "gaussian_weights.h"
#include "recfilter.h"

using namespace Halide;

using std::vector;
using std::string;
using std::cerr;
using std::endl;

void cpu_schedule(RecFilter& F, int width) {
    F.intra_schedule().compute_in_global()
        .split    (F.full(0), 4)
        .vectorize(F.full(0).split_var())
        .parallel (F.full(0))
        ;

//    F.intra_schedule().compute_in_global()
//        .reorder_storage(F.full(), F.inner(), F.outer())
//        .reorder  (F.inner_scan(), F.full(), F.outer())
//        .split    (F.full(0), 8)
//        .vectorize(F.full(0).split_var())
//        .parallel (F.full(0))
//        .parallel (F.outer(0));
//
//    F.inter_schedule().compute_in_global()
//        .reorder_storage(F.full(), F.tail(), F.outer())
//        .reorder (F.outer_scan(), F.tail(), F.full())
//        .vectorize(F.tail())
//        .split    (F.full(0), 8)
//        .vectorize(F.full(0).split_var())
//        .parallel (F.full(0));
}

int main(int argc, char **argv) {
    Arguments args(argc, argv);

    bool nocheck = true; //args.nocheck;
    int  width   = args.width;
    int  height  = args.width;
    int  tile_width = args.block;
    int  iterations = args.iterations;

    Image<float> image = generate_random_image<float>(width,height);

    // ----------------------------------------------------------------------------------------------

    double sigma = 5.0;
    vector<double> W3 = gaussian_weights(sigma, 3);

    RecFilterDim x("x", width);
    RecFilterDim y("y", height);

    RecFilter F;

    F.set_clamped_image_border();

    F(x, y) = image(clamp(x,0,image.width()-1), clamp(y,0,image.height()-1));

    F.add_filter(+x, W3);
    F.add_filter(-x, W3);
    F.add_filter(+y, W3);
    F.add_filter(-y, W3);

    vector<RecFilter> cascaded_filters = F.cascade({2,3}, {0,1});

    RecFilter F1 = cascaded_filters[0];
    RecFilter F2 = cascaded_filters[1];

//    F1.split(y, tile_width);
//    F2.split(x, tile_width);

    // ----------------------------------------------------------------------------------------------

    if (F.target().has_gpu_feature()) {
    } else {
        cpu_schedule(F1, width);
        cpu_schedule(F2, width);
    }

    cerr << F1 << endl;
    cerr << F2 << endl;

    // ----------------------------------------------------------------------------------------------
    cerr << "width\ttime (ms)\truns" << endl;

    F2.compile_jit("hl_stmt.html");

    Buffer out(type_of<float>(), width, height);
    double time = F.realize(out, iterations);
    cerr << width << "\t" << time << endl;

    // ----------------------------------------------------------------------------------------------

    if (!nocheck) {
        cerr << "\nChecking difference ...\n" << endl;
        Image<float> hl_out(out);
        Image<float> ref = reference_gaussian<float>(image, sigma);
        cerr << "Difference with true Gaussian \n" << CheckResult<float>(ref,hl_out) << endl;
    }

    return 0;
}
