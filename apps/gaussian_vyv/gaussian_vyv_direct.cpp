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

    cascaded_filters[0].split(y, tile_width);
    cascaded_filters[1].split(x, tile_width);

    // ----------------------------------------------------------------------------------------------

    F.intra_schedule().compute_in_global().parallel(F.full(0));

    for (int i=0; i<cascaded_filters.size(); i++) {
        RecFilter f = cascaded_filters[i];
        if (f.target().has_gpu_feature()) {
        } else {

            //f.intra_schedule().compute_in_global()
                //.split    (f.full(0), 4)
                //.vectorize(f.full(0).split_var())
                //.parallel (f.full(0))
                //;

            f.intra_schedule().compute_in_global()
                .reorder_storage(f.full(), f.inner(), f.outer())
                .reorder  (f.inner_scan(), f.full(), f.outer())
                .split    (f.full(0), 8)
                .vectorize(f.full(0).split_var())
                .parallel (f.full(0))
                .parallel (f.outer(0));

            f.inter_schedule().compute_in_global()
                .reorder_storage(f.full(), f.tail(), f.outer())
                .reorder (f.outer_scan(), f.tail(), f.full())
                .vectorize(f.tail())
                .split    (f.full(0), 8)
                .vectorize(f.full(0).split_var())
                .parallel (f.full(0));
        }

        cerr << f << endl;
    }


    // ----------------------------------------------------------------------------------------------

    cerr << "width\ttime (ms)" << endl;

    RecFilter f = cascaded_filters[cascaded_filters.size()-1];

    f.compile_jit("hl_stmt.html");

    Buffer out(type_of<float>(), width, height);
    double time = f.realize(out, iterations);
    cerr << width << "\t" << time << endl;
    time = F.realize(out, iterations);
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
