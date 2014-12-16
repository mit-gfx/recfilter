#include <iostream>
#include <Halide.h>

#include "recfilter.h"

using namespace Halide;

using std::vector;
using std::string;
using std::cerr;
using std::endl;


int main(int argc, char **argv) {
    Arguments args(argc, argv);

    int width   = args.width;
    int tile_width = args.block;
    int iterations = args.iterations;

    Image<float> image = generate_random_image<float>(width);

    vector<double> coeffs = {
        1.0, // feedforward
        1.0,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
    };
    int order = coeffs.size()-1;

    RecFilterDim x("x", width);

    RecFilter filter;
    filter(x) = image(clamp(x,0,width-1));
    filter.add_filter(+x, coeffs);

    if (!filter.target().has_gpu_feature()) {
        {
            cerr << "Non-tiled version, order = " << order << ", array size = " << width;

            filter.intra_schedule().compute_globally();

            filter.compile_jit();

            Buffer out(type_of<float>(), width);
            double time = filter.realize(out, iterations);
            cerr << " : " << time << " ms" << endl;
        }

        {
            cerr << "Tiled version, order = " << order << ", array size = " << width;
            filter.split(x, tile_width);

            filter.intra_schedule().compute_globally().parallel(filter.outer(0));
            filter.inter_schedule().compute_globally();

            filter.compile_jit();

            Buffer out(type_of<float>(), width);
            double time = filter.realize(out, iterations);
            cerr << " : " << time << " ms" << endl;
        }
    } else {
        cerr << "Filter only designed for CPU, change HL_JIT_TARGET to CPU target" << endl;
        assert(false);
    }

    // ----------------------------------------------------------------------------------------------

    return 0;
}
