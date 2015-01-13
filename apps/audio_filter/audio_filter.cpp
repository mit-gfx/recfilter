#include <iostream>
#include <Halide.h>

#include "recfilter.h"
#include "timing.h"

#define MAX_ORDER 40

using namespace Halide;

using std::vector;
using std::cerr;
using std::endl;

int main(int argc, char **argv) {
    Arguments args(argc, argv);

    int width     = args.width;
    int tile_width = args.block;
    int iterations = args.iterations;

    Image<float> image = generate_random_image<float>(width);

    vector<float> time_naive;
    vector<float> time_tiled;

    Log log("audio_filter.perflog");
    log << "Order\tNaive\tTiled" << endl;

    for (int order=1; order<MAX_ORDER; order++) {
        std::vector<float> coeffs(order+1, 0.01);
        coeffs[0] = 1.0;

        Buffer out;

        RecFilterDim x("x", width);

        // non-tiled implementation
        {
            RecFilter F("R_nontiled");
            F(x) = image(clamp(x,0,width-1));
            F.add_filter(+x, coeffs);
            F.compute_globally();
            F.compile_jit("nontiled.html");
            time_naive.push_back(F.profile(iterations));
        }

        // tiled implementation
        {
            int vw = 8;

            RecFilter F("R_tiled");
            F(x) = image(clamp(x,0,width-1));
            F.add_filter(+x, coeffs);
            F.split(x, tile_width);

            F.intra_schedule().compute_locally() .vectorize(F.inner(0),vw).parallel(F.outer(0));
            F.inter_schedule().compute_globally().vectorize(F.inner(0),vw);

            F.compile_jit("tiled.html");
            time_tiled.push_back(F.profile(iterations));
        }

        cerr << "Order=" << order << " "
             << "naive " << time_naive[time_naive.size()-1] << " "
             << "tiled " << time_tiled[time_tiled.size()-1] << " ms" << endl;

        log << order << "\t"
            << throughput(time_naive[time_naive.size()-1], width) << "\t"
            << throughput(time_tiled[time_tiled.size()-1], width) << endl;
    }

    return 0;
}
