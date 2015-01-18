/**
 * \file audio_biquads.cpp
 *
 * Performance comparison between tiled and non-tiled 1D filters on CPU
 * for multiple overlapped biquads (second order filters)
 */

#include <iostream>
#include <Halide.h>

#include "recfilter.h"
#include "timing.h"

#define MAX_NUM_SCANS 30

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

    // dummy coeff, for performance comparison only
    std::vector<float> coeffs = {1.0f, 0.1f, 0.1f};

    Log log_naive("audio_biquads.nontiled.perflog");
    Log log_tiled("audio_biquads.tiled.perflog");

    for (int num_scans=1; num_scans<=MAX_NUM_SCANS; num_scans++) {
        RecFilterDim x("x", width);

        // non-tiled implementation
        {
            RecFilter F("R_nontiled");
            F(x) = image(clamp(x,0,width-1));
            for (int i=0; i<=num_scans; i++) {
                F.add_filter(+x, coeffs);
            }
            F.compute_globally();
            time_naive.push_back(F.profile(iterations));
        }

        // tiled implementation
        {
            int vw = 8;

            RecFilter F("R_tiled");
            F(x) = image(clamp(x,0,width-1));
            for (int i=0; i<=num_scans; i++) {
                F.add_filter(+x, coeffs);
            }
            F.split(x, tile_width);

            F.intra_schedule().compute_locally() .vectorize(F.inner(0),vw).parallel(F.outer(0));
            F.inter_schedule().compute_globally().vectorize(F.inner(0),vw);

            time_tiled.push_back(F.profile(iterations));
        }

        log_naive << num_scans << "\t"
                  << time_naive[time_naive.size()-1] << "\t"
                  << throughput(time_naive[time_naive.size()-1], width) << endl;

        log_tiled << num_scans << "\t"
                  << time_tiled[time_tiled.size()-1] << "\t"
                  << throughput(time_tiled[time_tiled.size()-1], width) << endl;
    }

    return 0;
}
