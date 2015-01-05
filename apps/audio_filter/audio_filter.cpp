#include <iostream>
#include <Halide.h>

#include "recfilter.h"

using namespace Halide;

using std::cerr;
using std::endl;

#define CHANNELS 1
#define ORDER    3

int main(int argc, char **argv) {
    Arguments args(argc, argv);

    int width      = args.width;
    int tile_width = args.block;
    int iterations = args.iterations;

    Image<float> image = generate_random_image<float>(width,CHANNELS);

    std::vector<float> coeffs(ORDER+1, 1);
    coeffs[0] = 1.0;

    Buffer out;

    float time1, time2;

    RecFilterDim x("x", width);
    RecFilterDim c("c", CHANNELS);

    // non-tiled implementation
    {
        RecFilter F("R_nontiled");
        F(x,c) = image(x,c);
        F.add_filter(+x, coeffs);
        F.intra_schedule().compute_globally();
        F.compile_jit("nontiled.html");
        time1 = F.realize(out, iterations);
    }

    // tiled implementation
    {
        RecFilter F("R_tiled");
        F(x,c) = image(x,c);
        F.add_filter(+x, coeffs);
        F.split(x, tile_width);
        F.intra_schedule().compute_locally() ;
        F.inter_schedule().compute_globally();
        F.compile_jit("tiled.html");
        time2 = F.realize(out, iterations);
    }

    std::cerr << "\nOrder = " << ORDER << ", array = (" << width << ", " << CHANNELS << ")\n"
              << "Naive: " << time1 << " ms\n"
              << "Tiled: " << time2 << " ms\n" << std::endl;

    return 0;
}
