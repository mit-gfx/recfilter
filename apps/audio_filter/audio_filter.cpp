#include <iostream>
#include <Halide.h>

#include "recfilter.h"

using namespace Halide;

using std::vector;
using std::string;
using std::cerr;
using std::endl;

#define CHANNELS 1
#define ORDER    9

int main(int argc, char **argv) {
    Arguments args(argc, argv);

    int width      = args.width;
    int tile_width = args.block;
    int iterations = args.iterations;

    Image<float> image = generate_random_image<float>(width,CHANNELS);

    vector<double> coeffs(ORDER+1, 0.1);
    coeffs[0] = 1.0;

    cerr << "Order = " << ORDER << ", array size = " << width << " channels = " << CHANNELS << endl;

    // Tiled implementation
    {
        Buffer out;
        Image<float> a, b;

        RecFilterDim x("x", width);
        RecFilterDim c("c", CHANNELS);

        RecFilter F;
        F(x,c) = image(clamp(x,0,width-1), c);
        F.add_filter(+x, coeffs);

        if (F.target().has_gpu_feature()) {
            cerr << "Filter only designed for CPU, change HL_JIT_TARGET to CPU target" << endl;
            assert(false);
        }

        F.intra_schedule().compute_globally();
        F.compile_jit("nontiled.html");
        double time1 = F.realize(out, iterations);

        F.split(x, tile_width);
        F.intra_schedule().compute_locally() ;
        F.inter_schedule().compute_globally();
        F.compile_jit("tiled.html");
        double time2 = F.realize(out, iterations);

        cerr << "Naive: " << time1 << " ms" << endl;
        cerr << "Tiled: " << time2 << " ms" << endl;
    }
    return 0;

    // C++ non tiled implementation
    {
        unsigned long start, end, total;
        for (int n=0; n<iterations; n++) {
            Image<float> out(image.width(),CHANNELS);

            start = RecFilter::millisecond_timer();
            for (int j=0; j<CHANNELS; j++) {
                for (int i=0; i<image.width(); i++) {
                    out(i,j) = image(i,j);
                }
            }
            for (int j=0; j<CHANNELS; j++) {
                for (int i=0; i<image.width(); i++) {
                    float temp = 0.0f;
                    for (int k=0; k<ORDER+1; k++) {
                        temp += (i-k>=0 ? coeffs[k]*out(i-k,j) : 0.0f);
                    }
                    out(i,j) = temp;
                }
            }
            end = RecFilter::millisecond_timer();
            total += (end-start);
            cerr << out << endl;
        }

        cerr << "C++ naive: " << double(total)/double(iterations) << " ms" << endl;
    }

    return 0;
}
