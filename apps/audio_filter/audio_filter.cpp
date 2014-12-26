#include <iostream>
#include <Halide.h>

#include "recfilter.h"

using namespace Halide;

using std::vector;
using std::string;
using std::cerr;
using std::endl;

void filter_channel(Image<float> in, Image<float> out, vector<double> coeff, int j);

int main(int argc, char **argv) {
    Arguments args(argc, argv);

    int width      = args.width;
    int height     = 1;
    int tile_width = args.block;
    int iterations = args.iterations;

    Image<float> image = generate_random_image<float>(width,height);

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

    cerr << "Order = " << order << ", array size = " << width << " channels = " << height << endl;

    double time = 0.0;

    // Tiled implementation
    {
        Buffer out;

        RecFilterDim x("x", width);
        RecFilterDim c("c", height);

        RecFilter F;
        F(x,c) = image(clamp(x,0,width-1),c);
        F.add_filter(+x, coeffs);

        if (F.target().has_gpu_feature()) {
            cerr << "Filter only designed for CPU, change HL_JIT_TARGET to CPU target" << endl;
            assert(false);
        } else {
            F.intra_schedule().compute_globally().parallel(F.full(0));
            F.compile_jit();
            time = F.realize(out, iterations);
            cerr << "Ours non-tiled: \t" << time << " ms" << endl;

            F.split(x, tile_width);
            F.intra_schedule().compute_locally() .parallel(F.full(0)).parallel(F.outer(0));
            F.inter_schedule().compute_globally().parallel(F.full(0));
            F.compile_jit();
            time = F.realize(out, iterations);
            cerr << "Ours tiled: \t" << time << " ms" << endl;
        }
    }

    // C++ non tiled implementation
    {
        for (int n=0; n<iterations; n++) {
            float* out = new float[image.width() * image.height()];

            unsigned long start = RecFilter::millisecond_timer();
#pragma omp parallel for
            for (int j=0; j<image.height(); j++) {
                for (int i=0; i<image.width(); i++) {
                    int idx = image.height()*j+i;
                    out[idx] = coeffs[0] * image(i,j);
                    for (int k=0; k<order; k++) {
                        int prev_idx = image.height()*j+i-k;
                        out[idx] += (i-k>=0 ? coeffs[k+1]*out[prev_idx] : 0.0f);
                    }
                }
            }
            unsigned long end = RecFilter::millisecond_timer();

            time += double(end-start);
            delete [] out;
        }
        cerr << "C++ non-tiled: \t" << time/double(iterations) << " ms" << endl;
    }

    return 0;
}
