#include <iostream>
#include <Halide.h>

#include <recfilter.h>
#include <iir_coeff.h>

#include "image_io.h"

using namespace Halide;

using std::vector;
using std::string;
using std::cerr;
using std::endl;

#define NUM_BINS      (15)
#define BIN_WIDTH     (1.0f/float(NUM_BINS))
#define BIN_CENTER(i) (BIN_WIDTH*(float(i)+0.5f))
#define HIST_SIGMA    (BIN_WIDTH)
#define GAUSS_SIGMA   (5.0f)

Func smooth(Func I, RecFilterDim x, RecFilterDim y, float sigma);

int main(int argc, char **argv) {
    string filename;
    if (argc==2) {
        filename = argv[1];
    } else {
        cerr << "Usage: gaussian_filter_demo [name of png file]" << endl;
        return EXIT_FAILURE;
    }

    Image<uint8_t> input = load<uint8_t>(filename);

    int width   = input.width();
    int height  = input.height();
    int channels= input.channels();

    // ----------------------------------------------------------------------------------------------

    RecFilterDim x("x", width);
    RecFilterDim y("y", height);

    Expr r = cast<float>(input(x,y,0)) / 255.0f;
    Expr g = cast<float>(input(x,y,1)) / 255.0f;
    Expr b = cast<float>(input(x,y,2)) / 255.0f;

    // bins of the locally smoothed histogram
    vector<Func> Hist;

    for (int i=0; i<NUM_BINS; i++) {
        string name = Internal::int_to_string(i);
        Func L("L" + name);

        // create the histogram
        // pass image through histogram lookup table
        L(x,y) = Tuple(
            gaussIntegral(r, BIN_CENTER(i), HIST_SIGMA),
            gaussIntegral(g, BIN_CENTER(i), HIST_SIGMA),
            gaussIntegral(b, BIN_CENTER(i), HIST_SIGMA));

        // smooth the histogram
        // Gaussian filter using tile width = 32
        Func G = smooth(L, x, y, GAUSS_SIGMA);

        // add to the overall list
        Hist.push_back(G);
    }

    // compute the median from the smoothed histograms
    vector<Expr> median_rgb;
    for (int j=0; j<channels; j++) {
        Expr median;
        Expr g0     = Hist[0]         (x,y)[j];
        Expr gn     = Hist[NUM_BINS-1](x,y)[j];
        Expr target = g0 + 0.5f*(gn - g0);
        for (int i=0; i<NUM_BINS-1; i++) {
            Expr gi     = Hist[i]  (x,y)[j];
            Expr gi1    = Hist[i+1](x,y)[j];
            Expr frac   = (target-gi)/(gi1-gi);
            Expr cond   = (gi<target && gi1>=target);
            Expr value  = BIN_CENTER(i) + frac*BIN_WIDTH;

            median = (i==0 ? value : select(cond, value, median));
        }
        median_rgb.push_back(median);
    }

    RecFilter Median("Median");
    Median(x,y) = median_rgb;
    Median.gpu_auto_schedule(128, 32);

    // assemble channels again and save result
    // should also be done on the GPU
    {
        Var i("i"), j("j"), c("c");

        Func M = Median.as_func();

        Func Result;
        Result(i,j,c) = select(c==0, M(i,j)[0], c==1, M(i,j)[1], M(i,j)[2]);

        Buffer buff(type_of<float>(), width, height, channels);
        Result.realize(buff);
        Image<float> output(buff);

        save(output, "out.png");
    }

    return EXIT_SUCCESS;
}


Func smooth(Func I, RecFilterDim x, RecFilterDim y, float sigma) {
    vector<float> W3 = gaussian_weights(sigma,3);

    RecFilter S("Smooth_"+I.name());

    S.set_clamped_image_border();

    S(x,y) = I(x,y);
    S.add_filter(+x, W3);
    S.add_filter(-x, W3);
    S.add_filter(+y, W3);
    S.add_filter(-y, W3);

    vector<RecFilter> fc = S.cascade_by_dimension();

    fc[0].split_all_dimensions(32);     // tile width = 32
    fc[1].split_all_dimensions(32);
    fc[0].gpu_auto_schedule(128);       // max threads per CUDA tile = 128
    fc[1].gpu_auto_schedule(128);

    return fc[1].as_func();
}
