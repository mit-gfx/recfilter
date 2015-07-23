#include <iostream>
#include <Halide.h>

#include "recfilter.h"

#include "iir_coeff.h"

#define WARP_SIZE   32
#define MAX_THREADS 192

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


Func gaussian_blur(Func I, float sigma, Expr width, Expr height, int tile_width, string name);

int main(int argc, char **argv) {
    Arguments args(argc, argv);

    bool nocheck = args.nocheck;
    int  width   = args.width;
    int  height  = args.width;
    int  tile_width = args.block;

    Image<float> random_image = generate_random_image<float>(width,height);
    ImageParam image(type_of<float>(), 2);
    image.set(random_image);

    // ----------------------------------------------------------------------------------------------

    RecFilterDim x("x", width);
    RecFilterDim y("y", height);

    // bounds
    Expr iw = image.width()-1;
    Expr ih = image.height()-1;

    // bins of the locally smoothed histogram
    vector<Func> Gauss;

    for (int i=0; i<NUM_BINS; i++) {
        string name = Internal::int_to_string(i);
        Func L("L_" + name);

        // pass image through histogram lookup table
        L(x,y) = gaussIntegral(image(x,y), BIN_CENTER(i), HIST_SIGMA);

        // Gaussian filter using direct cognex blur
        Func G = gaussian_blur(L, GAUSS_SIGMA,
                image.width(), image.height(), tile_width, "G_"+name);

        // add to the overall list
        Gauss.push_back(G);
    }

    Func Median("Median");
    vector<Expr> median;
    for (int i=0; i<NUM_BINS-1; i++) {
        Expr target = Gauss[0](x,y) + 0.5f*(Gauss[NUM_BINS-1](x,y)-Gauss[0](x,y));
        Expr frac   = (target-Gauss[i](x,y))/(Gauss[i+1](x,y)-Gauss[i](x,y));
        Expr cond   = (Gauss[i](x,y)<target && Gauss[i+1](x,y)>=target);
        Expr value  = BIN_CENTER(i) + frac*BIN_WIDTH;
        if (i==0) {
            median.push_back(value);
        } else {
            median.push_back(select(cond, value, median[i-1]));
        }
    }
    Median(x,y) = median[median.size()-1];

    Median.compute_root();
    Median.gpu_tile(x.var(), y.var(), WARP_SIZE,MAX_THREADS);

    // ----------------------------------------------------------------------------------------------

    Realization out = Median.realize();

    // ----------------------------------------------------------------------------------------------

    if (!nocheck) {
        cerr << "\nChecking difference ... " << endl;
        Image<float> hl_out(out);
    }

    return 0;
}


Func gaussian_blur(Func I, float sigma, Expr width, Expr height, int tile_width, string name) {
    int   box  = gaussian_box_filter(3, sigma); // approx Gaussian with 3 box filters
    float norm = std::pow(box, 3*2);            // normalizing factor

    int num_scans = 2;
    int order     = 3;

    Image<float> W(num_scans,order);
    W(0,0) = 3.0f; W(0,1) = -3.0f; W(0,2) = 1.0f;
    W(1,0) = 3.0f; W(1,1) = -3.0f; W(1,2) = 1.0f;

    Func S;

    // TODO: add the fastest Gaussian blur

    return S;
}
