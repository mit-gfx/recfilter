#include <iostream>
#include <Halide.h>

#include "../../lib/split.h"

#include "../../lib/gaussian_weights.h"

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


Func gaussian_blur(Func I, float sigma, Expr width, Expr height, int tile, string name);

int main(int argc, char **argv) {
    Arguments args("lshist_median", argc, argv);

    bool nocheck = args.nocheck;
    int  width   = args.width;
    int  height  = args.width;
    int  tile    = args.block;

    Image<float> random_image = generate_random_image<float>(width,height);
    ImageParam image(type_of<float>(), 2);
    image.set(random_image);

    // ----------------------------------------------------------------------------------------------

    Var x("x");
    Var y("y");

    // bounds
    Expr iw = image.width()-1;
    Expr ih = image.height()-1;

    // bins of the locally smoothed histogram
    vector<Func> Gauss;

    for (int i=0; i<NUM_BINS; i++) {
        string name = Internal::int_to_string(i);
        Func L("L_" + name);

        // pass image through histogram lookup table
        L(x,y) = gaussIntegral(image(clamp(x,0,iw),clamp(y,0,ih)), BIN_CENTER(i), HIST_SIGMA);

        // Gaussian filter using direct cognex blur
        Func G = gaussian_blur(L, GAUSS_SIGMA,
                image.width(), image.height(), tile, "G_"+name);

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
    Median.gpu_tile(x,y,WARP_SIZE,MAX_THREADS);

    // ----------------------------------------------------------------------------------------------

    // realize the median filter

    cerr << "\nJIT compilation ... " << endl;
    Median.compile_jit();

    Buffer hl_out_buff(type_of<float>(), width,height);
    {
        Timer t("Running ... ");
        Median.realize(hl_out_buff);
        hl_out_buff.free_dev_buffer();
    }
    hl_out_buff.copy_to_host();
    hl_out_buff.free_dev_buffer();

    // ----------------------------------------------------------------------------------------------

    if (!nocheck) {
        cerr << "\nChecking difference ... " << endl;
        Image<float> hl_out(hl_out_buff);
    }

    return 0;
}


Func gaussian_blur(Func I, float sigma, Expr width, Expr height, int tile, string name) {
    int   box  = gaussian_box_filter(3, sigma); // approx Gaussian with 3 box filters
    float norm = std::pow(box, 3*2);            // normalizing factor

    int num_scans = 2;
    int order     = 3;

    Image<float> W(num_scans,order);
    W(0,0) = 3.0f; W(0,1) = -3.0f; W(0,2) = 1.0f;
    W(1,0) = 3.0f; W(1,1) = -3.0f; W(1,2) = 1.0f;

    Func S(name);

    RDom rx(0, width, "rx");
    RDom ry(0, height,"ry");

    Var x("x"),   y("y");
    Var xi("xi"), yi("yi");
    Var xo("xo"), yo("yo");

    RDom rxi(0, tile, "rxi");
    RDom ryi(0, tile, "ryi");

    // convolve image with third derivative of three box filters
    S(x,y) =
        (1.0f /norm) * I(x+0*box, y+0*box) +
        (-3.0f/norm) * I(x+1*box, y+0*box) +
        (3.0f /norm) * I(x+2*box, y+0*box) +
        (-3.0f/norm) * I(x+0*box, y+1*box) +
        (9.0f /norm) * I(x+1*box, y+1*box) +
        (-9.0f/norm) * I(x+2*box, y+1*box) +
        (3.0f /norm) * I(x+0*box, y+2*box) +
        (-9.0f/norm) * I(x+1*box, y+2*box) +
        (9.0f /norm) * I(x+2*box, y+2*box);

    // triple integral via third order filter
    S(rx,y) = S(rx,y) +
        select(rx>0, W(0,0)*S(max(0,rx-1),y), 0.0f) +
        select(rx>1, W(0,1)*S(max(0,rx-2),y), 0.0f) +
        select(rx>2, W(0,2)*S(max(0,rx-3),y), 0.0f);

    S(x,ry) = S(x,ry) +
        select(ry>0, W(1,0)*S(x,max(0,ry-1)), 0.0f) +
        select(ry>1, W(1,1)*S(x,max(0,ry-2)), 0.0f) +
        select(ry>2, W(1,2)*S(x,max(0,ry-3)), 0.0f);

    //split(S,W,
    //        Internal::vec(  0,  1),
    //        Internal::vec(  x,  y),
    //        Internal::vec( xi, yi),
    //        Internal::vec( xo, yo),
    //        Internal::vec( rx, ry),
    //        Internal::vec(rxi,ryi),
    //        Internal::vec(order,order));
    //
    //inline_function(S, "I");


    // schedule
    if (S.is_reduction()) {
        S.compute_root();
        S.gpu_tile(x,y,WARP_SIZE,MAX_THREADS/WARP_SIZE);
        S.update(0).gpu_tile(y,MAX_THREADS);
        S.update(1).gpu_tile(x,MAX_THREADS);
    } else {
        Target target = get_jit_target_from_environment();
        if (target.has_gpu_feature() || (target.features & Target::GPUDebug)) {
            map<string,Func> funcs = extract_func_calls(S);
            for (map<string,Func>::iterator f=funcs.begin(); f!=funcs.end(); f++) {
                f->second.compute_root();
            }
        }
    }

    return S;
}
