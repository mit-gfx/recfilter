#include <iostream>
#include <cstdio>
#include <cmath>

#include <Halide.h>

#include "../../lib/gaussian_weights.h"
#include "../../lib/split.h"

#define MAX_THREAD 192

using namespace Halide;

using std::vector;
using std::string;
using std::cerr;
using std::endl;

int main(int argc, char **argv) {
    Arguments args("gauss_cognex_overlapped", argc, argv);

    bool nocheck = args.nocheck;
    int  width  = args.width;
    int  height = args.width;
    int  tile   = args.block;

    Image<float> random_image = generate_random_image<float>(width,height);

    ImageParam image(type_of<float>(), 2);
    image.set(random_image);

    // ----------------------------------------------------------------------------------------------

    float sigma = 16.0f;
    int   box   = gaussian_box_filter(3, sigma); // approx Gaussian with 3 box filters
    float norm  = std::pow(box, 3*2);            // normalizing factor

    // ----------------------------------------------------------------------------------------------

    int order     = 2;
    int num_scans = 4;

    Image<float> W(num_scans,order);
    W(0,0) = 1.0f; W(0,1) =  0.0f;
    W(1,0) = 2.0f; W(1,1) = -1.0f;
    W(2,0) = 1.0f; W(2,1) =  0.0f;
    W(3,0) = 2.0f; W(3,1) = -1.0f;

    Var x("x"),   y("y");
    Var xi("xi"), yi("yi");
    Var xo("xo"), yo("yo");

    RDom rx (0, image.width(), "rx");
    RDom ry (0, image.height(),"ry");
    RDom rz (0, image.width(), "rz");
    RDom rw (0, image.height(),"rw");

    RDom rxi(0, tile, "rxi");
    RDom ryi(0, tile, "ryi");
    RDom rzi(0, tile, "rzi");
    RDom rwi(0, tile, "rwi");

    Func I("I");
    Func S("S");
    Func G("G");

    I(x,y) = image(clamp(x,0,image.width()-1), clamp(y,0,image.height()-1));

    // convolve image with third derivative of three box filters
    S(x,y) =
            ( 1.0f * I(x+0*box,y+0*box) +
             -3.0f * I(x+1*box,y+0*box) +
              3.0f * I(x+2*box,y+0*box) +
             -1.0f * I(x+3*box,y+0*box) +
             -3.0f * I(x+0*box,y+1*box) +
              9.0f * I(x+1*box,y+1*box) +
             -9.0f * I(x+2*box,y+1*box) +
              3.0f * I(x+3*box,y+1*box) +
              3.0f * I(x+0*box,y+2*box) +
             -9.0f * I(x+1*box,y+2*box) +
              9.0f * I(x+2*box,y+2*box) +
             -3.0f * I(x+3*box,y+2*box) +
             -1.0f * I(x+0*box,y+3*box) +
              3.0f * I(x+1*box,y+3*box) +
             -3.0f * I(x+2*box,y+3*box) +
              1.0f * I(x+3*box,y+3*box)) / norm;

    // triple integral by applying a first followed by second order filter
    S(rx,y) += (select(rx>0, W(0,0)*S(max(0,rx-1),y), 0.0f));
    S(ry,y) += (select(ry>0, W(1,0)*S(max(0,ry-1),y), 0.0f)
            +   select(ry>1, W(1,1)*S(max(0,ry-2),y), 0.0f));

    S(x,rz) += (select(rz>0, W(2,0)*S(x,max(0,rz-1)), 0.0f));
    S(x,rw) += (select(rw>0, W(3,0)*S(x,max(0,rw-1)), 0.0f)
            +   select(rw>1, W(3,1)*S(x,max(0,rw-2)), 0.0f));

    G(x,y) = S(x,y);

    split(S,W,
            Internal::vec(  0,  0,  1,  1),
            Internal::vec(  x,  x,  y,  y),
            Internal::vec( xi, xi, yi, yi),
            Internal::vec( xo, xo, yo, yo),
            Internal::vec( rx, ry, rz, rw),
            Internal::vec(rxi,ryi,rzi,rwi),
            Internal::vec(order,order,order,order), true);

    inline_function(G, "I");
    inline_function(G, "S");

    // ----------------------------------------------------------------------------------------------

    map<string,Func> functions = extract_func_calls(S);
    map<string,Func>::iterator f    = functions.begin();
    map<string,Func>::iterator fend = functions.end();
    for (; f!=fend; f++) {
        cerr << f->second << endl;
        f->second.compute_root();
    }

    Target target = get_jit_target_from_environment();
    if (target.has_gpu_feature() || (target.features & Target::GPUDebug)) {
    }

    // ----------------------------------------------------------------------------------------------

    cerr << "\nJIT compilation ... " << endl;
    G.compile_jit();

    Buffer hl_out_buff(type_of<float>(), width,height);
    {
        Timer t("Running ... ");
        G.realize(hl_out_buff);
    }
    hl_out_buff.copy_to_host();
    hl_out_buff.free_dev_buffer();

    // ----------------------------------------------------------------------------------------------

    if (!nocheck) {
        cerr << "\nChecking difference ...\n" << endl;
        Image<float> hl_out(hl_out_buff);
        Image<float> ref = reference_gaussian(random_image, sigma);
        cerr << "Difference with true Gaussian \n" << CheckResultVerbose(ref,hl_out) << endl;
    }

    return 0;
}
