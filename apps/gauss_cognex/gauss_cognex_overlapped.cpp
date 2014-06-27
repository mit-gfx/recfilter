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
    Arguments args("gauss_cognex", argc, argv);

    bool nocheck = args.nocheck;
    int  width  = args.width;
    int  height = args.width;
    int  tile   = args.block;

    float sigma = 16.0f;

    Image<float> random_image = generate_random_image<float>(width,height);

    ImageParam image(type_of<float>(), 2);
    image.set(random_image);

    // ----------------------------------------------------------------------------------------------

    Func G("G");
    Func S("S");

    Var x("x");
    Var y("y");

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

    S(x, y)  = image(clamp(x,0,image.width()-1),clamp(y,0,image.height()-1));
    S(rx,y) += select(rx>0, S(max(0,rx-1),y), 0.0f);
    S(x,ry) += select(ry>0, S(x,max(0,ry-1)), 0.0f);
    S(rz,y) += select(rz>0, S(max(0,rz-1),y), 0.0f);
    S(x,rw) += select(rw>0, S(x,max(0,rw-1)), 0.0f);

    G(x,y) = S(x,y);

    split(S,
            Internal::vec(  0,  1,  0,  1),
            Internal::vec(  x,  y,  x,  y),
            Internal::vec( xi, yi, xi, yi),
            Internal::vec( xo, yo, xo, yo),
            Internal::vec( rx, ry, rz, rw),
            Internal::vec(rxi,ryi,rzi,rwi));

    inline_function(G, "S");

    // ----------------------------------------------------------------------------------------------

    map<string,Func> functions = extract_func_calls(G);
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
