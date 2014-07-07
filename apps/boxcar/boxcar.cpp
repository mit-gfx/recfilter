#include <iostream>
#include <Halide.h>

#include "../../lib/split.h"

#define MAX_THREAD        192
#define BOX_FILTER_FACTOR 16

using namespace Halide;

using std::vector;
using std::string;
using std::cerr;
using std::endl;


int main(int argc, char **argv) {
    Arguments args("boxcar", argc, argv);

    bool  verbose = args.verbose;
    bool  nocheck = args.nocheck;
    int   width   = args.width;
    int   height  = args.height;
    int   tile_width = args.block;
    int   iterations = args.iterations;

    Image<float> random_image = generate_random_image<float>(width,height);

    ImageParam image(type_of<float>(), 2);
    image.set(random_image);

    // ----------------------------------------------------------------------------------------------

    int box_x = width/BOX_FILTER_FACTOR;        // radius of box filter
    int box_y = height/BOX_FILTER_FACTOR;

    Func S("S");
    Func B("Boxcar");

    Var x("x");
    Var y("y");

    RDom rx(0, image.width(), "rx");
    RDom ry(0, image.height(),"ry");

    S(x, y) = image(clamp(x,0,image.width()-1),clamp(y,0,image.height()-1));
    S(rx,y) = S(rx,y) + select(rx>0, S(max(0,rx-1),y), 0);
    S(x,ry) = S(x,ry) + select(ry>0, S(x,max(0,ry-1)), 0);

    B(x,y) = S(min(x+box_x,image.width()-1), min(y+box_y,image.height()-1))
           + select(x-box_x-1<0 || y-box_y-1<0, 0, S(max(x-box_x-1,0), max(y-box_y-1,0)))
           - select(y-box_y-1<0, 0, S(min(x+box_x,image.width()-1), max(y-box_y-1,0)))
           - select(x-box_x-1<0, 0, S(max(x-box_x-1,0), min(y+box_y,image.height()-1)));

    // ----------------------------------------------------------------------------------------------

    Var xi("xi"), yi("yi");
    Var xo("xo"), yo("yo");

    RDom rxi(0, tile_width, "rxi");
    RDom ryi(0, tile_width, "ryi");

    split(S,Internal::vec(0,1),
            Internal::vec(x,y),
            Internal::vec(xi,yi),
            Internal::vec(xo,yo),
            Internal::vec(rx,ry),
            Internal::vec(rxi,ryi), true);

    // ----------------------------------------------------------------------------------------------

    inline_function(B, "S");

    // ----------------------------------------------------------------------------------------------

    map<string,Func> functions = extract_func_calls(S);
    map<string,Func>::iterator f    = functions.begin();
    map<string,Func>::iterator fend = functions.end();
    for (; f!=fend; f++) {
        cerr << f->second << endl;
        f->second.compute_root();
    }

    Func S_intra0= functions["S--Intra_y-Intra_x"];
    Func S_tails = functions["S--Tail"];
    Func S_intra = functions["S--Intra_y-Intra_x-Recomp"];
    Func S_ctaily= functions["S--CTail_y"];
    Func S_ctailx= functions["S--Intra_y-CTail_x"];

    Target target = get_jit_target_from_environment();
    if (target.has_gpu_feature() || (target.features & Target::GPUDebug)) {
    }

    // ----------------------------------------------------------------------------------------------

    cerr << "\nJIT compilation ... " << endl;
    B.compile_jit();

    Buffer hl_out_buff(type_of<float>(), width,height);
    {
        Timer t("Running ... ");
        for (int k=0; k<iterations; k++) {
            B.realize(hl_out_buff);
            if (k < iterations-1) {
                hl_out_buff.free_dev_buffer();
            }
        }
    }
    hl_out_buff.copy_to_host();
    hl_out_buff.free_dev_buffer();

    // ----------------------------------------------------------------------------------------------

    if (!nocheck) {
        cerr << "\nChecking difference ... " << endl;
        Image<float> hl_out(hl_out_buff);
        Image<float> ref(width,height);

        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {
                ref(x,y) = 0;
                for (int v=y-box_y; v<=y+box_y; v++) {
                    for (int u=x-box_x; u<=x+box_x; u++) {
                        if (v>=0 && u>=0 && v<height && u<width)
                            ref(x,y) += random_image(u,v);
                    }
                }
            }
        }
        cerr << CheckResult(ref,hl_out) << endl;
    }

    return 0;
}
