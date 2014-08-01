#include <iostream>
#include <Halide.h>

#include "../../lib/recfilter.h"

#define MAX_THREAD        192
#define BOX_FILTER_FACTOR 16

using namespace Halide;

using std::vector;
using std::string;
using std::cerr;
using std::endl;


int main(int argc, char **argv) {
    Arguments args(argc, argv);

    bool  nocheck = args.nocheck;
    int   width   = args.width;
    int   height  = args.width;
    int   tile    = args.block;
    int   iterations = args.iterations;

    Image<float> random_image = generate_random_image<float>(width,height);

    ImageParam image(type_of<float>(), 2);
    image.set(random_image);

    // ----------------------------------------------------------------------------------------------

    int box = BOX_FILTER_FACTOR;        // radius of box filter

    Var x("x");
    Var y("y");

    RDom rx(0, image.width(), "rx");
    RDom ry(0, image.height(),"ry");

    RecFilter filter("S");
    filter.setArgs(x, y);
    filter.define(image(clamp(x,0,image.width()-1),clamp(y,0,image.height()-1)));
    filter.addScan(x, rx);
    filter.addScan(y, ry);

    Func S = filter.func();

    Func B("Boxcar");
    B(x,y) = S(min(x+box,image.width()-1), min(y+box,image.height()-1))
           + select(x-box-1<0 || y-box-1<0, 0, S(max(x-box-1,0), max(y-box-1,0)))
           - select(y-box-1<0, 0, S(min(x+box,image.width()-1), max(y-box-1,0)))
           - select(x-box-1<0, 0, S(max(x-box-1,0), min(y+box,image.height()-1)));

    filter.split(x, y, tile);

    cerr << filter << endl;

    // ----------------------------------------------------------------------------------------------

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
                for (int v=y-box; v<=y+box; v++) {
                    for (int u=x-box; u<=x+box; u++) {
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
