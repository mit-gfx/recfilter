#include <iostream>
#include <Halide.h>

#include "../../lib/recfilter.h"

#define MAX_THREAD 192

using namespace Halide;

using std::map;
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

    Var x("x");
    Var y("y");

    RDom rx(0, image.width(), "rx");
    RDom ry(0, image.height(),"ry");

    RecFilter filter;
    filter.setArgs(x, y);
    filter.define(image(clamp(x,0,image.width()-1),clamp(y,0,image.height()-1)));
    filter.addScan(x, rx);
    filter.addScan(y, ry);

    filter.split(x, y, tile);

    cerr << filter << endl;

    // ----------------------------------------------------------------------------------------------

    Target target = get_jit_target_from_environment();
    if (target.has_gpu_feature() || (target.features & Target::GPUDebug)) {
    }

    // ----------------------------------------------------------------------------------------------

    cerr << "\nJIT compilation ... " << endl;
    filter.func().compile_jit();

    Buffer hl_out_buff(type_of<float>(), width,height);
    {
        Timer t("Running ... ");
        for (int k=0; k<iterations; k++) {
            filter.func().realize(hl_out_buff);
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
                ref(x,y) = random_image(x,y);
            }
        }
        for (int y=0; y<height; y++) {
            for (int x=1; x<width; x++) {
                ref(x,y) += ref(x-1,y);
            }
        }
        for (int y=1; y<height; y++) {
            for (int x=0; x<width; x++) {
                ref(x,y) += ref(x,y-1);
            }
        }

        cerr << CheckResultVerbose(ref,hl_out) << endl;
    }

    return 0;
}
