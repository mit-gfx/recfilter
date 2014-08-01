#include <iostream>
#include <cstdio>
#include <cmath>

#include <Halide.h>

#include "../../lib/gaussian_weights.h"
#include "../../lib/recfilter.h"

#define MAX_THREAD 192

using namespace Halide;

using std::vector;
using std::string;
using std::cerr;
using std::endl;

int main(int argc, char **argv) {
    Arguments args(argc, argv);

    bool nocheck = args.nocheck;
    int  width   = args.width;
    int  height  = args.width;
    int  tile    = args.block;

    float sigma = 16.0f;

    Image<float> random_image = generate_random_image<float>(width,height);

    ImageParam image(type_of<float>(), 2);
    image.set(random_image);

    // ----------------------------------------------------------------------------------------------

    float         B3 = gaussian_weights(sigma, 3).first;
    vector<float> W3 = gaussian_weights(sigma, 3).second;

    Var x("x");
    Var y("y");

    RDom rx(0, image.width(), "rx");
    RDom ry(0, image.height(),"ry");

    RecFilter filter("Gauss");
    filter.setArgs(x, y);
    filter.define(image(clamp(x,0,image.width()-1), clamp(y,0,image.height()-1)));
    filter.addScan(x, rx, B3, W3, RecFilter::CAUSAL    , RecFilter::CLAMP_TO_SELF);
    filter.addScan(x, rx, B3, W3, RecFilter::ANTICAUSAL, RecFilter::CLAMP_TO_SELF);
    filter.addScan(y, ry, B3, W3, RecFilter::CAUSAL    , RecFilter::CLAMP_TO_SELF);
    filter.addScan(y, ry, B3, W3, RecFilter::ANTICAUSAL, RecFilter::CLAMP_TO_SELF);

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
        filter.func().realize(hl_out_buff);
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
