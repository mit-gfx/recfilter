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
    Arguments args("gauss_anticausal_cascaded", argc, argv);

    bool nocheck = args.nocheck;
    int  width   = args.width;
    int  height  = args.width;
    int  tile    = args.block;

    float sigma = 16.0f;

    Image<float> random_image = generate_random_image<float>(width,height);

    ImageParam image(type_of<float>(), 2);
    image.set(random_image);

    // ----------------------------------------------------------------------------------------------

    float         B1 = gaussian_weights(sigma, 1).first;
    float         B2 = gaussian_weights(sigma, 2).first;
    vector<float> W1 = gaussian_weights(sigma, 1).second;
    vector<float> W2 = gaussian_weights(sigma, 2).second;

    Var x("x");
    Var y("y");

    RDom rx(0, image.width(), "rx");
    RDom ry(0, image.height(),"rw");

    RecFilter filter;
    filter.setArgs(x, y);
    filter.define(image(clamp(x,0,image.width()-1), clamp(y,0,image.height()-1)));
    filter.addScan(x, rx, B1, W1, RecFilter::CAUSAL);
    filter.addScan(x, rx, B1, W1, RecFilter::ANTICAUSAL);
    filter.addScan(y, ry, B1, W1, RecFilter::CAUSAL);
    filter.addScan(y, ry, B1, W1, RecFilter::ANTICAUSAL);
    filter.addScan(x, rx, B2, W2, RecFilter::CAUSAL);
    filter.addScan(x, rx, B2, W2, RecFilter::ANTICAUSAL);
    filter.addScan(y, ry, B2, W2, RecFilter::CAUSAL);
    filter.addScan(y, ry, B2, W2, RecFilter::ANTICAUSAL);

    // TODO: cascade the scans

    filter.split(x, y, tile);

    // ----------------------------------------------------------------------------------------------

    map<string,Func> functions = filter.funcs();
    map<string,Func>::iterator f    = functions.begin();
    map<string,Func>::iterator fend = functions.end();
    for (; f!=fend; f++) {
        cerr << f->second << endl;
        f->second.compute_root();
    }

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
        cerr << "Difference with true Gaussian \n" << CheckResult(ref,hl_out) << endl;
    }

    return 0;
}
