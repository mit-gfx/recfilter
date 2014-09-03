#include <iostream>
#include <cstdio>
#include <cmath>

#include <Halide.h>

#include "../../lib/recfilter.h"

#define MAX_THREADS   192
#define UNROLL_FACTOR 6

using namespace Halide;

using std::vector;
using std::string;
using std::cerr;
using std::cout;
using std::endl;

int main(int argc, char **argv) {
    Arguments args(argc, argv);

    bool nocheck = args.nocheck;
    int  width   = args.width;
    int  height  = args.width;
    int  tile    = args.block;
    int  iter    = args.iterations;

    Image<float> random_image = generate_random_image<float>(width,height);

    ImageParam image(type_of<float>(), 2);
    image.set(random_image);

    // ----------------------------------------------------------------------------------------------

    float b0 = 0.425294f;
    vector<float> W2;
    W2.push_back(0.885641f);
    W2.push_back(-0.310935f);

    Var x("x");
    Var y("y");

    RDom rx(0, image.width(), "rx");
    RDom ry(0, image.height(),"ry");

    RecFilter filter("S");
    filter.setArgs(x, y);
    filter.define(image(clamp(x,0,image.width()-1), clamp(y,0,image.height()-1)));
    filter.addScan(x, rx, b0, W2, RecFilter::CAUSAL    , RecFilter::CLAMP_TO_SELF);
    filter.addScan(x, rx, b0, W2, RecFilter::ANTICAUSAL, RecFilter::CLAMP_TO_SELF);
    filter.addScan(y, ry, b0, W2, RecFilter::CAUSAL    , RecFilter::CLAMP_TO_SELF);
    filter.addScan(y, ry, b0, W2, RecFilter::ANTICAUSAL, RecFilter::CLAMP_TO_SELF);

    filter.split(tile);

    filter.swap_variables("S_Intra_Tail_y_2", "xi", "yi");
    filter.swap_variables("S_Intra_Tail_y_3", "xi", "yi");
    filter.merge_func("S_Intra_Tail_x_0", "S_Intra_Tail_x_1",
                 "S_Intra_Tail_y_2", "S_Intra_Tail_y_3",
                 "S_Intra_Tail");

    cerr << filter << endl;

    // ----------------------------------------------------------------------------------------------

    {
    }

    // ----------------------------------------------------------------------------------------------

    Target target = get_jit_target_from_environment();

    cerr << "\nJIT compilation ... " << endl;
    filter.compile_jit(target, "hl_stmt.html");

    cerr << "\nRunning ... " << endl;
    Buffer out(type_of<float>(), width, height);
    filter.realize(out, iter);

    // ----------------------------------------------------------------------------------------------

    if (!nocheck) {
        cerr << "\nChecking difference ... " << endl;
        Image<float> hl_out(out);
        Image<float> ref(width,height);

        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {
                ref(x,y) = random_image(x,y);
            }
        }
        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {
                ref(x,y) = b0*ref(x,y)
                    + W2[0]*ref(std::max(x-1,0),y)
                    + W2[1]*ref(std::max(x-2,0),y);
            }
        }
        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {
                ref(x,y) = b0*ref(x,y)
                    + W2[0]*ref(x,std::max(y-1,0))
                    + W2[1]*ref(x,std::max(y-2,0));
            }
        }
        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {
                ref(width-1-x,y) = b0*ref(width-1-x,y)
                    + W2[0]*ref(width-1-std::max(x-1,0),y)
                    + W2[1]*ref(width-1-std::max(x-2,0),y);
            }
        }
        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {
                ref(x,height-1-y) = b0*ref(x,height-1-y)
                    + W2[0]*ref(x,height-1-std::max(y-1,0))
                    + W2[1]*ref(x,height-1-std::max(y-2,0));
            }
        }

        cout << CheckResult(ref,hl_out) << endl;
    }

    return 0;
}
















































































