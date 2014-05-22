#include <iostream>
#include <cstdio>
#include <cmath>

#include <Halide.h>

#include "../../lib/split.h"

#define MAX_THREAD 192

using namespace Halide;

using std::vector;
using std::string;
using std::cerr;
using std::endl;

Image<int> reference_recursive_filter(int width, int height, int tile);

int main(int argc, char **argv) {
    Arguments args("test", argc, argv);

    bool  verbose = args.verbose;
    bool  nocheck = args.nocheck;
    int   width = args.width;
    int   height= args.height;
    int   tile  = args.block;


    Func I("Input");
    Func S("S");
    Func SI("SI");

    Var x("x"), y("y");
    Var xi("xi"), yi("yi");
    Var xo("xo"), yo("yo");

    RDom rxi(1, tile-1, "rxi");
    RDom ryi(1, tile-1, "ryi");

    I(x,y) = 1;

    SI(xo,xi,yo,yi)   = I(xo*tile+xi, yo*tile+yi);
    SI(xo,rxi,yo,yi) += SI(xo, rxi-1, yo, yi);
    SI(xo,xi,yo,ryi) += SI(xo, xi, yo, ryi-1);

    S(x,y) = SI(x/tile, x%tile, y/tile, y%tile);

    Target target = get_jit_target_from_environment();
    if (target.has_gpu_feature() || (target.features & Target::GPUDebug)) {
        Var t("t");

        SI.compute_at(S, Var("blockidx"));
        SI.reorder_storage(xi,yi,xo,yo);
#define BUG 1
#if BUG
        SI.split(yi,t,yi, MAX_THREAD/tile).reorder(t,xi,yi,xo,yo).gpu_threads(xi,yi);
#else
        SI.reorder(xi,yi,xo,yo).gpu_threads(yi);
#endif
        SI.update(0).reorder(rxi.x,yi,xo,yo).gpu_threads(yi);
        SI.update(1).reorder(ryi.x,xi,xo,yo).gpu_threads(xi);

        S.compute_root();
        S.split(x, xo,xi, tile).split(y, yo,yi, tile);
#if BUG
        S.split(yi,t,yi, MAX_THREAD/tile).reorder(t,xi,yi,xo,yo);
#else
        S.reorder(yi, xi, xo, yo);
#endif
        S.gpu_blocks(xo,yo).gpu_threads(xi);
        S.bound(x, 0, width).bound(y, 0, height);
    }

    Image<int> hl_out = S.realize(width,height);

    // ----------------------------------------------------------------------------------------------

    cerr << "\nChecking difference ... " << endl;
    Image<int> diff(width,height);
    Image<int> ref = reference_recursive_filter(width, height, tile);

    int diff_sum = 0;
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            diff(x,y) = ref(x,y) - hl_out(x,y);
            diff_sum += std::abs(diff(x,y));
        }
    }
    if (verbose) {
        cerr << "Reference" << endl << ref << endl;
        cerr << "Halide output" << endl << hl_out << endl;
        cerr << "Difference " << endl << diff << endl;
        cerr << "\nError = " << diff_sum << endl;
    } else {
        cerr << "\nError = " << diff_sum << endl;
        cerr << endl;
    }

    return 0;
}

Image<int> reference_recursive_filter(int width, int height, int tile) {
    Halide::Image<int> ref(width,height);

    for (int y=0; y<height/tile; y++) {
        for (int x=0; x<width/tile; x++) {
            for (int v=0; v<tile; v++) {
                for (int u=0; u<tile; u++) {
                    ref(x*tile+u, y*tile+v) = 1;
                }
            }
        }
    }

    for (int y=0; y<height/tile; y++) {          // x filtering
        for (int x=0; x<width/tile; x++) {
            for (int v=0; v<tile; v++) {
                for (int u=1; u<tile; u++) {
                    ref(x*tile+u, y*tile+v) += ref(x*tile+u-1, y*tile+v);
                }
            }
        }
    }

    for (int y=0; y<height/tile; y++) {          // y filtering
        for (int x=0; x<width/tile; x++) {
            for (int v=1; v<tile; v++) {
                for (int u=0; u<tile; u++) {
                    ref(x*tile+u, y*tile+v) += ref(x*tile+u, y*tile+v-1);
                }
            }
        }
    }

    return ref;
}

