// Code to divide the image into 2D tile and compute a summed area table
// within each tile in parallel

#include <iostream>
#include <Halide.h>

#define MAX_THREAD 192

using namespace Halide;

void check_correctness(Image<int> hl_out, int tile);

int main(int argc, char **argv) {
    int width = 2048;
    int height= 2048;
    int tile  = 64;

    Func I("Input");
    Func S("S");
    Func SI("SI");

    Var x("x"), y("y");
    Var xi("xi"), yi("yi");
    Var xo("xo"), yo("yo");

    RDom rxi(1, tile-1, "rxi");     // prefix sums within tiles
    RDom ryi(1, tile-1, "ryi");

    I(x,y) = 1;

    SI(xo,xi,yo,yi)   = I(xo*tile+xi, yo*tile+yi);  // divide image into tiles
    SI(xo,rxi,yo,yi) += SI(xo, rxi-1, yo, yi);      // x prefix sum within each tile
    SI(xo,xi,yo,ryi) += SI(xo, xi, yo, ryi-1);      // y prefix sum within each tile

    S(x,y) = SI(x/tile, x%tile, y/tile, y%tile);    // final image

    Target target = get_jit_target_from_environment();
    if (target.has_gpu_feature() || (target.features & Target::GPUDebug)) {
        Var t("t");

        SI.compute_at(S, Var("blockidx"));
        SI.reorder_storage(xi,yi,xo,yo);
        SI.reorder(xi,yi,xo,yo).gpu_threads(yi);
        SI.update(0).reorder(rxi.x,yi,xo,yo).gpu_threads(yi);
        SI.update(1).reorder(ryi.x,xi,xo,yo).gpu_threads(xi);

        S.compute_root();
        S.split(x, xo,xi, tile).split(y, yo,yi, tile);
        S.reorder(yi, xi, xo, yo);
        S.gpu_blocks(xo,yo).gpu_threads(xi);
        S.bound(x, 0, width).bound(y, 0, height);
    }

    S.trace_realizations();
    //SI.trace_realizations();

    Image<int> hl_out = S.realize(width,height);

    check_correctness(hl_out, tile);

    return 0;
}

void check_correctness(Image<int> hl_out, int tile) {
    int width = hl_out.width();
    int height = hl_out.height();

    Image<int> diff(width,height);
    Image<int> ref(width,height);

    for (int y=0; y<height/tile; y++) {
        for (int x=0; x<width/tile; x++) {
            for (int v=0; v<tile; v++) {
                for (int u=0; u<tile; u++) {
                    ref(x*tile+u, y*tile+v) = 1; // input image is all 1
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

    int diff_sum = 0;
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            diff(x,y) = ref(x,y) - hl_out(x,y);
            diff_sum += std::abs(diff(x,y));
        }
    }

    std::cerr << "\nError = " << diff_sum << std::endl;
}
