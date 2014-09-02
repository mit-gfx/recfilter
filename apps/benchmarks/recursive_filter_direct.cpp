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

    Target target = get_jit_target_from_environment();
    if (target.has_gpu_feature() || (target.features & Target::GPUDebug)) {
        Var t("t");

        Var xi("xi"), yi("yi"), rxi("rxi"), rxt("rxt");
        Var xo("xo"), yo("yo"), ryi("ryi"), ryt("ryt");

        Var rxox("rxo.x$r"), rxoy("rxo.y$r"), rxoz("rxo.z$r");
        Var ryox("ryo.x$r"), ryoy("ryo.y$r"), ryoz("ryo.z$r");

        Func S_Final       = filter.func("S_Final");
        Func S_Final_sub   = filter.func("S_Final_Sub");
        Func S_Intra       = filter.func("S_Intra");
        Func S_Tail        = filter.func("S_Intra_Tail");
        Func S_CTail_x     = filter.func("S_Intra_CTail_x_0");
        Func S_CTail_y     = filter.func("S_Intra_CTail_y_1");
        Func S_CTail_x_sub = filter.func("S_Intra_CTail_x_0_Sub");
        Func S_CTail_y_sub = filter.func("S_Intra_CTail_y_1_Sub");
        Func S_CTail_xy    = filter.func("S_Intra_CTail_x_0_y_1");
        Func S_Deps_x      = filter.func("S_Intra_Deps_x_0");
        Func S_Deps_y      = filter.func("S_Intra_Deps_y_1");

        // stage 1
        {
            S_Intra.compute_at(S_Tail, Var::gpu_blocks());
            S_Intra.split(yi,yi,t,UNROLL_FACTOR).reorder(t,xi,yi,xo,yo).gpu_threads(xi,yi).unroll(t);
            S_Intra.update(0).reorder(rxi,ryi,xo,yo).gpu_threads(ryi).unroll(rxi);
            S_Intra.update(1).reorder(rxt,ryi,xo,yo).gpu_threads(ryi).unroll(rxt);
            S_Intra.update(2).reorder(ryi,rxi,xo,yo).gpu_threads(rxi).unroll(ryi);
            S_Intra.update(3).reorder(ryt,rxi,xo,yo).gpu_threads(rxi).unroll(ryt);

            S_Tail.compute_root();
            S_Tail.reorder_storage(xi,yi,xo,yo);
            S_Tail.split(yi,yi,t,UNROLL_FACTOR).reorder(t,xi,yi,xo,yo).gpu(xo,yo,xi,yi).unroll(t);
        }

        // stage 2:
        {
            S_CTail_x_sub.compute_root();
            S_CTail_x_sub.reorder_storage(yi,yo,xi,xo);
            S_CTail_x_sub.reorder(xi,xo,yi,yo).fuse(yi,yo,y).gpu_tile(y,MAX_THREADS);
            S_CTail_x_sub.update().reorder(rxox,rxoy,rxoz,yi,yo).fuse(yi,yo,y).gpu_tile(y,MAX_THREADS);
        }

        // stage 3
        {
            S_CTail_xy.compute_at(S_CTail_y_sub, Var::gpu_blocks());
            S_CTail_xy.reorder(xi,yi,xo,yo).gpu_threads(yi).unroll(xi);
            S_CTail_xy.update().reorder(ryi,rxt,xo,yo).unroll(rxt).unroll(ryi);

            S_CTail_y_sub.compute_root();
            S_CTail_y_sub.reorder_storage(xi,xo,yi,yo);
            S_CTail_y_sub.reorder(xi,yi,xo,yo).gpu_blocks(xo,yo).gpu_threads(xi);

            S_CTail_y_sub.update().reorder(ryox,ryoy,ryoz,xi,xo).fuse(xi,xo,x).gpu_tile(x,MAX_THREADS);
        }

        // stage 4
        {
            S_Deps_x.compute_at(S_Final, Var::gpu_blocks());
            S_Deps_y.compute_at(S_Final, Var::gpu_blocks());
            S_Deps_x.split(yi,yi,t,UNROLL_FACTOR).reorder(t,xi,yi,xo,yo).gpu_threads(xi,yi).unroll(t);
            S_Deps_y.split(yi,yi,t,UNROLL_FACTOR).reorder(t,xi,yi,xo,yo).gpu_threads(xi,yi).unroll(t);

            S_Final_sub.compute_at(S_Final, Var::gpu_blocks());
            S_Final_sub.split(yi,yi,t,UNROLL_FACTOR).reorder(t,xi,yi,xo,yo).gpu_threads(xi,yi).unroll(t);
            S_Final_sub.update(0).reorder(rxi,ryi,xo,yo).gpu_threads(ryi).unroll(rxi);
            S_Final_sub.update(1).reorder(ryi,rxi,xo,yo).gpu_threads(rxi).unroll(ryi);

            S_Final.compute_root();
            S_Final.reorder_storage(xi,xo,yi,yo);
            S_Final.split(yi,yi,t,UNROLL_FACTOR).reorder(t,xi,yi,xo,yo).gpu(xo,yo,xi,yi).unroll(t);
        }
    }

    // ----------------------------------------------------------------------------------------------

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
















































































