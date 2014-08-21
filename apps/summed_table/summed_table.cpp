#include <iostream>
#include <Halide.h>

#include "../../lib/recfilter.h"

#define MAX_THREADS   192
#define UNROLL_FACTOR 6

using namespace Halide;

using std::map;
using std::vector;
using std::string;
using std::cerr;
using std::cout;
using std::endl;


int main(int argc, char **argv) {
    Arguments args(argc, argv);

    bool  nocheck = args.nocheck;
    int   width   = args.width;
    int   height  = args.width;
    int   tile    = args.block;
    int   iter    = args.iterations;

    Image<float> random_image = generate_random_image<float>(width,height);

    ImageParam image(type_of<float>(), 2);
    image.set(random_image);

    // ----------------------------------------------------------------------------------------------

    Var x("x");
    Var y("y");

    RDom rx(0, image.width(), "rx");
    RDom ry(0, image.height(),"ry");

    RecFilter filter("SAT");
    filter.setArgs(x, y);
    filter.define(image(clamp(x,0,image.width()-1),clamp(y,0,image.height()-1)));
    filter.addScan(x, rx);
    filter.addScan(y, ry);

    filter.split(tile);

    filter.swap_variables("SAT_Intra_Tail_y_1", "xi", "yi");
    filter.merge_func(
            "SAT_Intra_Tail_x_0",
            "SAT_Intra_Tail_y_1",
            "SAT_Intra_Tail");

    cerr << filter << endl;

    // ----------------------------------------------------------------------------------------------

    Target target = get_jit_target_from_environment();
    if (target.has_gpu_feature() || (target.features & Target::GPUDebug)) {
        Var t("t");

        Var xi("xi"), yi("yi"), rxi("rxi"), rxt("rxt");
        Var xo("xo"), yo("yo"), ryi("ryi"), ryt("ryt");

        Var rxox("rxo.x$r"), rxoy("rxo.y$r"), rxoz("rxo.z$r");
        Var ryox("ryo.x$r"), ryoy("ryo.y$r"), ryoz("ryo.z$r");

        Func SAT_Final       = filter.func("SAT_Final");
        Func SAT_Final_sub   = filter.func("SAT_Final_Sub");
        Func SAT_Intra       = filter.func("SAT_Intra");
        Func SAT_Tail        = filter.func("SAT_Intra_Tail");
        Func SAT_CTail_x     = filter.func("SAT_Intra_CTail_x_0");
        Func SAT_CTail_y     = filter.func("SAT_Intra_CTail_y_1");
        Func SAT_CTail_x_sub = filter.func("SAT_Intra_CTail_x_0_Sub");
        Func SAT_CTail_y_sub = filter.func("SAT_Intra_CTail_y_1_Sub");
        Func SAT_CTail_xy    = filter.func("SAT_Intra_CTail_x_0_y_1");
        Func SAT_Deps_x      = filter.func("SAT_Intra_Deps_x_0");
        Func SAT_Deps_y      = filter.func("SAT_Intra_Deps_y_1");

        // stage 1
        {
            SAT_Intra.compute_at(SAT_Tail, Var::gpu_blocks());
            SAT_Intra.split(yi,yi,t,UNROLL_FACTOR).reorder(t,xi,yi,xo,yo).gpu_threads(xi,yi).unroll(t);
            SAT_Intra.update(0).reorder(rxi,ryi,xo,yo).gpu_threads(ryi).unroll(rxi);
            SAT_Intra.update(1).reorder(rxt,ryi,xo,yo).gpu_threads(ryi).unroll(rxt);
            SAT_Intra.update(2).reorder(ryi,rxi,xo,yo).gpu_threads(rxi).unroll(ryi);
            SAT_Intra.update(3).reorder(ryt,rxi,xo,yo).gpu_threads(rxi).unroll(ryt);

            SAT_Tail.compute_root();
            SAT_Tail.reorder_storage(xi,yi,xo,yo);
            SAT_Tail.split(yi,yi,t,UNROLL_FACTOR).reorder(t,xi,yi,xo,yo).gpu(xo,yo,xi,yi).unroll(t);
        }

        // stage 2:
        {
            SAT_CTail_x_sub.compute_root();
            SAT_CTail_x_sub.reorder_storage(yi,yo,xi,xo);
            SAT_CTail_x_sub.reorder(xi,xo,yi,yo).fuse(yi,yo,y).gpu_tile(y,MAX_THREADS);
            SAT_CTail_x_sub.update().reorder(rxox,rxoy,rxoz,yi,yo).fuse(yi,yo,y).gpu_tile(y,MAX_THREADS);
        }

        // stage 3
        {
#if 0
            SAT_CTail_xy.compute_at(SAT_CTail_y_sub, Var::gpu_blocks());
            SAT_CTail_xy.reorder(xi,yi,xo,yo).split(xo,xo,x,MAX_THREADS/tile).gpu_threads(yi);
            SAT_CTail_xy.update().reorder(ryi,rxt,xo,yo).split(xo,xo,x,MAX_THREADS/tile).gpu_threads(x).unroll(ryi);

            SAT_CTail_y_sub.compute_root();
            SAT_CTail_y_sub.reorder_storage(xi,yi,xo,yo);
            SAT_CTail_y_sub.split(xo,xo,x,MAX_THREADS/tile).fuse(x,xi,xi).reorder(xi,yi,xo,yo).gpu_blocks(xo,yo).gpu_threads(xi,yi);

            SAT_CTail_y_sub.bound(xo,0,image.width()/tile ).bound(xi,0,tile);
            SAT_CTail_y_sub.bound(yo,0,image.height()/tile).bound(yi,0,1);
#else
            SAT_CTail_xy.compute_at(SAT_CTail_y_sub, Var::gpu_blocks());
            SAT_CTail_xy.reorder(xi,yi,xo,yo).gpu_threads(yi).unroll(xi);
            SAT_CTail_xy.update().reorder(ryi,rxt,xo,yo).unroll(rxt).unroll(ryi);

            SAT_CTail_y_sub.compute_root();
            SAT_CTail_y_sub.reorder_storage(xi,xo,yi,yo);
            SAT_CTail_y_sub.reorder(xi,yi,xo,yo).gpu_blocks(xo,yo).gpu_threads(xi);
#endif
            SAT_CTail_y_sub.update().reorder(ryox,ryoy,ryoz,xi,xo).fuse(xi,xo,x).gpu_tile(x,MAX_THREADS);
        }

        // stage 4
        {
            SAT_Deps_x.compute_at(SAT_Final, Var::gpu_blocks());
            SAT_Deps_y.compute_at(SAT_Final, Var::gpu_blocks());
            SAT_Deps_x.split(yi,yi,t,UNROLL_FACTOR).reorder(t,xi,yi,xo,yo).gpu_threads(xi,yi).unroll(t);
            SAT_Deps_y.split(yi,yi,t,UNROLL_FACTOR).reorder(t,xi,yi,xo,yo).gpu_threads(xi,yi).unroll(t);

            SAT_Final_sub.compute_at(SAT_Final, Var::gpu_blocks());
            SAT_Final_sub.split(yi,yi,t,UNROLL_FACTOR).reorder(t,xi,yi,xo,yo).gpu_threads(xi,yi).unroll(t);
            SAT_Final_sub.update(0).reorder(rxi,ryi,xo,yo).gpu_threads(ryi).unroll(rxi);
            SAT_Final_sub.update(1).reorder(ryi,rxi,xo,yo).gpu_threads(rxi).unroll(ryi);

            SAT_Final.compute_root();
            SAT_Final.reorder_storage(xi,xo,yi,yo);
            SAT_Final.split(yi,yi,t,UNROLL_FACTOR).reorder(t,xi,yi,xo,yo).gpu(xo,yo,xi,yi).unroll(t);
        }
    }

    // ----------------------------------------------------------------------------------------------

    cerr << "\nJIT compilation ... " << endl;
    filter.compile_jit("hl_stmt.html");

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
            for (int x=1; x<width; x++) {
                ref(x,y) += ref(x-1,y);
            }
        }
        for (int y=1; y<height; y++) {
            for (int x=0; x<width; x++) {
                ref(x,y) += ref(x,y-1);
            }
        }

        cout << CheckResult(ref,hl_out) << endl;
    }

    return 0;
}
