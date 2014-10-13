#include <iostream>
#include <Halide.h>

#include "../../lib/recfilter.h"

#define MAX_THREADS    192
#define UNROLL_FACTOR  6
#define TILES_PER_WARP 4

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

    filter.swap_variables ("SAT_Intra_Tail_y_1", "xi", "yi");
    filter.interleave_func("SAT_Intra_Tail_x_0", "SAT_Intra_Tail_y_1", "SAT_Intra_Tail", "xi", 1);

    filter.remove_pure_def("SAT_Intra_CTail_x_0_Sub");
    filter.remove_pure_def("SAT_Intra_CTail_y_1_Sub");

    cerr << filter << endl;

    // ----------------------------------------------------------------------------------------------

    {
        Var t("t");

        Var xi("xi"), yi("yi"), rxi("rxi"), rxf("rxf"), rxt("rxt");
        Var xo("xo"), yo("yo"), ryi("ryi"), ryf("ryf"), ryt("ryt");

        Var rxox("rxo.x$r"), rxoy("rxo.y$r");
        Var ryox("ryo.x$r"), ryoy("ryo.y$r");

        Func SAT             = filter.func("SAT");
        Func SAT_Final       = filter.func("SAT_Final");
        Func SAT_Intra       = filter.func("SAT_Intra");
        Func SAT_Tail        = filter.func("SAT_Intra_Tail");
        Func SAT_CTail_x     = filter.func("SAT_Intra_CTail_x_0");
        Func SAT_CTail_y     = filter.func("SAT_Intra_CTail_y_1");
        Func SAT_CTail_x_sub = filter.func("SAT_Intra_CTail_x_0_Sub");
        Func SAT_CTail_y_sub = filter.func("SAT_Intra_CTail_y_1_Sub");
        Func SAT_CTail_xy    = filter.func("SAT_Intra_CTail_x_0_y_1");
        Func SAT_CTail_xy_sub= filter.func("SAT_Intra_CTail_x_0_y_1_Sub");
        Func SAT_Deps_x      = filter.func("SAT_Intra_Deps_x_0");
        Func SAT_Deps_y      = filter.func("SAT_Intra_Deps_y_1");

        SAT_Intra.compute_at(SAT_Tail, Var::gpu_blocks());
        SAT_Intra.update(0).split(ryi,yi,t,UNROLL_FACTOR).reorder(t,rxi,yi,xo,yo).gpu_threads(rxi,yi).unroll(t);
        SAT_Intra.update(1).reorder(rxi,ryi,xo,yo).gpu_threads(ryi).unroll(rxi);
        SAT_Intra.update(2).reorder(rxt,ryi,xo,yo).gpu_threads(ryi).unroll(rxt);
        SAT_Intra.update(3).reorder(ryi,rxi,xo,yo).gpu_threads(rxi).unroll(ryi);
        SAT_Intra.update(4).reorder(ryt,rxi,xo,yo).gpu_threads(rxi).unroll(ryt);

        SAT_Tail.compute_root();
        SAT_Tail.reorder_storage(xi,yi,xo,yo);
        SAT_Tail.split(yi,yi,t,UNROLL_FACTOR).reorder(t,xi,yi,xo,yo).gpu(xo,yo,xi,yi).unroll(t);

        //

        SAT_CTail_x_sub.compute_root().reorder_storage(yi,yo,xi,xo);
        SAT_CTail_x_sub.update().reorder(rxox,rxoy,yi,yo).fuse(yi,yo,y).gpu_tile(y,MAX_THREADS);

        //

        SAT_CTail_xy_sub.compute_at(SAT_CTail_xy, Var::gpu_blocks());
        SAT_CTail_xy_sub.split(xo,xo,x,TILES_PER_WARP).reorder(yi,xi,x,xo,yo).gpu_threads(x,yi);
        SAT_CTail_xy_sub.update().split(xo,xo,x,TILES_PER_WARP).reorder(ryi,rxt,x,xo,yo).gpu_threads(x).unroll(ryi);

        SAT_CTail_xy.compute_root();
        SAT_CTail_xy.split(xo,xo,x,TILES_PER_WARP).reorder(yi,xi,x,xo,yo).gpu_threads(x,yi).gpu_blocks(xo,yo);

        //

        SAT_CTail_y_sub.compute_root().reorder_storage(xi,xo,yi,yo);
        SAT_CTail_y_sub.update().reorder(ryox,ryoy,xi,xo).fuse(xi,xo,x).gpu_tile(x,MAX_THREADS);

        //

        SAT_Deps_x.compute_at(SAT, Var::gpu_blocks()).reorder(xi,yi,xo,yo).gpu_threads(yi);
        SAT_Deps_y.compute_at(SAT, Var::gpu_blocks()).reorder(xi,yi,xo,yo).gpu_threads(xi);

        SAT_Final.compute_at(SAT, Var::gpu_blocks());
        SAT_Final.update(0).split(ryi,yi,t,UNROLL_FACTOR).reorder(t,rxi,yi,xo,yo).gpu_threads(rxi,yi).unroll(t);
        SAT_Final.update(1).reorder(rxt,ryi,xo,yo).gpu_threads(ryi).unroll(rxt);
        SAT_Final.update(2).reorder(rxf,ryi,xo,yo).gpu_threads(ryi).unroll(rxf);
        SAT_Final.update(3).reorder(ryt,rxi,xo,yo).gpu_threads(rxi).unroll(ryt);
        SAT_Final.update(4).reorder(ryf,rxi,xo,yo).gpu_threads(rxi).unroll(ryf);

        SAT.compute_root();
        SAT.split(x,xo,xi,tile).split(y,yo,yi,tile);
        SAT.split(yi,yi,t,UNROLL_FACTOR).reorder(t,xi,yi,xo,yo).gpu(xo,yo,xi,yi).unroll(t);

        SAT.bound(x,0,width).bound(y,0,height);
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
