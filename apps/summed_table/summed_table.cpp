#include <iostream>
#include <Halide.h>

#include "../../lib/recfilter.h"

#define MAX_THREADS 192

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

    filter.swap_variables("SAT-Intra-Tail_y_1", "xi", "yi");
    filter.merge_func(
            "SAT-Intra-Tail_x_0",
            "SAT-Intra-Tail_y_1",
            "SAT-Intra-Tail");

    cerr << filter << endl;

    // ----------------------------------------------------------------------------------------------

    Target target = get_jit_target_from_environment();
    if (target.has_gpu_feature() || (target.features & Target::GPUDebug)) {
        Var t("t");

        Var xi("xi"), yi("yi"), rxi("rxi"), rxt("rxt");
        Var xo("xo"), yo("yo"), ryi("ryi"), ryt("ryt");

        Var rxox("rxo.x$r"), rxoy("rxo.y$r"), rxoz("rxo.z$r");
        Var ryox("ryo.x$r"), ryoy("ryo.y$r"), ryoz("ryo.z$r");

        Var bx = Var::gpu_blocks();

        Func SAT                = filter.func("SAT");
        Func SAT_GPU            = filter.func("SAT-Final");
        Func SAT_Final          = filter.func("SAT-Final-Sub");
        Func SAT_Intra          = filter.func("SAT-Intra");
        Func SAT_Intra_Tail     = filter.func("SAT-Intra-Tail");
        Func SAT_Intra_CTail_x  = filter.func("SAT-Intra-CTail_x_0");
        Func SAT_Intra_CTail_y  = filter.func("SAT-Intra-CTail_y_1");
        Func SAT_Intra_CTail_xy = filter.func("SAT-Intra-CTail_x_0-y-1");

        SAT_Intra.compute_at(SAT_Intra_Tail, bx);
        SAT_Intra.split(yi,yi,t,6).reorder(t,xi,yi,xo,yo).gpu_threads(xi,yi).unroll(t);
        SAT_Intra.update(0).reorder(rxi,ryi,xo,yo).gpu_threads(ryi).unroll(rxi);
        SAT_Intra.update(1).reorder(rxt,ryi,xo,yo).gpu_threads(ryi).unroll(rxt);
        SAT_Intra.update(2).reorder(ryi,rxi,xo,yo).gpu_threads(rxi).unroll(ryi);
        SAT_Intra.update(3).reorder(ryt,rxi,xo,yo).gpu_threads(rxi).unroll(ryt);

        SAT_Intra_Tail.compute_root();
        SAT_Intra_Tail.split(yi,yi,t,6).reorder(t,xi,yi,xo,yo).gpu(xo,yo,xi,yi).unroll(t);

        SAT_Intra_CTail_x.compute_root();
        SAT_Intra_CTail_x.reorder(xi,xo,yi,yo).fuse(yi,yo,y).gpu_tile(y,MAX_THREADS);
        SAT_Intra_CTail_x.update().reorder(rxox,rxoy,rxoz,yi,yo).fuse(yi,yo,y).gpu_tile(y,MAX_THREADS);

        SAT_Intra_CTail_xy.compute_at(SAT_Intra_CTail_y, bx);
        SAT_Intra_CTail_xy.split(yi,yi,t,1).reorder(t,xi,yi,xo,yo).gpu_threads(xi).unroll(t);
        SAT_Intra_CTail_xy.update(0).reorder(ryi,rxt,xo,yo).gpu_threads(rxt).unroll(ryi);

        SAT_Intra_CTail_y.compute_root();
        SAT_Intra_CTail_y.split(yi,yi,t,6).reorder(t,xi,yi,xo,yo).gpu(xo,yo,xi,yi).unroll(t);
        SAT_Intra_CTail_y.update().reorder(ryox,ryoy,ryoz,xi,xo).fuse(xi,xo,x).gpu_tile(x,MAX_THREADS);

        SAT_Final.compute_at(SAT_GPU, bx);
        SAT_Final.split(yi,yi,t,6).reorder(t,xi,yi,xo,yo).gpu_threads(xi,yi).unroll(t);
        SAT_Final.update(0).reorder(rxi,ryi,xo,yo).gpu_threads(ryi).unroll(rxi);
        SAT_Final.update(1).reorder(ryi,rxi,xo,yo).gpu_threads(rxi).unroll(ryi);

        SAT_GPU.compute_root();
        SAT_GPU.split(yi,yi,t,6).reorder(t,xi,yi,xo,yo).gpu(xo,yo,xi,yi).unroll(t);

        SAT.bound(x, 0, image.width()).bound(y, 0, image.height());
    }

    // ----------------------------------------------------------------------------------------------

    cerr << "\nJIT compilation ... " << endl;
    filter.func().compile_to_lowered_stmt("hl_stmt.html", HTML);
    filter.func().compile_jit();

    Buffer hl_out_buff(type_of<float>(), width,height);
    {
        Timer t("Running ... ");
        for (int i=0; i<iter; i++) {
            filter.func().realize(hl_out_buff);
            if (i!=iter-1) {
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

        cerr << CheckResult(ref,hl_out) << endl;
    }

    return 0;
}
