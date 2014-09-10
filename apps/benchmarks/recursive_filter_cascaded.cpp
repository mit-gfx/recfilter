#include <iostream>
#include <cstdio>
#include <cmath>

#include <Halide.h>

#include "../../lib/recfilter.h"


using namespace Halide;

using std::vector;
using std::string;
using std::cout;
using std::cerr;
using std::endl;

int main(int argc, char **argv) {
    Arguments args(argc, argv);

    bool nocheck = args.nocheck;
    int  width   = args.width;
    int  height  = args.width;
    int  tile    = args.block;
    int  iter    = args.iterations;
    int  threads = args.threads;

    Image<float> random_image = generate_random_image<float>(width,height);

    ImageParam image(type_of<float>(), 2);
    image.set(random_image);

    // ----------------------------------------------------------------------------------------------

    float b0 = 0.425294f;
    vector<float> W2;
    W2.push_back(0.885641f);
//    W2.push_back(-0.310935f);
//    W2.push_back(-0.000005f);

    Var x("x");
    Var y("y");

    RDom rx(0, image.width(), "rx");
    RDom ry(0, image.height(),"ry");

    RecFilter filterx;
    RecFilter filtery;
    {
        RecFilter filter("S");
        filter.setArgs(x, y);
        filter.define(image(clamp(x,0,image.width()-1), clamp(y,0,image.height()-1)));
        filter.addScan(x, rx, b0, W2, RecFilter::CAUSAL    , RecFilter::CLAMP_TO_SELF);
        filter.addScan(x, rx, b0, W2, RecFilter::ANTICAUSAL, RecFilter::CLAMP_TO_SELF);
        filter.addScan(y, ry, b0, W2, RecFilter::CAUSAL    , RecFilter::CLAMP_TO_SELF);
        filter.addScan(y, ry, b0, W2, RecFilter::ANTICAUSAL, RecFilter::CLAMP_TO_SELF);

        // cascade the scans
        vector<RecFilter> cascaded_filters = filter.cascade(
                Internal::vec(0,1), Internal::vec(2,3));

        filterx = cascaded_filters[0];
        filtery = cascaded_filters[1];
    }

    filterx.split(tile);
    filtery.split(tile);

    filterx.merge_func("S_0_Intra_Tail_x_0", "S_0_Intra_Tail_x_1", "S_0_Intra_Tail");
    filtery.merge_func("S_1_Intra_Tail_y_0", "S_1_Intra_Tail_y_1", "S_1_Intra_Tail");
//    filterx.merge_func("S_0_Intra_CTail_x_0","S_0_Intra_CTail_x_1","S_0_Intra_CTail_x");
//    filtery.merge_func("S_1_Intra_CTail_y_0","S_1_Intra_CTail_y_1","S_1_Intra_CTail_y");

    cerr << filterx << endl;

    // ----------------------------------------------------------------------------------------------

    {
        Var t("t");

        Var xi("xi"), yi("yi"), rxi("rxi"), rxt("rxt");
        Var xo("xo"), yo("yo"), ryi("ryi"), ryt("ryt");

        Var rxox("rxo.x$r"), rxoy("rxo.y$r"), rxoz("rxo.z$r");
        Var ryox("ryo.x$r"), ryoy("ryo.y$r"), ryoz("ryo.z$r");

        Func Sx             = filterx.func("S_0");
        Func Sx_Final       = filterx.func("S_0_Final");
        Func Sx_Intra       = filterx.func("S_0_Intra");
        Func Sx_Tail        = filterx.func("S_0_Intra_Tail");
        Func Sx_CTail_0     = filterx.func("S_0_Intra_CTail_x_0_Sub");
        Func Sx_CTail_1     = filterx.func("S_0_Intra_CTail_x_1_Sub");
//        Func Sx_CTail       = filterx.func("S_0_Intra_CTail_x");
        Func Sx_Deps_0      = filterx.func("S_0_Intra_Deps_x_0");
        Func Sx_Deps_1      = filterx.func("S_0_Intra_Deps_x_1");

        Func Sy             = filtery.func("S_1");
        Func Sy_Final       = filtery.func("S_1_Final");
        Func Sy_Intra       = filtery.func("S_1_Intra");
        Func Sy_Tail        = filtery.func("S_1_Intra_Tail");
        Func Sy_CTail_0     = filtery.func("S_1_Intra_CTail_y_0_Sub");
        Func Sy_CTail_1     = filtery.func("S_1_Intra_CTail_y_1_Sub");
//        Func Sy_CTail       = filtery.func("S_1_Intra_CTail_y");
        Func Sy_Deps_0      = filtery.func("S_1_Intra_Deps_y_0");
        Func Sy_Deps_1      = filtery.func("S_1_Intra_Deps_y_1");

        int t1 = 256;
        int t2 = 256;

        // stage 1
        {
            Sx_Intra.compute_at(Sx_Tail, Var::gpu_blocks());
            Sx_Intra.split(y,yo,yi,t1).reorder(xi,yi,xo,yo).gpu_threads(yi).unroll(xi);
            Sx_Intra.update(0).split(y,yo,yi,t1).reorder(rxi,yi,xo,yo).gpu_threads(yi).unroll(rxi);
            Sx_Intra.update(1).split(y,yo,yi,t1).reorder(rxt,yi,xo,yo).gpu_threads(yi).unroll(rxt);
            Sx_Intra.update(2).split(y,yo,yi,t1).reorder(rxi,yi,xo,yo).gpu_threads(yi).unroll(rxi);
            Sx_Intra.update(3).split(y,yo,yi,t1).reorder(rxt,yi,xo,yo).gpu_threads(yi).unroll(rxt);

            Sx_Tail.compute_root();
            Sx_Tail.reorder_storage(y,xi,xo);
            Sx_Tail.split(y,yo,yi,t1).reorder(xi,yi,xo,yo).gpu_blocks(xo,yo).gpu_threads(yi).unroll(xi);

//            Sx_CTail_0.compute_at(Sx_CTail, Var::gpu_blocks());
            Sx_CTail_0.compute_root();
            Sx_CTail_0.reorder(xi,xo,y).unroll(xi).split(y,yo,yi,t2).gpu_threads(yi).gpu_blocks(yo);
            Sx_CTail_0.update().reorder(rxox,rxoy,rxoz,y).unroll(rxox).unroll(rxoy).split(y,yo,yi,t2).gpu_threads(yi).gpu_blocks(yo);

//            Sx_CTail_1.compute_at(Sx_CTail, Var::gpu_blocks());
            Sx_CTail_1.compute_root();
            Sx_CTail_1.reorder(xi,xo,y).unroll(xi).split(y,yo,yi,t2).gpu_threads(yi).gpu_blocks(yo);
            Sx_CTail_1.update().reorder(rxox,rxoy,rxoz,y).unroll(rxox).unroll(rxoy).split(y,yo,yi,t2).gpu_threads(yi).gpu_blocks(yo);

//            Sx_CTail.compute_root();
//            Sx_CTail.reorder_storage(y,xi,xo);
//            Sx_CTail.reorder(xi,xo,y).unroll(xi).split(y,yo,yi,t2).gpu_threads(yi).gpu_blocks(yo);

            Sx_Final.compute_at(Sx, Var::gpu_blocks());
            Sx_Final.split(y,yo,yi,t1).reorder(xi,yi,xo,yo).gpu_threads(yi).unroll(xi);
            Sx_Final.update(0).split(y,yo,yi,t1).reorder(rxt,yi,xo,yo).gpu_threads(yi).unroll(rxt);
            Sx_Final.update(1).split(y,yo,yi,t1).reorder(rxi,yi,xo,yo).gpu_threads(yi).unroll(rxi);
            Sx_Final.update(2).split(y,yo,yi,t1).reorder(rxt,yi,xo,yo).gpu_threads(yi).unroll(rxt);
            Sx_Final.update(3).split(y,yo,yi,t1).reorder(rxi,yi,xo,yo).gpu_threads(yi).unroll(rxi);

            Sx.compute_root();
            Sx.split(x,xo,xi,tile); //.reorder_storage(y,xi,xo);
            Sx.split(y,yo,yi,t1).reorder(xi,yi,xo,yo).gpu_blocks(xo,yo).gpu_threads(yi).unroll(xi);
        }

        // y filtering
        {
            Sy_Intra.compute_at(Sy_Tail, Var::gpu_blocks());
            Sy_Intra.split(x,xo,xi,t1).reorder(yi,xi,yo,xo).gpu_threads(xi).unroll(yi);
            Sy_Intra.update(0).split(x,xo,xi,t1).reorder(ryi,xi,yo,xo).gpu_threads(xi).unroll(ryi);
            Sy_Intra.update(1).split(x,xo,xi,t1).reorder(ryt,xi,yo,xo).gpu_threads(xi).unroll(ryt);
            Sy_Intra.update(2).split(x,xo,xi,t1).reorder(ryi,xi,yo,xo).gpu_threads(xi).unroll(ryi);
            Sy_Intra.update(3).split(x,xo,xi,t1).reorder(ryt,xi,yo,xo).gpu_threads(xi).unroll(ryt);

            Sy_Tail.compute_root();
            Sy_Tail.reorder_storage(x,yi,yo);
            Sy_Tail.split(x,xo,xi,t1).reorder(yi,xi,yo,xo).gpu_blocks(yo,xo).gpu_threads(xi).unroll(yi);

            Sy_CTail_0.compute_root();
            Sy_CTail_0.reorder(yi,yo,x).unroll(yi).split(x,xo,xi,t1).gpu_threads(xi).gpu_blocks(xo);
            Sy_CTail_0.update().reorder(ryox,ryoy,ryoz,x).unroll(ryox).unroll(ryoy).split(x,xo,xi,t1).gpu_threads(xi).gpu_blocks(xo);

            Sy_CTail_1.compute_root();
            Sy_CTail_1.reorder(yi,yo,x).unroll(yi).split(x,xo,xi,t1).gpu_threads(xi).gpu_blocks(xo);
            Sy_CTail_1.update().reorder(ryox,ryoy,ryoz,x).unroll(ryox).unroll(ryoy).split(x,xo,xi,t1).gpu_threads(xi).gpu_blocks(xo);

//            Sy_CTail.compute_root();
//            Sy_CTail.reorder_storage(x,yi,yo);
//            Sy_CTail.reorder(yi,yo,x).unroll(yi).split(x,xo,xi,t1).gpu_threads(xi).gpu_blocks(xo);

            Sy_Final.compute_at(Sy, Var::gpu_blocks());
            Sy_Final.split(x,xo,xi,t1).reorder(yi,xi,yo,xo).gpu_threads(xi).unroll(yi);
            Sy_Final.update(0).split(x,xo,xi,t1).reorder(ryt,xi,yo,xo).gpu_threads(xi).unroll(ryt);
            Sy_Final.update(1).split(x,xo,xi,t1).reorder(ryi,xi,yo,xo).gpu_threads(xi).unroll(ryi);
            Sy_Final.update(2).split(x,xo,xi,t1).reorder(ryt,xi,yo,xo).gpu_threads(xi).unroll(ryt);
            Sy_Final.update(3).split(x,xo,xi,t1).reorder(ryi,xi,yo,xo).gpu_threads(xi).unroll(ryi);

            Sy.compute_root();
            Sy.split(y,yo,yi,tile); //.reorder_storage(x,yi,yo);
            Sy.split(x,xo,xi,t1).reorder(yi,xi,yo,xo).gpu_blocks(yo,xo).gpu_threads(xi).unroll(yi);
        }
    }

    // ----------------------------------------------------------------------------------------------

    Target target = get_jit_target_from_environment();

    cerr << "\nJIT compilation ... " << endl;
    filterx.compile_jit(target, "stmt.html");

    cerr << "\nRunning ... " << endl;
    Buffer out(type_of<float>(), width, height);
    filterx.realize(out, iter);

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
