#include <iostream>
#include <cstdio>
#include <cmath>

#include <Halide.h>

#include "../../lib/recfilter.h"

#define MAX_THREADS 192
#define UNROLL      4

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

    Image<float> random_image = generate_random_image<float>(width,height);

    ImageParam image(type_of<float>(), 2);
    image.set(random_image);

    // ----------------------------------------------------------------------------------------------

    float b0 = 0.425294f;
    vector<float> W2;
    W2.push_back(0.885641f);
    W2.push_back(-0.310935f);
    int filter_order = W2.size();

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

    filterx.interleave_func("S_0_Intra_Tail_x_0", "S_0_Intra_Tail_x_1", "S_0_Intra_Tail", "xi", filter_order);
    filtery.interleave_func("S_1_Intra_Tail_y_0", "S_1_Intra_Tail_y_1", "S_1_Intra_Tail", "yi", filter_order);

    filterx.remove_pure_def("S_0_Intra_CTail_x_0_Sub");
    filterx.remove_pure_def("S_0_Intra_CTail_x_1_Sub");
    filtery.remove_pure_def("S_1_Intra_CTail_y_0_Sub");
    filtery.remove_pure_def("S_1_Intra_CTail_y_1_Sub");

    cerr << filtery << endl;

    // ----------------------------------------------------------------------------------------------

    {
        Var t("t");

        Var xi("xi"), yi("yi"), rxi("rxi"), rxt("rxt"), rxf("rxf");
        Var xo("xo"), yo("yo"), ryi("ryi"), ryt("ryt"), ryf("ryf");

        Var rxox("rxo.x$r"), rxoy("rxo.y$r");
        Var ryox("ryo.x$r"), ryoy("ryo.y$r");

        Func Sx         = filterx.func("S_0");
        Func Sx_Final   = filterx.func("S_0_Final");
        Func Sx_Intra   = filterx.func("S_0_Intra");
        Func Sx_Tail    = filterx.func("S_0_Intra_Tail");
        Func Sx_CTail_0 = filterx.func("S_0_Intra_CTail_x_0_Sub");
        Func Sx_CTail_1 = filterx.func("S_0_Intra_CTail_x_1_Sub");
        Func Sx_Deps_0  = filterx.func("S_0_Intra_Deps_x_0");
        Func Sx_Deps_1  = filterx.func("S_0_Intra_Deps_x_1");

        Func Sy         = filtery.func("S_1");
        Func Sy_Final   = filtery.func("S_1_Final");
        Func Sy_Intra   = filtery.func("S_1_Intra");
        Func Sy_Tail    = filtery.func("S_1_Intra_Tail");
        Func Sy_CTail_0 = filtery.func("S_1_Intra_CTail_y_0_Sub");
        Func Sy_CTail_1 = filtery.func("S_1_Intra_CTail_y_1_Sub");
        Func Sy_Deps_0  = filtery.func("S_1_Intra_Deps_y_0");
        Func Sy_Deps_1  = filtery.func("S_1_Intra_Deps_y_1");

#if 0
            Func P("P");
            P(xi, xo, yi, yo) = select(yi<tile, Sx_Final(xi, xo, yo*tile+yi), Sy_Tail(xo*tile+xi, yi-tile, yo));

            Sx_Intra.compute_at(Sx_Tail, Var::gpu_blocks());
            Sx_Intra.update(0).split(y,yo,yi,tile).split(yi,yi,t,UNROLL).reorder(t,rxi,yi,xo,yo).gpu_threads(rxi,yi).unroll(t);
            Sx_Intra.update(1).split(y,yo,yi,tile).reorder(rxi,yi,xo,yo).gpu_threads(yi).unroll(rxi);
            Sx_Intra.update(2).split(y,yo,yi,tile).reorder(rxt,yi,xo,yo).gpu_threads(yi).unroll(rxt);
            Sx_Intra.update(3).split(y,yo,yi,tile).reorder(rxi,yi,xo,yo).gpu_threads(yi).unroll(rxi);
            Sx_Intra.update(4).split(y,yo,yi,tile).reorder(rxt,yi,xo,yo).gpu_threads(yi).unroll(rxt);

            Sx_Tail.compute_root().reorder_storage(y,xi,xo);
            Sx_Tail.split(y,yo,yi,tile).split(yi,yi,t,UNROLL).reorder(t,xi,yi,xo,yo).gpu(xo,yo,xi,yi).unroll(t);

            Sx_CTail_0.compute_root().update().reorder(rxox,rxoy,y).vectorize(rxox).split(y,yo,yi,MAX_THREADS).gpu(yo,yi);
            Sx_CTail_1.compute_root().update().reorder(rxox,rxoy,y).vectorize(rxox).split(y,yo,yi,MAX_THREADS).gpu(yo,yi);

            Sx_Deps_0.compute_at(P, Var::gpu_blocks()).split(y,yo,yi,tile).reorder(xi,yi,xo,yo).gpu_threads(yi,xi);
            Sx_Deps_1.compute_at(P, Var::gpu_blocks()).split(y,yo,yi,tile).reorder(xi,yi,xo,yo).gpu_threads(yi,xi);

            Sx_Final.compute_at(P, Var::gpu_blocks());
            Sx_Final.update(0).split(y,yo,yi,tile).split(yi,yi,t,UNROLL).reorder(t,rxi,yi,xo,yo).gpu_threads(rxi,yi).unroll(t);
            Sx_Final.update(1).split(y,yo,yi,tile).reorder(rxt,yi,xo,yo).gpu_threads(yi).unroll(rxt);
            Sx_Final.update(2).split(y,yo,yi,tile).reorder(rxf,yi,xo,yo).gpu_threads(yi).unroll(rxf);
            Sx_Final.update(3).split(y,yo,yi,tile).reorder(rxt,yi,xo,yo).gpu_threads(yi).unroll(rxt);
            Sx_Final.update(4).split(y,yo,yi,tile).reorder(rxf,yi,xo,yo).gpu_threads(yi).unroll(rxf);

            Sy_Intra.compute_at(P, Var::gpu_blocks());
            Sy_Intra.update(0).split(x,xo,xi,tile).split(xi,xi,t,UNROLL).reorder(t,ryi,xi,yo,xo).gpu_threads(ryi,xi).unroll(t);
            Sy_Intra.update(1).split(x,xo,xi,tile).reorder(ryi,xi,yo,xo).gpu_threads(xi).unroll(ryi);
            Sy_Intra.update(2).split(x,xo,xi,tile).reorder(ryt,xi,yo,xo).gpu_threads(xi).unroll(ryt);
            Sy_Intra.update(3).split(x,xo,xi,tile).reorder(ryi,xi,yo,xo).gpu_threads(xi).unroll(ryi);
            Sy_Intra.update(4).split(x,xo,xi,tile).reorder(ryt,xi,yo,xo).gpu_threads(xi).unroll(ryt);

            P.compute_root();
            P.split(xi,xi,t,UNROLL).reorder(t,yi,xi,yo,xo).gpu(yo,xo,yi,xi).unroll(t);

            P.bound(xo, 0, width/tile).bound(yo, 0, height/tile).bound(xi, 0, tile).bound(yi, 0, tile+2*filter_order);


            Target target = get_jit_target_from_environment();
            for (int i=0; i<iter; i++) {
                P.realize(tile, width/tile, tile+2*filter_order, height/tile);
            }

            return 0;
#endif


        {
            // stage 1: x filtering

            Sx_Intra.compute_at(Sx_Tail, Var::gpu_blocks());
            Sx_Intra.update(0).split(y,yo,yi,tile).split(yi,yi,t,UNROLL).reorder(t,rxi,yi,xo,yo).gpu_threads(rxi,yi).unroll(t);
            Sx_Intra.update(1).split(y,yo,yi,tile).reorder(rxi,yi,xo,yo).gpu_threads(yi).unroll(rxi);
            Sx_Intra.update(2).split(y,yo,yi,tile).reorder(rxt,yi,xo,yo).gpu_threads(yi).unroll(rxt);
            Sx_Intra.update(3).split(y,yo,yi,tile).reorder(rxi,yi,xo,yo).gpu_threads(yi).unroll(rxi);
            Sx_Intra.update(4).split(y,yo,yi,tile).reorder(rxt,yi,xo,yo).gpu_threads(yi).unroll(rxt);

            Sx_Tail.compute_root().reorder_storage(y,xi,xo);
            Sx_Tail.split(y,yo,yi,tile).split(yi,yi,t,UNROLL).reorder(t,xi,yi,xo,yo).gpu(xo,yo,xi,yi).unroll(t);

            // stage 2 and 3

            Sx_CTail_0.compute_root().update().reorder(rxox,rxoy,y).split(y,yo,yi,MAX_THREADS).gpu(yo,yi).vectorize(rxox);
            Sx_CTail_1.compute_root().update().reorder(rxox,rxoy,y).split(y,yo,yi,MAX_THREADS).gpu(yo,yi).vectorize(rxox);

            // stage 4

            Sx_Deps_0.compute_at(Sx, Var::gpu_blocks()).split(y,yo,yi,tile).reorder(xi,yi,xo,yo).gpu_threads(yi,xi);
            Sx_Deps_1.compute_at(Sx, Var::gpu_blocks()).split(y,yo,yi,tile).reorder(xi,yi,xo,yo).gpu_threads(yi,xi);

            Sx_Final.compute_at(Sx, Var::gpu_blocks());
            Sx_Final.update(0).split(y,yo,yi,tile).split(yi,yi,t,UNROLL).reorder(t,rxi,yi,xo,yo).gpu_threads(rxi,yi).unroll(t);
            Sx_Final.update(1).split(y,yo,yi,tile).reorder(rxt,yi,xo,yo).gpu_threads(yi).unroll(rxt);
            Sx_Final.update(2).split(y,yo,yi,tile).reorder(rxf,yi,xo,yo).gpu_threads(yi).unroll(rxf);
            Sx_Final.update(3).split(y,yo,yi,tile).reorder(rxt,yi,xo,yo).gpu_threads(yi).unroll(rxt);
            Sx_Final.update(4).split(y,yo,yi,tile).reorder(rxf,yi,xo,yo).gpu_threads(yi).unroll(rxf);

            Sx.compute_root().reorder_storage(y,x).split(x,xo,xi,tile).split(y,yo,yi,tile);
            Sx.split(yi,yi,t,UNROLL).reorder(t,xi,yi,xo,yo).gpu(xo,yo,xi,yi).unroll(t);

            // stage 5: y filtering

            Sy_Intra.compute_at(Sy_Tail, Var::gpu_blocks());
            Sy_Intra.update(0).split(x,xo,xi,tile).split(xi,xi,t,UNROLL).reorder(t,ryi,xi,yo,xo).gpu_threads(ryi,xi).unroll(t);
            Sy_Intra.update(1).split(x,xo,xi,tile).reorder(ryi,xi,yo,xo).gpu_threads(xi).unroll(ryi);
            Sy_Intra.update(2).split(x,xo,xi,tile).reorder(ryt,xi,yo,xo).gpu_threads(xi).unroll(ryt);
            Sy_Intra.update(3).split(x,xo,xi,tile).reorder(ryi,xi,yo,xo).gpu_threads(xi).unroll(ryi);
            Sy_Intra.update(4).split(x,xo,xi,tile).reorder(ryt,xi,yo,xo).gpu_threads(xi).unroll(ryt);

            Sy_Tail.compute_root();
            Sy_Tail.split(x,xo,xi,tile).split(xi,xi,t,UNROLL).reorder(t,yi,xi,yo,xo).gpu(yo,xo,yi,xi).unroll(t);

            // stage 6 and 7

            Sy_CTail_0.compute_root().update().reorder(ryox,ryoy,x).split(x,xo,xi,MAX_THREADS).gpu(xo,xi).vectorize(ryox);
            Sy_CTail_1.compute_root().update().reorder(ryox,ryoy,x).split(x,xo,xi,MAX_THREADS).gpu(xo,xi).vectorize(ryox);

            // stage 8

            Sy_Deps_0.compute_at(Sy, Var::gpu_blocks()).split(x,xo,xi,tile).reorder(yi,xi,yo,xo).gpu_threads(xi);
            Sy_Deps_1.compute_at(Sy, Var::gpu_blocks()).split(x,xo,xi,tile).reorder(yi,xi,yo,xo).gpu_threads(xi);

            Sy_Final.compute_at(Sy, Var::gpu_blocks());
            Sy_Final.update(0).split(x,xo,xi,tile).split(xi,xi,t,UNROLL).reorder(t,ryi,xi,yo,xo).gpu_threads(ryi,xi).unroll(t);
            Sy_Final.update(1).split(x,xo,xi,tile).reorder(ryt,xi,yo,xo).gpu_threads(xi).unroll(ryt);
            Sy_Final.update(2).split(x,xo,xi,tile).reorder(ryf,xi,yo,xo).gpu_threads(xi).unroll(ryf);
            Sy_Final.update(3).split(x,xo,xi,tile).reorder(ryt,xi,yo,xo).gpu_threads(xi).unroll(ryt);
            Sy_Final.update(4).split(x,xo,xi,tile).reorder(ryf,xi,yo,xo).gpu_threads(xi).unroll(ryf);

            Sy.compute_root().reorder_storage(y,x).split(x,xo,xi,tile).split(y,yo,yi,tile);
            Sy.split(xi,xi,t,UNROLL).reorder(t,yi,xi,yo,xo).gpu(yo,xo,yi,xi).unroll(t);


            // bounds

            Sx_Tail.bound(xo,0,width/tile).bound(y,0,height).bound(xi,0,2*filter_order);
            Sy_Tail.bound(yo,0,width/tile).bound(x,0,height).bound(yi,0,2*filter_order);

            Sx_CTail_0.bound(xo,0,width/tile).bound(y,0,height).bound(xi,0,filter_order);
            Sx_CTail_1.bound(xo,0,width/tile).bound(y,0,height).bound(xi,0,filter_order);

            Sy_CTail_0.bound(yo,0,width/tile).bound(x,0,height).bound(yi,0,filter_order);
            Sy_CTail_1.bound(yo,0,width/tile).bound(x,0,height).bound(yi,0,filter_order);

            Sx.bound(y,0,height).bound(x,0,width);
            Sy.bound(y,0,height).bound(x,0,width);
        }
    }

    // ----------------------------------------------------------------------------------------------

    Target target = get_jit_target_from_environment();

    cerr << "\nJIT compilation ... " << endl;
    filtery.compile_jit(target, "stmt.html");

    cerr << "\nRunning ... " << endl;
    Buffer out(type_of<float>(), width, height);
    filtery.realize(out, iter);

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
                    + W2[1]*ref(std::max(x-2,0),y)
                    ;
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
