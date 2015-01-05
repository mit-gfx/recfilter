#include <iostream>
#include <Halide.h>

#include "recfilter.h"

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
    int   tile_width = args.block;
    int   iter    = args.iterations;

    Image<float> random_image = generate_random_image<float>(width,height);

    ImageParam image(type_of<float>(), 2);
    image.set(random_image);

    // ----------------------------------------------------------------------------------------------

    RecFilterDim x("x", width);
    RecFilterDim y("y", height);

    RecFilter F;

    F(x,y) = image(x,y);

    F.add_filter(+x, {1.0, 1.0});
    F.add_filter(+y, {1.0, 1.0});

    F.split(x, tile_width, y, tile_width);

///     F.transpose_dimensions ("SAT_Intra_Tail_y_1", "xi", "yi");
///     F.interleave_func("SAT_Intra_Tail_x_0", "SAT_Intra_Tail_y_1", "SAT_Intra_Tail", "xi", 1);
///
///     F.remove_pure_def("SAT_Intra_CTail_x_0_Sub");
///     F.remove_pure_def("SAT_Intra_CTail_y_1_Sub");
///
///     cerr << F << endl;
///
///    // ----------------------------------------------------------------------------------------------
///
///    {
///        Var t("t");
///
///        RecFilterDim x", channels)i("xi"), yi("yi"), rxi("rxi"), rxf("rxf"), rxt("rxt");
///        RecFilterDim x", channels)o("xo"), yo("yo"), ryi("ryi"), ryf("ryf"), ryt("ryt");
///
///        Var rxox("rxo.x$r"), rxoy("rxo.y$r");
///        Var ryox("ryo.x$r"), ryoy("ryo.y$r");
///
///        Func SAT             = F.func("SAT");
///        Func SAT_Final       = F.func("SAT_Final");
///        Func SAT_Intra       = F.func("SAT_Intra");
///        Func SAT_Tail        = F.func("SAT_Intra_Tail");
///        Func SAT_CTail_x     = F.func("SAT_Intra_CTail_x_0_Sub");
///        Func SAT_CTail_y     = F.func("SAT_Intra_CTail_y_1_Sub");
///        Func SAT_CTail_xy    = F.func("SAT_Intra_CTail_x_0_y_1");
///        Func SAT_CTail_xy_sub= F.func("SAT_Intra_CTail_x_0_y_1_Sub");
///        Func SAT_Deps_x      = F.func("SAT_Intra_Deps_x_0");
///        Func SAT_Deps_y      = F.func("SAT_Intra_Deps_y_1");
///
///        SAT_Intra.compute_at(SAT_Tail, Var::gpu_blocks());
///        SAT_Intra.update(0).split(ryi,yi,t,UNROLL).reorder(t,rxi,yi,xo,yo).gpu_threads(rxi,yi).unroll(t);
///        SAT_Intra.update(1).reorder(rxi,ryi,xo,yo).gpu_threads(ryi).unroll(rxi);
///        SAT_Intra.update(2).reorder(rxt,ryi,xo,yo).gpu_threads(ryi).unroll(rxt);
///        SAT_Intra.update(3).reorder(ryi,rxi,xo,yo).gpu_threads(rxi).unroll(ryi);
///        SAT_Intra.update(4).reorder(ryt,rxi,xo,yo).gpu_threads(rxi).unroll(ryt);
///
///        SAT_Tail.compute_root().reorder_storage(xi,yi,xo,yo);
///        SAT_Tail.split(yi,yi,t,UNROLL).reorder(t,xi,yi,xo,yo).gpu(xo,yo,xi,yi).unroll(t);
///
///        SAT_Tail.compile_to_simplified_lowered_stmt("stmt2.html", 2, width/tile, tile, width/tile, HTML, get_jit_target_from_environment());
///
///        //
///
///        SAT_CTail_x.compute_root().reorder_storage(yi,xi,yo,xo);
///        SAT_CTail_x.update().reorder(rxox,rxoy,yi,yo).fuse(yi,yo,y).gpu_tile(y,MAX_THREADS);
///
///        //
///
///        SAT_CTail_xy_sub.compute_at(SAT_CTail_xy, Var::gpu_blocks());
///        SAT_CTail_xy_sub.split(xo,xo,x,TILES_PER_WARP).reorder(yi,xi,x,xo,yo).gpu_threads(x,yi);
///        SAT_CTail_xy_sub.update().split(xo,xo,x,TILES_PER_WARP).reorder(ryi,rxt,x,xo,yo).gpu_threads(x).unroll(ryi);
///
///        SAT_CTail_xy.compute_root().reorder_storage(yi,xi,yo,xo);
///        SAT_CTail_xy.split(xo,xo,x,TILES_PER_WARP).reorder(yi,xi,x,xo,yo).gpu_threads(x,yi).gpu_blocks(xo,yo);
///
///
///        SAT_CTail_y.compute_root().reorder_storage(xi,yi,xo,yo);
///        SAT_CTail_y.update().reorder(ryox,ryoy,xi,xo).fuse(xi,xo,x).gpu_tile(x,MAX_THREADS);
///
///        //
///
///        SAT_Deps_x.compute_at(SAT, Var::gpu_blocks()).reorder(xi,yi,xo,yo).gpu_threads(yi);
///        SAT_Deps_y.compute_at(SAT, Var::gpu_blocks()).reorder(xi,yi,xo,yo).gpu_threads(xi);
///
///        SAT_Final.compute_at(SAT, Var::gpu_blocks());
///        SAT_Final.update(0).split(ryi,yi,t,UNROLL).reorder(t,rxi,yi,xo,yo).gpu_threads(rxi,yi).unroll(t);
///        SAT_Final.update(1).reorder(rxt,ryi,xo,yo).gpu_threads(ryi).unroll(rxt);
///        SAT_Final.update(2).reorder(rxf,ryi,xo,yo).gpu_threads(ryi).unroll(rxf);
///        SAT_Final.update(3).reorder(ryt,rxi,xo,yo).gpu_threads(rxi).unroll(ryt);
///        SAT_Final.update(4).reorder(ryf,rxi,xo,yo).gpu_threads(rxi).unroll(ryf);
///
///        SAT.compute_root();
///        SAT.split(x,xo,xi,tile).split(y,yo,yi,tile);
///        SAT.split(yi,yi,t,UNROLL).reorder(t,xi,yi,xo,yo).gpu(xo,yo,xi,yi).unroll(t);
///
///        SAT.bound(x,0,width).bound(y,0,height);
///    }

    // ----------------------------------------------------------------------------------------------

    // innermost: unrolled dimensions, then gpu_thread dimensions, then gpu_block dimensions

    int unroll_w = 4;
    int tiles_per_warp = 4;

    F.intra_schedule().compute_locally()
        .reorder_storage(F.tail(), F.inner(), F.outer())
        .unroll         (F.inner_scan());

    F.intra_schedule(1)
        .split          (F.inner(1), unroll_w)
        .unroll         (F.inner(1).split_var())
        .reorder        (F.inner_scan(), F.inner(1).split_var(), F.tail(), F.inner(), F.outer())
        .gpu_threads    (F.inner(0), F.inner(1))
        .gpu_blocks     (F.outer(0), F.outer(1));

    F.intra_schedule(2)
        .split          (F.outer(0), 4)
        .reorder        (F.inner_scan(), F.tail(), F.inner(), F.outer(0).split_var(), F.outer())
        .gpu_threads    (F.outer(0).split_var(), F.inner(0))
        .gpu_blocks     (F.outer(0), F.outer(1));

    F.inter_schedule().compute_globally()
        .reorder_storage(F.inner(), F.tail(), F.outer())
        .unroll         (F.outer_scan())
        .split          (F.outer(0), 2)
        .reorder        (F.outer_scan(), F.inner(), F.tail(), F.outer(0).split_var(), F.outer())
        .gpu_threads    (F.inner(0), F.outer(0).split_var())
        .gpu_blocks     (F.outer(0));

    cerr << F << endl;

    cerr << "\nJIT compilation ... " << endl;
    F.compile_jit("hl_stmt.html");

    cerr << "\nRunning ... " << endl;
    Buffer out(type_of<float>(), width, height);
    float time = F.realize(out, iter);
    cerr << width << "\t" << time << endl;

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

        cout << CheckResult<float>(ref,hl_out) << endl;
    }

    return 0;
}
