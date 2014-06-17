#include <iostream>
#include <Halide.h>

#include "../../lib/split.h"

#define MAX_THREAD 192

using namespace Halide;

using std::map;
using std::vector;
using std::string;
using std::cerr;
using std::endl;


int main(int argc, char **argv) {
    Arguments args("summed_table", argc, argv);

    bool  verbose = args.verbose;
    bool  nocheck = args.nocheck;
    int   width   = args.width;
    int   height  = args.height;
    int   tile_width = args.block;
    int   iterations = args.iterations;

    Image<float> random_image = generate_random_image<float>(width,height);

    ImageParam image(type_of<float>(), 2);
    image.set(random_image);

    // ----------------------------------------------------------------------------------------------

    Func I("Input");
    Func S("S");

    Var x("x");
    Var y("y");

    RDom rx(0, image.width(), "rx");
    RDom ry(0, image.height(),"ry");

    I(x,y) = select((x<0 || y<0 || x>image.width()-1 || y>image.height()-1), 0, image(clamp(x,0,image.width()-1),clamp(y,0,image.height()-1)));

    S(x, y) = I(x,y);
    S(rx,y) = S(rx,y) + select(rx>0, S(max(0,rx-1),y), 0);
    S(x,ry) = S(x,ry) + select(ry>0, S(x,max(0,ry-1)), 0);

    // ----------------------------------------------------------------------------------------------

    Var xi("xi"), yi("yi");
    Var xo("xo"), yo("yo");

    RDom rxi(0, tile_width, "rxi");
    RDom ryi(0, tile_width, "ryi");

    split(S,Internal::vec(0,1),   Internal::vec(x,y),
            Internal::vec(xi,yi), Internal::vec(xo,yo),
            Internal::vec(rx,ry), Internal::vec(rxi,ryi));

    // ----------------------------------------------------------------------------------------------

    inline_function(S, "S-Intra-Deps_x");
    inline_function(S, "S-Intra-Deps_y");
    swap_variables (S, "S-Intra-Tail_y_1", xi, yi);
    merge(S, "S-Intra-Tail_x_0", "S-Intra-Tail_y_1", "S-Tail");

    // ----------------------------------------------------------------------------------------------

    vector<Func> func_list;
    extract_func_calls(S, func_list);

    map<string,Func> functions;
    for (size_t i=0; i<func_list.size(); i++) {
        cerr << func_list[i] << endl;
        functions[func_list[i].name()] = func_list[i];
    }

    Func S_intra  = functions["S-Intra"];
    Func S_tails  = functions["S-Tail"];
    Func S_final  = functions["S-Final-Sub"];
    Func S_ctailx = functions["S-Intra-CTail_x_0"];
    Func S_ctaily = functions["S-Intra-CTail_y_1"];
    Func S_ctailxy= functions["S-Intra-CTail_x_0-y-1"];

    assert(S_intra  .defined());
    assert(S_tails  .defined());
    assert(S_final  .defined());
    assert(S_ctailx .defined());
    assert(S_ctaily .defined());
    assert(S_ctailxy.defined());

    Target target = get_jit_target_from_environment();
    if (target.has_gpu_feature() || (target.features & Target::GPUDebug)) {
        Var t("t");
        Var xs("ScanStage");

        //S_intra.compute_at(S_tails, Var("__block_id_x"));
        S_intra.compute_root();
        S_intra.split(yi,t,yi, MAX_THREAD/tile_width).reorder(xs,t,xi,yi,xo,yo);//.gpu_threads(xi,yi).gpu_blocks(xo,yo);
        S_intra.update(0).reorder(rxi.x,yi,xo,yo);//.gpu_threads(yi).gpu_blocks(xo,yo);
        S_intra.update(1).reorder(ryi.x,xi,xo,yo);//.gpu_threads(xi).gpu_blocks(xo,yo);

        S_tails.compute_root();
        S_tails.reorder_storage(yi,xi,xo,yo);
#if 0
        S_tails.split(yi,t,yi, MAX_THREAD/tile_width).reorder(t,xi,yi,xo,yo).gpu_threads(xi,yi);
#else
        S_tails.reorder(xi,yi,xo,yo).gpu_threads(yi);
#endif
        S_tails.gpu_blocks(xo,yo);

        S_ctailx.compute_root();
        S_ctailx.reorder_storage(yi,yo,xi,xo);
        S_ctailx.split(yo,yo,t,MAX_THREAD/tile_width);
        S_ctailx.reorder(xo,xi,yi,t,yo).gpu_blocks(yo).gpu_threads(yi,t);
        S_ctailx.update().split(yo,yo,t,MAX_THREAD/tile_width);
        S_ctailx.update().reorder(yi,t,yo).gpu_blocks(yo).gpu_threads(yi,t);

        S_ctailxy.compute_at(S_ctaily, Var("__block_id_x"));
        S_ctailxy.reorder(yo,yi,xi,xo).gpu_threads(xi);
        S_ctailxy.update().reorder(yo,ryi.x,xi,xo).gpu_threads(xi);

        S_ctaily.compute_root();
        S_ctaily.reorder_storage(xi,xo,yi,yo);
        S_ctaily.split(xo,xo,t,MAX_THREAD/tile_width);
        S_ctaily.reorder(yo,yi,xi,t,xo).gpu_blocks(xo).gpu_threads(xi,t);
        S_ctaily.update().split(xo,xo,t,MAX_THREAD/tile_width);
        S_ctaily.update().reorder(xi,t,xo).gpu_blocks(xo).gpu_threads(xi,t);

        S_final.compute_at(S, Var("__block_id_x"));
        S_final.split(yi,t,yi, MAX_THREAD/tile_width).reorder(t,xi,yi,xo,yo).gpu_threads(xi,yi);
        S_final.update(0).reorder(rxi.x,yi,xo,yo).gpu_threads(yi);
        S_final.update(1).reorder(ryi.x,xi,xo,yo).gpu_threads(xi);

        S.compute_root();
        S.split(x, xo,xi, tile_width).split(y, yo,yi, tile_width);
        S.split(yi,t,yi, MAX_THREAD/tile_width).reorder(t,xi,yi,xo,yo);
        S.gpu_blocks(xo,yo).gpu_threads(xi,yi);
        S.bound(x, 0, image.width()).bound(y, 0, image.height());
    }
    else {
        S_intra.compute_root();
        S_tails.compute_root();
        S_ctailx.compute_root();
        S_ctailxy.compute_root();
        S_ctaily.compute_root();
        S_final.compute_root();
        S.compute_root();
    }

    // ----------------------------------------------------------------------------------------------

    cerr << "\nJIT compilation ... " << endl;
    S.compile_jit();

    Buffer hl_out_buff(type_of<float>(), width,height);
    {
        Timer t("Running ... ");
        for (int k=0; k<iterations; k++) {
            S.realize(hl_out_buff);
            if (k < iterations-1) {
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

        cerr << CheckResultVerbose(ref,hl_out) << endl;
    }

    return 0;
}
