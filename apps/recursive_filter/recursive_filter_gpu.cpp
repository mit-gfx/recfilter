#include <iostream>
#include <cstdio>
#include <cmath>

#include <Halide.h>

#include "../../lib/split.h"

#define WARP_SIZE  32
#define MAX_THREAD 192

using namespace Halide;

using std::vector;
using std::string;
using std::cerr;
using std::endl;

template<typename T>
vector<T> vec(T a, T b) { return Halide::Internal::vec(a,b); }


int main(int argc, char **argv) {

    Arguments args("recursive_filter", argc, argv);

    bool  debug   = args.debug;
    bool  verbose = args.verbose;
    bool  nocheck = args.nocheck;
    int   width   = args.width;
    int   height  = args.height;
    int   tile_width = args.block;
    int   iterations = args.iterations;

    Image<int> image = generate_random_image<int>(width,height);

    // ----------------------------------------------------------------------------------------------

    //
    // Algorithm
    //

    int filter_order = 1;

    Func I("Input");
    Func S("S");

    Var x("x");
    Var y("y");

    RDom rx(1, width-1, "rx");
    RDom ry(1, height-1,"ry");

    I(x,y) = select((x<0 || y<0 || x>width-1 || y>height-1), 0, image(clamp(x,0,width-1),clamp(y,0,height-1)));

    S(x, y) = I(x,y);
    S(rx,y) = S(rx,y) + S(rx-1,y);
    S(x,ry) = S(x,ry) + S(x,ry-1);

    // ----------------------------------------------------------------------------------------------

    //
    // Splitting into tiles
    //

    Var xi("xi"), yi("yi");
    Var xo("xo"), yo("yo");

    RDom rxi(1, tile_width-1,       "rxi");
    RDom ryi(1, tile_width-1,       "ryi");
    RDom rxo(1, width / tile_width, "rxo");
    RDom ryo(1, height/ tile_width, "ryo");

    split(S, vec(0,1), vec(x,y), vec(xi,yi), vec(xo,yo),
             vec(rx,ry), vec(rxi,ryi), vec(rxo,ryo), filter_order);

    // ----------------------------------------------------------------------------------------------


    //
    // Reordering split terms
    //

    float_dependencies_to_root(S);
    merge_and_inline(S,
            "S$split$$Intra2_x$$Intra2_y$",
            "S$split$$Intra2_x$$Intra_y$",
            "S$split$$Intra_x$$Intra2_y$",
            "SIntra_Tail");
    merge_and_inline(S, "S$split$$Intra_x$$Tail_y$", "S$split$$Intra2_x$$Tail_y$" , "STail_y");
    merge_and_inline(S, "S$split$$Intra_x$$CTail_y$", "S$split$$Intra2_x$$CTail_y$", "SCTail_y");
    inline_function (S, "S$split$$Intra2_x$$Deps_y$");
    inline_function (S, "S$split$$Intra2_x$");
    swap_variables  (S, "STail_y", xi, yi);
    merge_and_inline(S, "STail_y", "S$split$$Tail_x$", "STail");
    inline_function (S, "S$split$$Intra_x$$Deps_y$");
    inline_function (S, "S$split$$Deps_x$");
    inline_function (S, "S$split$$Intra_x$");
    inline_function (S, "S$split$");
//    inline_non_split_functions(S, 2);

    // ----------------------------------------------------------------------------------------------

    //
    // Scheduling
    //

    vector<Func> func_list;
    extract_func_calls(S, func_list);

    map<string,Func> functions;
    for (size_t i=0; i<func_list.size(); i++) {
        cerr << func_list[i] << endl;
        functions[func_list[i].name()] = func_list[i];
    }


    Target target = get_jit_target_from_environment();
    if (target.has_gpu_feature() || (target.features & Target::GPUDebug)) {
        Func S_intra0= functions["SIntra_Tail"];
        Func S_tails = functions["STail"];
        Func S_intra = functions["S$split$$Intra_x$$Intra_y$"];
        Func S_ctailx= functions["S$split$$CTail_x$"];
        Func S_ctaily= functions["SCTail_y"];
        Func S_final = functions["S"];

        Var t("t");

        S_intra0.compute_at(S_tails, Var("blockidx"));
        //S_intra0.split(yi,t,yi, MAX_THREAD/WARP_SIZE).reorder(t,xi,yi,xo,yo).gpu_threads(xi,yi).gpu_blocks(xo,yo);
        S_intra0.reorder_storage(xi,yi,xo,yo);
        S_intra0.reorder(xi,yi,xo,yo).gpu_threads(yi);
        S_intra0.update(0).reorder(rxi.x,yi,xo,yo).gpu_threads(yi);
        S_intra0.update(1).reorder(ryi.x,xi,xo,yo).gpu_threads(xi);

        S_tails.compute_root();
        S_tails.reorder_storage(yi,xi,xo,yo);
        //S_tails.split(yi,t,yi, MAX_THREAD/WARP_SIZE).reorder(t,xi,yi,xo,yo);
        S_tails.reorder(xi,yi,xo,yo);
        S_tails.gpu_blocks(xo,yo).gpu_threads(xi);

        S_ctaily.compute_root();
        S_ctailx.reorder_storage(yi,yo,xi,xo);
        S_ctaily.split(xo,xo,t,MAX_THREAD/tile_width);
        S_ctaily.reorder(yo,yi,xi,t,xo).gpu_blocks(xo).gpu_threads(xi,t);
        S_ctaily.update().split(xo,xo,t,MAX_THREAD/tile_width);
        S_ctaily.update().reorder(ryo.x,yi,xi,t,xo).gpu_blocks(xo).gpu_threads(xi,t);

        S_ctailx.compute_root();
        S_ctailx.reorder_storage(yi,yo,xi,xo);
        S_ctailx.split(yo,yo,t,MAX_THREAD/tile_width);
        S_ctailx.reorder(xo,xi,yi,t,yo).gpu_blocks(yo).gpu_threads(yi,t);
        S_ctailx.update().split(yo,yo,t,MAX_THREAD/tile_width);
        S_ctailx.update().reorder(rxo.x,xi,yi,t,yo).gpu_blocks(yo).gpu_threads(yi,t);

        S_intra.compute_at(S_final, Var("blockidx"));
        S_intra.reorder_storage(xi,yi,xo,yo);
        //S_intra.split(yi,t,yi, MAX_THREAD/tile_width).reorder(t,xi,yi,xo,yo).gpu_threads(xi,yi);
        S_intra.reorder(xi,yi,xo,yo).gpu_threads(yi);
        S_intra.update(0).reorder(rxi.x,yi,xo,yo).gpu_threads(yi);
        S_intra.update(1).reorder(ryi.x,xi,xo,yo).gpu_threads(xi);

        S_final.compute_root();
        S_final.split(x, xo,xi, tile_width).split(y, yo,yi, tile_width);
        //S_final.split(yi,t,yi, MAX_THREAD/tile_width).reorder(t,xi,yi,xo,yo);
        S.reorder(yi, xi, xo, yo);
        S_final.gpu_blocks(xo,yo).gpu_threads(xi);
        S_final.bound(x, 0, width).bound(y, 0, height);
    }
    else {
        cerr << "Warning: No CPU scheduling" << endl;
    }

    // ----------------------------------------------------------------------------------------------

    //
    // JIT compilation
    //

    cerr << "\nJIT compilation ... " << endl;
    S.compile_jit();

    Buffer hl_out_buff(type_of<int>(), width,height);
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
        Image<int> hl_out(hl_out_buff);
        Image<int> diff(width,height);
        Image<int> ref   = reference_recursive_filter<int>(image);

        int diff_sum = 0;
        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {
                diff(x,y) = std::abs(ref(x,y) - hl_out(x,y));
                diff_sum += diff(x,y);
            }
        }

        if (verbose) {
            cerr << "Input" << endl << image << endl;
            cerr << "Reference" << endl << ref << endl;
            cerr << "Halide output" << endl << hl_out << endl;
            cerr << "Difference " << endl << diff << endl;
            cerr << "\nDifference = " << diff_sum << endl;
        } else {
            cerr << "\nDifference = " << diff_sum << endl;
            cerr << endl;
        }
    }

    return EXIT_SUCCESS;
}
