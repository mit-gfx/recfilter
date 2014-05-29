#include <iostream>
#include <Halide.h>

#include "../../lib/split.h"

#include "ls_hist_defs.h"

#define MAX_THREAD 192

using namespace Halide;

using std::vector;
using std::string;
using std::cerr;
using std::endl;


int main(int argc, char **argv) {
    Arguments args("lshist", argc, argv);

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

    int box_x = width/BOX_FILTER_FACTOR;        // radius of histogram filter
    int box_y = height/BOX_FILTER_FACTOR;

    Var x("x"), y("y"), b("b");

    // input image
    Func I("Input");
    I(x,y) = select((x<0 || y<0 || x>image.width()-1 || y>image.height()-1), 0, image(clamp(x,0,image.width()-1),clamp(y,0,image.height()-1)));


    // Lookup table for converting pixel value into smooth histogram
    Func L("Lookup_table");
    vector<Expr> hist_tuple;
    hist_tuple.push_back( SMOOTH_HIST(I(x,y)-BIN_CENTER0) );
    hist_tuple.push_back( SMOOTH_HIST(I(x,y)-BIN_CENTER1) );
    hist_tuple.push_back( SMOOTH_HIST(I(x,y)-BIN_CENTER2) );
    hist_tuple.push_back( SMOOTH_HIST(I(x,y)-BIN_CENTER3) );
    hist_tuple.push_back( SMOOTH_HIST(I(x,y)-BIN_CENTER4) );
    hist_tuple.push_back( SMOOTH_HIST(I(x,y)-BIN_CENTER5) );
    hist_tuple.push_back( SMOOTH_HIST(I(x,y)-BIN_CENTER6) );
    hist_tuple.push_back( SMOOTH_HIST(I(x,y)-BIN_CENTER7) );
    hist_tuple.push_back( SMOOTH_HIST(I(x,y)-BIN_CENTER8) );
    hist_tuple.push_back( SMOOTH_HIST(I(x,y)-BIN_CENTER9) );
    hist_tuple.push_back( SMOOTH_HIST(I(x,y)-BIN_CENTER10) );
    hist_tuple.push_back( SMOOTH_HIST(I(x,y)-BIN_CENTER11) );
    hist_tuple.push_back( SMOOTH_HIST(I(x,y)-BIN_CENTER12) );
    hist_tuple.push_back( SMOOTH_HIST(I(x,y)-BIN_CENTER13) );
    hist_tuple.push_back( SMOOTH_HIST(I(x,y)-BIN_CENTER14) );
    L(x,y) = hist_tuple;


    // ----------------------------------------------------------------------------------------------

    RDom rx(1, image.width()-1, "rx");
    RDom ry(1, image.height()-1,"ry");

    // summed area table for computing local histogram
    Func S("S");
    S(x, y, b) = L(x,y)[b];
    S(rx,y, b) = S(rx,y, b) + S(rx-1,y, b);
    S(x,ry, b) = S(x,ry, b) + S(x,ry-1, b);

    // histogram computation
    Func H("Hist");             // histogram
    H(x, y, b) = S(min(x+box_x,image.width()-1), min(y+box_y,image.height()-1),b)
           + select(x-box_x-1<0 || y-box_y-1<0, 0, S(max(x-box_x-1,0), max(y-box_y-1,0),b))
           - select(y-box_y-1<0, 0, S(min(x+box_x,image.width()-1), max(y-box_y-1,0),b))
           - select(x-box_x-1<0, 0, S(max(x-box_x-1,0), min(y+box_y,image.height()-1),b));

    // spatial smoothing of histogram
    Func G("Gaussian");


    // ----------------------------------------------------------------------------------------------

    Var xi("xi"), yi("yi");
    Var xo("xo"), yo("yo");

    RDom rxi(1, tile_width-1, "rxi");
    RDom ryi(1, tile_width-1, "ryi");

    split(S,Internal::vec(0,1),
            Internal::vec(x,y),
            Internal::vec(xi,yi),
            Internal::vec(xo,yo),
            Internal::vec(rx,ry),
            Internal::vec(rxi,ryi));

    // ----------------------------------------------------------------------------------------------

    float_dependencies_to_root(S);
    inline_function(S, "S--Intra_y-Deps_x");
    inline_function(S, "S--Intra_y");
    inline_function(S, "S--Deps_y");
    swap_variables (S, "S--Intra_y-Tail_x", xi, yi);
    merge(S, "S--Intra_y-Tail_x", "S--Tail_y", "S--Tail");
    recompute(S, "S", "S--Intra_y-Intra_x");
    inline_function(H, "L");
    inline_function(H, "S");

    // ----------------------------------------------------------------------------------------------

    vector<Func> func_list;
    extract_func_calls(S, func_list);

    map<string,Func> functions;
    for (size_t i=0; i<func_list.size(); i++) {
        cerr << func_list[i] << endl;
        functions[func_list[i].name()] = func_list[i];
    }

    Func S_intra0= functions["S--Intra_y-Intra_x"];
    Func S_tails = functions["S--Tail"];
    Func S_intra = functions["S--Intra_y-Intra_x-Recomp"];
    Func S_ctaily= functions["S--CTail_y"];
    Func S_ctailx= functions["S--Intra_y-CTail_x"];

    Target target = get_jit_target_from_environment();
    if (target.has_gpu_feature() || (target.features & Target::GPUDebug)) {
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
        S_ctaily.reorder_storage(xi,xo,yi,yo);
        S_ctaily.split(xo,xo,t,MAX_THREAD/tile_width);
        S_ctaily.reorder(yo,yi,xi,t,xo).gpu_blocks(xo).gpu_threads(xi,t);
        S_ctaily.update().split(xo,xo,t,MAX_THREAD/tile_width);
        S_ctaily.update().reorder(xi,t,xo).gpu_blocks(xo).gpu_threads(xi,t);

        S_ctailx.compute_root();
        S_ctailx.reorder_storage(yi,yo,xi,xo);
        S_ctailx.split(yo,yo,t,MAX_THREAD/tile_width);
        S_ctailx.reorder(xo,xi,yi,t,yo).gpu_blocks(yo).gpu_threads(yi,t);
        S_ctailx.update().split(yo,yo,t,MAX_THREAD/tile_width);
        S_ctailx.update().reorder(yi,t,yo).gpu_blocks(yo).gpu_threads(yi,t);

        S_intra.compute_at(H, Var("blockidx"));
        S_intra.reorder_storage(xi,yi,xo,yo);
        //S_intra.split(yi,t,yi, MAX_THREAD/tile_width).reorder(t,xi,yi,xo,yo).gpu_threads(xi,yi);
        S_intra.reorder(xi,yi,xo,yo).gpu_threads(yi);
        S_intra.update(0).reorder(rxi.x,yi,xo,yo).gpu_threads(yi);
        S_intra.update(1).reorder(ryi.x,xi,xo,yo).gpu_threads(xi);

        H.compute_root();
        H.split(x, xo,xi, tile_width).split(y, yo,yi, tile_width);
        //H.split(yi,t,yi, MAX_THREAD/tile_width).reorder(t,xi,yi,xo,yo);
        H.reorder(yi, xi, xo, yo);
        H.gpu_blocks(xo,yo).gpu_threads(xi);
        H.bound(x, 0, image.width()).bound(y, 0, image.height());
    }
    else {
        cerr << "Warning: No CPU scheduling" << endl;
    }

    // ----------------------------------------------------------------------------------------------

    cerr << "\nJIT compilation ... " << endl;
    B.compile_jit();

    Buffer hl_out_buff(type_of<int>(), width,height);
    {
        Timer t("Running ... ");
        for (int k=0; k<iterations; k++) {
            B.realize(hl_out_buff);
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
                ref(x,y) = 0;
                for (int v=y-box_y; v<=y+box_y; v++) {
                    for (int u=x-box_x; u<=x+box_x; u++) {
                        if (v>=0 && u>=0 && v<height && u<width)
                            ref(x,y) += random_image(u,v);
                    }
                }
            }
        }

        cerr << CheckResult(ref,hl_out) << endl;
    }

    return 0;
}
