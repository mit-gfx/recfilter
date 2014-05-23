#include <iostream>
#include <cstdio>
#include <cmath>

#include <Halide.h>

#include "../../lib/split.h"

#define MAX_THREAD 192

using namespace Halide;

using std::vector;
using std::string;
using std::cerr;
using std::endl;


int main(int argc, char **argv) {
    Arguments args("gauss_cognex", argc, argv);

    bool  verbose = args.verbose;
    bool  nocheck = args.nocheck;
    int   width   = args.width;
    int   height  = args.height;
    int   tile_width = args.block;
    int   iterations = args.iterations;

    Image<int> random_image = generate_random_image<int>(width,height);

    ImageParam image(type_of<int>(), 2);
    image.set(random_image);

    // ----------------------------------------------------------------------------------------------

    Func I("Input");
    Func S("S");

    Var x("x");
    Var y("y");

    RDom rx(1, image.width()-1, "rx");
    RDom ry(1, image.height()-1,"ry");
    RDom rz(1, image.width()-1, "rz");
    RDom rw(1, image.height()-1,"rw");

    I(x,y) = select((x<0 || y<0 || x>image.width()-1 || y>image.height()-1), 0, image(clamp(x,0,image.width()-1),clamp(y,0,image.height()-1)));

    S(x, y) = I(x,y);
    S(rx,y) = S(rx,y) + S(rx-1,y);
    S(x,ry) = S(x,ry) + S(x,ry-1);
    S(rz,y) = S(rz,y) + S(rz-1,y);
    S(x,rw) = S(x,rw) + S(x,rw-1);

    // ----------------------------------------------------------------------------------------------

    Var xi("xi"), yi("yi");
    Var xo("xo"), yo("yo");

    RDom rxi(1, tile_width-1, "rxi");
    RDom ryi(1, tile_width-1, "ryi");
    RDom rzi(1, tile_width-1, "rzi");
    RDom rwi(1, tile_width-1, "rwi");

    split(S, Internal::vec(0,1,0,1),     Internal::vec(x,y,x,y),     Internal::vec(xi,yi,xi,yi),
             Internal::vec(xo,yo,xo,yo), Internal::vec(rx,ry,rz,rw), Internal::vec(rxi,ryi,rzi,rwi));

    // ----------------------------------------------------------------------------------------------

    float_dependencies_to_root(S);

    swap_variables (S, "S--Intra_y-Intra_x-Intra_y-Tail_x", xi, yi);
    swap_variables (S, "S--Intra_y-Tail_x", xi, yi);
    merge(S,"S--Intra_y-Intra_x-Intra_y-Tail_x",
            "S--Intra_y-Intra_x-Tail_y",
            "S--Intra_y-Tail_x",
            "S--Tail_y",
            "S--Tail");

    inline_function(S, "S--Intra_y-Intra_x-Intra_y-Deps_x");
    inline_function(S, "S--Intra_y-Intra_x-Deps_y");
    inline_function(S, "S--Intra_y-Deps_x");
    inline_function(S, "S--Deps_y");

    inline_function(S, "S--Intra_y-Intra_x-Intra_y");
    inline_function(S, "S--Intra_y-Intra_x");
    inline_function(S, "S--Intra_y");

    recompute(S, "S", "S--Intra_y-Intra_x-Intra_y-Intra_x");

    // ----------------------------------------------------------------------------------------------

    vector<Func> func_list;
    extract_func_calls(S, func_list);

    map<string,Func> functions;
    for (size_t i=0; i<func_list.size(); i++) {
        cerr << func_list[i] << endl;
        functions[func_list[i].name()] = func_list[i];
    }

    Func S_intra0 = functions["S--Intra_y-Intra_x-Intra_y-Intra_x"];
    Func S_intra  = functions["S--Intra_y-Intra_x-Intra_y-Intra_x-Recomp"];
    Func S_tails  = functions["S--Tail"];
    Func S_ctailw = functions["S--CTail_y"];
    Func S_ctailz = functions["S--Intra_y-CTail_x"];
    Func S_ctaily = functions["S--Intra_y-Intra_x-CTail_y"];
    Func S_ctailx = functions["S--Intra_y-Intra_x-Intra_y-CTail_x"];

    assert(S_intra0.defined());
    assert(S_tails.defined());
    assert(S_intra.defined());
    assert(S_ctailx.defined());
    assert(S_ctaily.defined());
    assert(S_ctailz.defined());
    assert(S_ctailw.defined());

    Target target = get_jit_target_from_environment();
    if (target.has_gpu_feature() || (target.features & Target::GPUDebug)) {
        Var t("t");

        S_intra0.compute_at(S_tails, Var("blockidx"));
        //S_intra0.split(yi,t,yi, MAX_THREAD/WARP_SIZE).reorder(t,xi,yi,xo,yo).gpu_threads(xi,yi).gpu_blocks(xo,yo);
        S_intra0.reorder_storage(xi,yi,xo,yo);
        S_intra0.reorder(xi,yi,xo,yo).gpu_threads(yi);
        S_intra0.update(0).reorder(rxi.x,yi,xo,yo).gpu_threads(yi);
        S_intra0.update(1).reorder(ryi.x,xi,xo,yo).gpu_threads(xi);
        S_intra0.update(2).reorder(rzi.x,yi,xo,yo).gpu_threads(yi);
        S_intra0.update(3).reorder(rwi.x,xi,xo,yo).gpu_threads(xi);

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

        S_ctailw.compute_root();
        S_ctailw.reorder_storage(xi,xo,yi,yo);
        S_ctailw.split(xo,xo,t,MAX_THREAD/tile_width);
        S_ctailw.reorder(yo,yi,xi,t,xo).gpu_blocks(xo).gpu_threads(xi,t);
        S_ctailw.update().split(xo,xo,t,MAX_THREAD/tile_width);
        S_ctailw.update().reorder(xi,t,xo).gpu_blocks(xo).gpu_threads(xi,t);

        S_ctailz.compute_root();
        S_ctailz.reorder_storage(yi,yo,xi,xo);
        S_ctailz.split(yo,yo,t,MAX_THREAD/tile_width);
        S_ctailz.reorder(xo,xi,yi,t,yo).gpu_blocks(yo).gpu_threads(yi,t);
        S_ctailz.update().split(yo,yo,t,MAX_THREAD/tile_width);
        S_ctailz.update().reorder(yi,t,yo).gpu_blocks(yo).gpu_threads(yi,t);

        S_intra.compute_at(S, Var("blockidx"));
        S_intra.reorder_storage(xi,yi,xo,yo);
        //S_intra.split(yi,t,yi, MAX_THREAD/tile_width).reorder(t,xi,yi,xo,yo).gpu_threads(xi,yi);
        S_intra.reorder(xi,yi,xo,yo).gpu_threads(yi);
        S_intra.update(0).reorder(rxi.x,yi,xo,yo).gpu_threads(yi);
        S_intra.update(1).reorder(ryi.x,xi,xo,yo).gpu_threads(xi);
        S_intra.update(2).reorder(rzi.x,yi,xo,yo).gpu_threads(yi);
        S_intra.update(3).reorder(rwi.x,xi,xo,yo).gpu_threads(xi);

        S.compute_root();
        S.split(x, xo,xi, tile_width).split(y, yo,yi, tile_width);
        //S.split(yi,t,yi, MAX_THREAD/tile_width).reorder(t,xi,yi,xo,yo);
        S.reorder(yi, xi, xo, yo);
        S.gpu_blocks(xo,yo).gpu_threads(xi);
        S.bound(x, 0, image.width()).bound(y, 0, image.height());
    }
    else {
        cerr << "Warning: No CPU scheduling" << endl;
    }

    // ----------------------------------------------------------------------------------------------

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
        Image<int> ref(width,height);

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

        int diff_sum = 0;
        int all_sum = 0;
        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {
                diff(x,y) = std::abs(ref(x,y) - hl_out(x,y));
                diff_sum += diff(x,y);
                all_sum += ref(x,y);
            }
        }
        float diff_ratio = 100.0f * float(diff_sum) / float(all_sum);

        if (verbose) {
            cerr << "Input" << endl << random_image << endl;
            cerr << "Reference" << endl << ref << endl;
            cerr << "Halide output" << endl << hl_out << endl;
            cerr << "Difference " << endl << diff << endl;
            cerr << "\nError = " << diff_sum << " ~ " << diff_ratio << "%" << endl;
        } else {
            cerr << "\nError = " << diff_sum << " ~ " << diff_ratio << "%" << endl;
            cerr << endl;
        }
    }

    return EXIT_SUCCESS;
}