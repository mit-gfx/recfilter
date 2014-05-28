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

template<typename T>
Image<T> reference_recursive_filter(Image<T> in, Image<T> weights);


int main(int argc, char **argv) {
    Arguments args("recursive_filter", argc, argv);

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

    int fx = 3;
    int fy = 3;

    Image<float> W(2,3);
    W(0,0) = 0.125f; // x dimension filtering weights
    W(0,1) = 0.0625f;
    W(0,2) = 0.03125f;
    W(1,0) = 0.125f; // y dimension filtering weights
    W(1,1) = 0.0625f;
    W(1,2) = 0.03125f;

    Func I("Input");
    Func S("S");

    Var x("x");
    Var y("y");

    RDom rx(0, image.width(), "rx");
    RDom ry(0, image.height(),"ry");

    I(x,y) = select((x<0 || y<0 || x>image.width()-1 || y>image.height()-1), 0.0f, image(clamp(x,0,image.width()-1),clamp(y,0,image.height()-1)));

    S(x, y) = I(x,y);
    S(rx,y) = S(rx,y) + select(rx>0, W(0,0)*S(rx-1,y), 0.0f) + select(rx>1, W(0,1)*S(rx-2,y), 0.0f) + select(rx>2, W(0,2)*S(rx-3,y), 0.0f);
    S(x,ry) = S(x,ry) + select(ry>0, W(1,0)*S(x,ry-1), 0.0f) + select(ry>1, W(1,1)*S(x,ry-2), 0.0f) + select(ry>2, W(1,2)*S(x,ry-3), 0.0f);

    // ----------------------------------------------------------------------------------------------

    Var xi("xi"), yi("yi");
    Var xo("xo"), yo("yo");

    RDom rxi(0, tile_width, "rxi");
    RDom ryi(0, tile_width, "ryi");

    split(S, W, Internal::vec(0,1),
            Internal::vec(x,y), Internal::vec(xi,yi), Internal::vec(xo,yo),
            Internal::vec(rx,ry), Internal::vec(rxi,ryi),
            Internal::vec(fx, fy));

    // ----------------------------------------------------------------------------------------------

    float_dependencies_to_root(S);
    inline_function(S, "S--Intra_y-Deps_x");
    inline_function(S, "S--Intra_y");
    inline_function(S, "S--Deps_y");
    swap_variables (S, "S--Intra_y-Tail_x", xi, yi);
    merge(S, "S--Intra_y-Tail_x", "S--Tail_y", "S--Tail");
    recompute(S, "S", "S--Intra_y-Intra_x");

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

    assert(S_intra0.defined());
    assert(S_tails.defined());
    assert(S_intra.defined());
    assert(S_ctailx.defined());
    assert(S_ctaily.defined());

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
        S_ctailx.reorder_storage(yi,yo,xi,xo);
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

        S_intra.compute_at(S, Var("blockidx"));
        S_intra.reorder_storage(xi,yi,xo,yo);
        //S_intra.split(yi,t,yi, MAX_THREAD/tile_width).reorder(t,xi,yi,xo,yo).gpu_threads(xi,yi);
        S_intra.reorder(xi,yi,xo,yo).gpu_threads(yi);
        S_intra.update(0).reorder(rxi.x,yi,xo,yo).gpu_threads(yi);
        S_intra.update(1).reorder(ryi.x,xi,xo,yo).gpu_threads(xi);

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

    //
    // JIT compilation
    //

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
        Image<float> diff(width,height);
        Image<float> ref = reference_recursive_filter<float>(random_image, W);

        float diff_sum = 0.0f;
        float all_sum = 0.0f;
        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {
                diff(x,y) = std::abs(ref(x,y) - hl_out(x,y));
                diff_sum += diff(x,y);
                all_sum += ref(x,y);
            }
        }
        float diff_ratio = 100.0f * diff_sum / all_sum;

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

template<typename T>
Image<T> reference_recursive_filter(Image<T> in, Image<T> weights)
{
    int width = in.width();
    int height= in.height();

    int order_x = weights.height();
    int order_y = weights.height();

    Halide::Image<T> ref(width,height);

    for (int y=0; y<height; y++) {          // init the solution
        for (int x=0; x<width; x++) {
            ref(x,y) = in(x,y);
        }
    }

    for (int y=0; y<height; y++) {          // x filtering
        for (int x=0; x<width; x++) {
            for (int k=1; k<=order_x; k++) {
                ref(x,y) += (x>=k ? weights(0,k-1)*ref(x-k,y) : T(0));
            }
        }
    }

    for (int y=0; y<height; y++) {          // y filtering
        for (int x=0; x<width; x++) {
            for (int k=1; k<=order_y; k++) {
                ref(x,y) += (y>=k ? weights(1,k-1)*ref(x,y-k) : T(0));
            }
        }
    }

    return ref;
}
