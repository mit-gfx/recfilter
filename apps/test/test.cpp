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

    Arguments args("test", argc, argv);

    int   width   = args.width;
    int   height  = args.width;
    int   tile_width = args.block;

    Image<float> image = generate_random_image<float>(width,height);

    // ----------------------------------------------------------------------------------------------

    Func I("Input");
    Func S("S");

    Var x("x"), y("y");

    RDom rx(1, width-1, "rx");

    I(x,y) = select((x<0 || x>width-1 || y<0 || y>height-1), 0, image(clamp(x,0,width-1), clamp(y,0,height-1)));

    S(x,y) = I(x,y);
    S(rx,y) = S(rx,y) + S(rx-1,y);

    // ----------------------------------------------------------------------------------------------

    Var xi("xi");
    Var xo("xo");

    RDom rxi(1, tile_width-1,       "rxi");
    RDom rxo(1, width / tile_width, "rxo");

    split(S, 0, x, xi, xo, rx, rxi, rxo);

    float_dependencies_to_root(S);
    inline_function(S, "S$split$");
    inline_function(S, "S$split$$Deps_x$");

    // ----------------------------------------------------------------------------------------------

    vector<Func> func_list;
    extract_func_calls(S, func_list);

    map<string,Func> functions;
    for (size_t i=0; i<func_list.size(); i++) {
        cerr << func_list[i] << endl;
        functions[func_list[i].name()] = func_list[i];
    }

    Target target = get_jit_target_from_environment();
    if (target.has_gpu_feature() || (target.features & Target::GPUDebug)) {

        Func S_intra = functions["S$split$$Intra_x$"];

        Func SDebug("SDebug");
        SDebug(x,y) = S_intra(x%tile_width, x/tile_width, y);


        Var t("t"), yi("yi"), yo("yo");

        S_intra.compute_at(SDebug, Var("blockidx"));
        //S_intra.compute_root();
        S_intra.split(y,yo,yi,MAX_THREAD/tile_width).reorder(xi,yi,xo,yo).gpu_threads(xi,yi); //.gpu_blocks(xo,yo);
        S_intra.update(0).split(y,yi,yo,tile_width).reorder(rxi.x,yi,xo,yo).gpu_threads(yi);
        //S_intra.update(0).gpu_blocks(xo,yo);

        SDebug.compute_root();
        SDebug.split(x, xo,xi, tile_width).split(y, yo,yi, tile_width/2);
        SDebug.split(yi,t,yi, MAX_THREAD/tile_width).reorder(t,xi,yi,xo,yo);
        SDebug.gpu_blocks(xo,yo).gpu_threads(xi,yi);

        Image<float> out = SDebug.realize(width,height);
        Image<float> ref(width,height);
        Image<float> diff(width,height);
        int diff_sum = 0;
        for (int v=0; v<height; v++) {
            for (int u=0; u<width/tile_width; u++) {
                for (int i=0; i<tile_width; i++) {
                    ref(u*tile_width+i, v) = image(u*tile_width+i, v);
                }
            }
        }
        for (int v=0; v<height; v++) {
            for (int u=0; u<width/tile_width; u++) {
                for (int i=1; i<tile_width; i++) {
                    ref(u*tile_width+i, v) += ref(u*tile_width+i-1,v);
                }
            }
        }
        for (int v=0; v<height; v++) {
            for (int u=0; u<width; u++) {
                diff(u,v) = out(u,v)-ref(u,v);
                diff_sum += std::abs(diff(u,v));
            }
        }

        if (width < 256) {
            cerr << "Reference" << endl << ref << endl;
            cerr << "Out" << endl << out << endl;
            cerr << "Diff" << endl << diff << endl;
        }
        cerr << "Diff" << endl << diff_sum << endl;
    }

    return EXIT_SUCCESS;
}
