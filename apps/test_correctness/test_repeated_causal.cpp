#include <iostream>
#include <Halide.h>

#include "../../lib/split.h"

using namespace Halide;

using std::cerr;
using std::endl;

int main(int argc, char **argv) {
    int width  = 20;
    int height = 1;
    int tile   = 4;

    Image<float> random_image = generate_random_image<float>(width,height);

    ImageParam image(type_of<float>(), 2);
    image.set(random_image);

    // ----------------------------------------------------------------------------------------------

    int fx = 3;

    Image<float> W(4,3);
    W(0,0) = 1.00f; W(0,1) = 0.250f; W(0,2) = 0.0625f;
    W(1,0) = 0.75f; W(1,1) = 0.500f; W(1,2) = 0.0625f;
    W(2,0) = 0.50f; W(2,1) = 0.250f; W(2,2) = 0.0625f;
    W(3,0) = 0.25f; W(3,1) = 0.125f; W(3,2) = 0.0625f;

    Func S("S");

    Var x("x");
    Var y("y");

    RDom rx(0, image.width(),"rx");
    RDom ry(0, image.width(),"ry");
    RDom rz(0, image.width(),"rz");
    RDom rw(0, image.width(),"rw");

    S(x, y) = image(clamp(x,0,image.width()-1),clamp(y,0,image.height()-1));
    S(rx,y) = S(rx,y)
        + select(rx>0, W(0,0)*S(max(0,rx-1),y), 0.0f)
        + select(rx>1, W(0,1)*S(max(0,rx-2),y), 0.0f)
        + select(rx>2, W(0,2)*S(max(0,rx-3),y), 0.0f);

    S(ry,y) = S(ry,y)
        + select(ry>0, W(1,0)*S(max(0,ry-1),y), 0.0f)
        + select(ry>1, W(1,1)*S(max(0,ry-2),y), 0.0f)
        + select(ry>2, W(1,2)*S(max(0,ry-3),y), 0.0f);

    S(rz,y) = S(rz,y)
        + select(rz>0, W(2,0)*S(max(0,rz-1),y), 0.0f)
        + select(rz>1, W(2,1)*S(max(0,rz-2),y), 0.0f)
        + select(rz>2, W(2,2)*S(max(0,rz-3),y), 0.0f);

    S(rw,y) = S(rw,y)
        + select(rw>0, W(3,0)*S(max(0,rw-1),y), 0.0f)
        + select(rw>1, W(3,1)*S(max(0,rw-2),y), 0.0f)
        + select(rw>2, W(3,2)*S(max(0,rw-3),y), 0.0f);

    // ----------------------------------------------------------------------------------------------

    Var xi("xi");
    Var xo("xo");

    RDom rxi(0, tile, "rxi");
    RDom ryi(0, tile, "ryi");
    RDom rzi(0, tile, "rzi");
    RDom rwi(0, tile, "rwi");

    split(S,W,Internal::vec(  0,  0,  0,  0),
              Internal::vec(  x,  x,  x,  x),
              Internal::vec( xi, xi, xi, xi),
              Internal::vec( xo, xo, xo, xo),
              Internal::vec( rx, ry, rz, rw),
              Internal::vec(rxi,ryi,rzi,rwi),
              Internal::vec( fx, fx, fx, fx)
            );

    // ----------------------------------------------------------------------------------------------

    cerr << "\nGenerated Halide functions ... " << endl;
    std::vector<Func> func_list;
    extract_func_calls(S, func_list);
    for (size_t i=0; i<func_list.size(); i++) {
        func_list[i].compute_root();
        cerr << func_list[i] << endl;
    }

    Image<float> hl_out = S.realize(width,height);

    // ----------------------------------------------------------------------------------------------

    cerr << "\nChecking difference ... " << endl;
    Image<float> diff(width,height);
    Image<float> ref(width,height);

    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            ref(x,y) = random_image(x,y);
        }
    }
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            ref(x,y) +=
                (x>0 ? W(0,0)*ref(x-1,y) : 0.0f) +
                (x>1 ? W(0,1)*ref(x-2,y) : 0.0f) +
                (x>2 ? W(0,2)*ref(x-3,y) : 0.0f);
        }
    }
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            ref(x,y) +=
                (x>0 ? W(1,0)*ref(x-1,y) : 0.0f) +
                (x>1 ? W(1,1)*ref(x-2,y) : 0.0f) +
                (x>2 ? W(1,2)*ref(x-3,y) : 0.0f);
        }
    }
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            ref(x,y) +=
                (x>0 ? W(2,0)*ref(x-1,y) : 0.0f) +
                (x>1 ? W(2,1)*ref(x-2,y) : 0.0f) +
                (x>2 ? W(2,2)*ref(x-3,y) : 0.0f);
        }
    }
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            ref(x,y) +=
                (x>0 ? W(3,0)*ref(x-1,y) : 0.0f) +
                (x>1 ? W(3,1)*ref(x-2,y) : 0.0f) +
                (x>2 ? W(3,2)*ref(x-3,y) : 0.0f);
        }
    }

    cerr << CheckResultVerbose(ref, hl_out) << endl;

    return 0;
}
