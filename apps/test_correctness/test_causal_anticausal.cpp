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

    int fx = 2;

    Image<float> W(2,2);
    W(0,0) = 0.500f; W(0,1) = 0.250f;
    W(1,0) = 0.500f; W(1,1) = 0.250f;

    Func I("Input");
    Func S("S");

    Var x("x");
    Var y("y");

    RDom rx(0, image.width(),"rx");
    RDom ry(0, image.width(),"ry");

    Expr iw = image.width()-1;

    I(x,y) = select((x<0 || y<0 || x>image.width()-1 || y>image.height()-1), 0, image(clamp(x,0,image.width()-1),clamp(y,0,image.height()-1)));

    S(x, y)     = I(x,y);
    S(rx,y)    += select(rx>0, W(0,0)*S(max(0,rx-1),y), 0.0f)     + select(rx>1, W(0,1)*S(max(0,rx-2),y), 0.0f);
    S(iw-ry,y) += select(ry>0, W(1,0)*S(min(iw,iw-ry+1),y), 0.0f) + select(ry>1, W(1,1)*S(min(iw,iw-ry+2),y), 0.0f);

    // ----------------------------------------------------------------------------------------------

    Var xi("xi"), yi("yi");
    Var xo("xo"), yo("yo");

    RDom rxi(0, tile, "rxi");
    RDom ryi(0, tile, "ryi");

    split(S,W,Internal::vec(  0,  0),
              Internal::vec(  x,  x),
              Internal::vec( xi, xi),
              Internal::vec( xo, xo),
              Internal::vec( rx, ry),
              Internal::vec(rxi,ryi),
              Internal::vec( fx, fx)
            );

    // ----------------------------------------------------------------------------------------------

    std::vector<Func> func_list;
    extract_func_calls(S, func_list);
    for (size_t i=0; i<func_list.size(); i++) {
        func_list[i].compute_root();
    }

    Image<float> hl_out = S.realize(width,height);

    // ----------------------------------------------------------------------------------------------

    cerr << "\nChecking difference ... " << endl;
    Image<float> diff(width,height);
    Image<float> ref(width,height);

    for (int y=height-1; y>=0; y--) {
        for (int x=width-1; x>=0; x--) {
            ref(x,y) = random_image(x,y);
        }
    }
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            ref(x,y) +=
                (x>0 ? W(0,0)*ref(x-1,y) : 0.0f) +
                (x>1 ? W(0,1)*ref(x-2,y) : 0.0f);
        }
    }
    for (int y=height-1; y>=0; y--) {
        for (int x=width-1; x>=0; x--) {
            ref(x,y) +=
                (x<width-1 ? W(1,0)*ref(x+1,y) : 0.0f) +
                (x<width-2 ? W(1,1)*ref(x+2,y) : 0.0f);
        }
    }

    cerr << CheckResultVerbose(ref, hl_out) << endl;

    return 0;
}
