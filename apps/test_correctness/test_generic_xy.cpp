#include <iostream>
#include <Halide.h>

#include "../../lib/split.h"

using namespace Halide;

using std::cerr;
using std::endl;

int main(int argc, char **argv) {
    int width  = 16;
    int height = 16;
    int tile   = 4;

    Image<float> random_image = generate_random_image<float>(width,height);

    ImageParam image(type_of<float>(), 2);
    image.set(random_image);

    // ----------------------------------------------------------------------------------------------

    int fx = 2;
    int fy = 2;

    Image<float> W(7,2);
    W(0,0) = 0.5f; W(0,1) = 0.25f;
    W(1,0) = 0.5f; W(1,1) = 0.125f;
    W(2,0) = 0.5f; W(2,1) = 0.0625f;
    W(3,0) = 0.5f; W(3,1) = 0.125f;
    W(4,0) = 0.5f; W(4,1) = 0.250f;
    W(5,0) = 0.5f; W(5,1) = 0.0625f;
    W(6,0) = 0.5f; W(6,1) = 0.125f;

    Func I("Input");
    Func S("S");

    Var x("x");
    Var y("y");

    RDom rxa(0, image.width(), "rxa");
    RDom rxb(0, image.width(), "rxb");
    RDom rxc(0, image.width(), "rxc");
    RDom rxd(0, image.width(), "rxd");
    RDom rya(0, image.height(),"rya");
    RDom ryb(0, image.height(),"ryb");
    RDom ryc(0, image.height(),"ryc");

    I(x,y) = select((x<0 || y<0 || x>image.width()-1 || y>image.height()-1), 0, image(clamp(x,0,image.width()-1),clamp(y,0,image.height()-1)));

    Expr iw = image.width()-1;
    Expr ih = image.height()-1;

    S(x, y) = I(x,y);

    S(rxa,y) = S(rxa,y)
        + select(rxa>0, W(0,0)*S(max(0,rxa-1),y), 0.0f)
        + select(rxa>1, W(0,1)*S(max(0,rxa-2),y), 0.0f);

    S(iw-rxb,y) = S(iw-rxb,y)
        + select(rxb>0, W(1,0)*S(min(iw,iw-rxb+1),y), 0.0f)
        + select(rxb>1, W(1,1)*S(min(iw,iw-rxb+2),y), 0.0f);

    S(rxc,y) = S(rxc,y)
        + select(rxc>0, W(2,0)*S(max(0,rxc-1),y), 0.0f)
        + select(rxc>1, W(2,1)*S(max(0,rxc-2),y), 0.0f);

    S(iw-rxd,y) = S(iw-rxd,y)
        + select(rxd>0, W(3,0)*S(min(iw,iw-rxd+1),y), 0.0f)
        + select(rxd>1, W(3,1)*S(min(iw,iw-rxd+2),y), 0.0f);

    S(x,rya) = S(x,rya)
        + select(rya>0, W(4,0)*S(x,max(0,rya-1)), 0.0f)
        + select(rya>1, W(4,1)*S(x,max(0,rya-2)), 0.0f);

    S(x,ih-ryb) = S(x,ih-ryb)
        + select(ryb>0, W(5,0)*S(x,min(ih,ih-ryb+1)), 0.0f)
        + select(ryb>1, W(5,1)*S(x,min(ih,ih-ryb+2)), 0.0f);

    S(x,ih-ryc) = S(x,ih-ryc)
        + select(ryc>0, W(6,0)*S(x,min(ih,ih-ryc+1)), 0.0f)
        + select(ryc>1, W(6,1)*S(x,min(ih,ih-ryc+2)), 0.0f);

    // ----------------------------------------------------------------------------------------------

    Var xi("xi"), yi("yi");
    Var xo("xo"), yo("yo");

    RDom rxai(0, tile, "rxai");
    RDom rxbi(0, tile, "rxbi");
    RDom rxci(0, tile, "rxci");
    RDom rxdi(0, tile, "rxdi");
    RDom ryai(0, tile, "ryai");
    RDom rybi(0, tile, "rybi");
    RDom ryci(0, tile, "ryci");

    split(S,W,Internal::vec(   0,   0,   0,   0,   1,   1,   1),
              Internal::vec(   x,   x,   x,   x,   y,   y,   y),
              Internal::vec(  xi,  xi,  xi,  xi,  yi,  yi,  yi),
              Internal::vec(  xo,  xo,  xo,  xo,  yo,  yo,  yo),
              Internal::vec( rxa, rxb, rxc, rxd, rya, ryb, ryc),
              Internal::vec(rxai,rxbi,rxci,rxdi,ryai,rybi,ryci),
              Internal::vec(  fx,  fx,  fx,  fx,  fy,  fy,  fy)
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
                (x>1 ? W(0,1)*ref(x-2,y) : 0.0f);
        }
    }
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            ref(width-1-x,y) +=
                (x>0 ? W(1,0)*ref(width-1-x+1,y) : 0.0f) +
                (x>1 ? W(1,1)*ref(width-1-x+2,y) : 0.0f);
        }
    }
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            ref(x,y) +=
                (x>0 ? W(2,0)*ref(x-1,y) : 0.0f) +
                (x>1 ? W(2,1)*ref(x-2,y) : 0.0f);
        }
    }
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            ref(width-1-x,y) +=
                (x>0 ? W(3,0)*ref(width-1-x+1,y) : 0.0f) +
                (x>1 ? W(3,1)*ref(width-1-x+2,y) : 0.0f);
        }
    }
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            ref(x,y) +=
                (y>0 ? W(4,0)*ref(x,y-1) : 0.0f) +
                (y>1 ? W(4,1)*ref(x,y-2) : 0.0f);
        }
    }
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            ref(x,height-1-y) +=
                (y>0 ? W(5,0)*ref(x,height-1-y+1) : 0.0f) +
                (y>1 ? W(5,1)*ref(x,height-1-y+2) : 0.0f);
        }
    }
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            ref(x,height-1-y) +=
                (y>0 ? W(6,0)*ref(x,height-1-y+1) : 0.0f) +
                (y>1 ? W(6,1)*ref(x,height-1-y+2) : 0.0f);
        }
    }

    cerr << CheckResultVerbose(ref, hl_out) << endl;

    return 0;
}
