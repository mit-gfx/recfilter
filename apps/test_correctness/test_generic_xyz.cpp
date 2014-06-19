#include <iostream>
#include <Halide.h>

#include "../../lib/split.h"

using namespace Halide;

using std::cerr;
using std::endl;

int main(int argc, char **argv) {
    int width    = 16;
    int height   = 16;
    int channels = 16;
    int tile     = 4;

    Image<float> random_image = generate_random_image<float>(width,height,channels);

    ImageParam image(type_of<float>(), 3);
    image.set(random_image);

    // ----------------------------------------------------------------------------------------------

    int fx = 2;
    int fy = 2;
    int fz = 2;

    Image<float> W(6,2);
    W(0,0) = 0.5f; W(0,1) = 0.25f;
    W(1,0) = 0.5f; W(1,1) = 0.125f;
    W(2,0) = 0.5f; W(2,1) = 0.0625f;
    W(3,0) = 0.5f; W(3,1) = 0.125f;
    W(4,0) = 0.5f; W(4,1) = 0.250f;
    W(5,0) = 0.5f; W(5,1) = 0.0625f;

    Func S("S");

    Var x("x");
    Var y("y");
    Var z("z");

    RDom rxa(0, image.width(), "rxa");
    RDom rxb(0, image.width(), "rxb");
    RDom rya(0, image.height(),"rya");
    RDom ryb(0, image.height(),"ryb");
    RDom rza(0, image.channels(),"rza");
    RDom rzb(0, image.channels(),"rzb");

    Expr iw = image.width()-1;
    Expr ih = image.height()-1;
    Expr ic = image.channels()-1;

    S(x,y,z) = image(clamp(x,0,image.width()-1), clamp(y,0,image.height()-1), clamp(z,0,image.channels()-1));

    S(rxa,y,z) = S(rxa,y,z)
        + select(rxa>0, W(0,0)*S(max(0,rxa-1),y,z), 0.0f)
        + select(rxa>1, W(0,1)*S(max(0,rxa-2),y,z), 0.0f);

    S(iw-rxb,y,z) = S(iw-rxb,y,z)
        + select(rxb>0, W(1,0)*S(min(iw,iw-rxb+1),y,z), 0.0f)
        + select(rxb>1, W(1,1)*S(min(iw,iw-rxb+2),y,z), 0.0f);

    S(x,rya,z) = S(x,rya,z)
        + select(rya>0, W(2,0)*S(x,max(0,rya-1),z), 0.0f)
        + select(rya>1, W(2,1)*S(x,max(0,rya-2),z), 0.0f);

    S(x,ih-ryb,z) = S(x,ih-ryb,z)
        + select(ryb>0, W(3,0)*S(x,min(ih,ih-ryb+1),z), 0.0f)
        + select(ryb>1, W(3,1)*S(x,min(ih,ih-ryb+2),z), 0.0f);

    S(x,y,rza) = S(x,y,rza)
        + select(rza>0, W(4,0)*S(x,y,max(0,rza-1)), 0.0f)
        + select(rza>1, W(4,1)*S(x,y,max(0,rza-2)), 0.0f);

    S(x,y,ic-rzb) = S(x,y,ic-rzb)
        + select(rzb>0, W(5,0)*S(x,y,min(ic,ic-rzb+1)), 0.0f)
        + select(rzb>1, W(5,1)*S(x,y,min(ic,ic-rzb+2)), 0.0f);

    // ----------------------------------------------------------------------------------------------

    Var xi("xi"), yi("yi"), zi("zi");
    Var xo("xo"), yo("yo"), zo("zo");

    RDom rxai(0, tile, "rxai");
    RDom rxbi(0, tile, "rxbi");
    RDom ryai(0, tile, "ryai");
    RDom rybi(0, tile, "rybi");
    RDom rzai(0, tile, "rzai");
    RDom rzbi(0, tile, "rzbi");

    split(S,W,Internal::vec(   0,   0,   1,   1,   2,   2),
              Internal::vec(   x,   x,   y,   y,   z,   z),
              Internal::vec(  xi,  xi,  yi,  yi,  zi,  zi),
              Internal::vec(  xo,  xo,  yo,  yo,  zo,  zo),
              Internal::vec( rxa, rxb, rya, ryb, rza, rzb),
              Internal::vec(rxai,rxbi,ryai,rybi,rzai,rzbi),
              Internal::vec(  fx,  fx,  fy,  fy,  fz,  fz)
            );

    // ----------------------------------------------------------------------------------------------

    cerr << "\nGenerated Halide functions ... " << endl;
    std::vector<Func> func_list;
    extract_func_calls(S, func_list);
    for (size_t i=0; i<func_list.size(); i++) {
        func_list[i].compute_root();
        cerr << func_list[i] << endl;
    }

    Image<float> hl_out = S.realize(width,height,channels);

    // ----------------------------------------------------------------------------------------------

    cerr << "\nChecking difference ... " << endl;
    Image<float> diff(width,height,channels);
    Image<float> ref(width,height,channels);

    for (int z=0; z<channels; z++) {
        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {
                ref(x,y,z) = random_image(x,y,z);
            }
        }
    }
    for (int z=0; z<channels; z++) {
        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {
                ref(x,y,z) +=
                    (x>0 ? W(0,0)*ref(x-1,y,z) : 0.0f) +
                    (x>1 ? W(0,1)*ref(x-2,y,z) : 0.0f);
            }
        }
    }
    for (int z=0; z<channels; z++) {
        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {
                ref(width-1-x,y,z) +=
                    (x>0 ? W(1,0)*ref(width-1-x+1,y,z) : 0.0f) +
                    (x>1 ? W(1,1)*ref(width-1-x+2,y,z) : 0.0f);
            }
        }
    }
    for (int z=0; z<channels; z++) {
        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {
                ref(x,y,z) +=
                    (y>0 ? W(2,0)*ref(x,y-1,z) : 0.0f) +
                    (y>1 ? W(2,1)*ref(x,y-2,z) : 0.0f);
            }
        }
    }
    for (int z=0; z<channels; z++) {
        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {
                ref(x,height-1-y,z) +=
                    (y>0 ? W(3,0)*ref(x,height-1-y+1,z) : 0.0f) +
                    (y>1 ? W(3,1)*ref(x,height-1-y+2,z) : 0.0f);
            }
        }
    }
    for (int z=0; z<channels; z++) {
        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {
                ref(x,y,z) +=
                    (z>0 ? W(4,0)*ref(x,y,z-1) : 0.0f) +
                    (z>1 ? W(4,1)*ref(x,y,z-2) : 0.0f);
            }
        }
    }
    for (int z=0; z<channels; z++) {
        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {
                ref(x,y,channels-1-z) +=
                    (z>0 ? W(5,0)*ref(x,y,channels-1-z+1) : 0.0f) +
                    (z>1 ? W(5,1)*ref(x,y,channels-1-z+2) : 0.0f);
            }
        }
    }

    cerr << CheckResultVerbose(ref, hl_out) << endl;

    return 0;
}
