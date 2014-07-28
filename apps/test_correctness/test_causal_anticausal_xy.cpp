#include <iostream>
#include <Halide.h>

#include "../../lib/recfilter.h"

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

    Image<float> W(4,3);
    W(0,0) = 0.5f; W(0,1) = 0.50f; W(0,2) = 0.125f;
    W(1,0) = 0.5f; W(1,1) = 0.25f; W(1,2) = 0.125f;
    W(2,0) = 0.5f; W(2,1) = 0.125f; W(2,2) = 0.0625f;
    W(3,0) = 0.5f; W(3,1) = 0.125f; W(3,2) = 0.03125f;

    Var x("x");
    Var y("y");

    RDom rx(0, image.width(),"rx");
    RDom ry(0, image.height(),"ry");

    RecFilter filter("S");
    filter.setArgs(x, y);
    filter.define(image(clamp(x,0,image.width()-1),clamp(y,0,image.height()-1)));
    filter.addScan(x, rx, Internal::vec(W(0,0), W(0,1), W(0,2)), RecFilter::CAUSAL);
    filter.addScan(x, rx, Internal::vec(W(1,0), W(1,1), W(1,2)), RecFilter::ANTICAUSAL);
    filter.addScan(y, ry, Internal::vec(W(2,0), W(2,1), W(2,2)), RecFilter::CAUSAL);
    filter.addScan(y, ry, Internal::vec(W(3,0), W(3,1), W(3,2)), RecFilter::ANTICAUSAL);

    filter.split(x, tile, y, tile);

    // ----------------------------------------------------------------------------------------------

    cerr << "\nGenerated Halide functions ... " << endl;
    map<string,Func> functions = filter.funcs();
    map<string,Func>::iterator f    = functions.begin();
    map<string,Func>::iterator fend = functions.end();
    for (; f!=fend; f++) {
        cerr << f->second << endl;
        f->second.compute_root();
    }

    Image<float> hl_out = filter.func().realize(width,height);

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
                (y>0 ? W(2,0)*ref(x,y-1) : 0.0f) +
                (y>1 ? W(2,1)*ref(x,y-2) : 0.0f) +
                (y>2 ? W(2,2)*ref(x,y-3) : 0.0f);
        }
    }
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            ref(width-1-x,y) +=
                (x>0 ? W(1,0)*ref(width-1-x+1,y) : 0.0f) +
                (x>1 ? W(1,1)*ref(width-1-x+2,y) : 0.0f) +
                (x>2 ? W(1,2)*ref(width-1-x+3,y) : 0.0f);
        }
    }
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            ref(x,height-1-y) +=
                (y>0 ? W(3,0)*ref(x,height-1-y+1) : 0.0f) +
                (y>1 ? W(3,1)*ref(x,height-1-y+2) : 0.0f) +
                (y>2 ? W(3,2)*ref(x,height-1-y+3) : 0.0f);
        }
    }

    cerr << CheckResultVerbose(ref, hl_out) << endl;

    return 0;
}
