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

    Image<float> W(7,2);
    W(0,0) = 0.5f; W(0,1) = 0.25f;
    W(1,0) = 0.5f; W(1,1) = 0.125f;
    W(2,0) = 0.5f; W(2,1) = 0.0625f;
    W(3,0) = 0.5f; W(3,1) = 0.125f;
    W(4,0) = 0.5f; W(4,1) = 0.250f;
    W(5,0) = 0.5f; W(5,1) = 0.0625f;
    W(6,0) = 0.5f; W(6,1) = 0.125f;

    Var x("x");
    Var y("y");

    RDom rx(0, image.width(), "rx");
    RDom ry(0, image.height(),"ry");

    RecFilter filter("S");
    filter.set_args(x, y, width, height);
    filter.define(image(clamp(x,0,image.width()-1),clamp(y,0,image.height()-1)));
    filter.add_filter(x, 1.0f, Internal::vec(W(0,0), W(0,1)), RecFilter::CAUSAL);
    filter.add_filter(x, 1.0f, Internal::vec(W(1,0), W(1,1)), RecFilter::ANTICAUSAL);
    filter.add_filter(x, 1.0f, Internal::vec(W(2,0), W(2,1)), RecFilter::CAUSAL);
    filter.add_filter(x, 1.0f, Internal::vec(W(3,0), W(3,1)), RecFilter::ANTICAUSAL);
    filter.add_filter(y, 1.0f, Internal::vec(W(4,0), W(4,1)), RecFilter::CAUSAL);
    filter.add_filter(y, 1.0f, Internal::vec(W(5,0), W(5,1)), RecFilter::ANTICAUSAL);
    filter.add_filter(y, 1.0f, Internal::vec(W(6,0), W(6,1)), RecFilter::ANTICAUSAL);

    filter.split(x, tile, y, tile);

    // ----------------------------------------------------------------------------------------------

    cerr << "\nGenerated Halide functions ... " << endl;
    cerr << filter << endl;

    Buffer out(type_of<float>(),width,height);
    filter.realize(out);

    // ----------------------------------------------------------------------------------------------

    cerr << "\nChecking difference ... " << endl;
    Image<float> hl_out(out);
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
