#include <iostream>
#include <Halide.h>

#include "../../lib/recfilter.h"

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

    Image<float> W(4,3);
    W(0,0) = 0.500f; W(0,1) = 0.250f; W(0,1) = 0.06250f;
    W(1,0) = 0.500f; W(1,1) = 0.250f; W(1,1) = 0.06250f;
    W(2,0) = 0.500f; W(2,1) = 0.250f; W(2,1) = 0.06250f;
    W(3,0) = 0.500f; W(3,1) = 0.250f; W(3,1) = 0.06250f;

    Var x("x"), y("y");
    RDom rx(0, image.width(),"rx");

    RecFilter filter("S");
    filter.set_args(x, y, width, height);
    filter.define(image(clamp(x,0,image.width()-1),clamp(y,0,image.height()-1)));
    filter.add_filter(x, 1.0f, Internal::vec(W(0,0), W(0,1), W(0,2)), RecFilter::CAUSAL);
    filter.add_filter(x, 1.0f, Internal::vec(W(1,0), W(1,1), W(1,2)), RecFilter::ANTICAUSAL);
    filter.add_filter(x, 1.0f, Internal::vec(W(2,0), W(2,1), W(2,2)), RecFilter::CAUSAL);
    filter.add_filter(x, 1.0f, Internal::vec(W(3,0), W(3,1), W(3,2)), RecFilter::ANTICAUSAL);

    filter.split(x, tile);

    // ----------------------------------------------------------------------------------------------

    cerr << filter << endl;

    Buffer out(type_of<float>(),width,height);
    filter.realize(out);

    // ----------------------------------------------------------------------------------------------

    cerr << "\nChecking difference ... " << endl;
    Image<float> hl_out(out);
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
                (x>1 ? W(0,1)*ref(x-2,y) : 0.0f) +
                (x>2 ? W(0,2)*ref(x-3,y) : 0.0f);
        }
    }
    for (int y=height-1; y>=0; y--) {
        for (int x=width-1; x>=0; x--) {
            ref(x,y) +=
                (x<width-1 ? W(1,0)*ref(x+1,y) : 0.0f) +
                (x<width-2 ? W(1,1)*ref(x+2,y) : 0.0f) +
                (x<width-3 ? W(1,2)*ref(x+3,y) : 0.0f);
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
    for (int y=height-1; y>=0; y--) {
        for (int x=width-1; x>=0; x--) {
            ref(x,y) +=
                (x<width-1 ? W(3,0)*ref(x+1,y) : 0.0f) +
                (x<width-2 ? W(3,1)*ref(x+2,y) : 0.0f) +
                (x<width-3 ? W(3,2)*ref(x+3,y) : 0.0f);
        }
    }

    cerr << CheckResultVerbose(ref, hl_out) << endl;

    return 0;
}
