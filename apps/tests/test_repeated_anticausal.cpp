#include <iostream>
#include <Halide.h>

#include "recfilter.h"

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

    Image<float> W(4,2);
    W(0,0) = 0.700f; W(0,1) = 0.500f;
    W(1,0) = 0.500f; W(1,1) = 0.500f;
    W(2,0) = 0.250f; W(2,1) = 0.125f;
    W(3,0) = 0.125f; W(3,1) = 0.0625f;

    RecFilterDim x("x", width), y("y", height);

    RecFilter filter;
    filter(x, y) = image(clamp(x,0,image.width()-1),clamp(y,0,image.height()-1));

    filter.add_filter(-x, {1.0f, W(0,0), W(0,1)});
    filter.add_filter(-x, {1.0f, W(1,0), W(1,1)});
    filter.add_filter(-x, {1.0f, W(2,0), W(2,1)});
    filter.add_filter(-x, {1.0f, W(3,0), W(3,1)});

    filter.split(x, tile);

    // ----------------------------------------------------------------------------------------------

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
    for (int y=height-1; y>=0; y--) {
        for (int x=width-1; x>=0; x--) {
            ref(x,y) +=
                (x<width-1 ? W(0,0)*ref(x+1,y) : 0.0f) +
                (x<width-2 ? W(0,1)*ref(x+2,y) : 0.0f);
        }
    }
    for (int y=height-1; y>=0; y--) {
        for (int x=width-1; x>=0; x--) {
            ref(x,y) +=
                (x<width-1 ? W(1,0)*ref(x+1,y) : 0.0f) +
                (x<width-2 ? W(1,1)*ref(x+2,y) : 0.0f);
        }
    }
    for (int y=height-1; y>=0; y--) {
        for (int x=width-1; x>=0; x--) {
            ref(x,y) +=
                (x<width-1 ? W(2,0)*ref(x+1,y) : 0.0f) +
                (x<width-2 ? W(2,1)*ref(x+2,y) : 0.0f);
        }
    }
    for (int y=height-1; y>=0; y--) {
        for (int x=width-1; x>=0; x--) {
            ref(x,y) +=
                (x<width-1 ? W(3,0)*ref(x+1,y) : 0.0f) +
                (x<width-2 ? W(3,1)*ref(x+2,y) : 0.0f);
        }
    }

    cerr << CheckResultVerbose<float>(ref, hl_out) << endl;

    return 0;
}
