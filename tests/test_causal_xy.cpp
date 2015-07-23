#include <iostream>
#include <Halide.h>

#include "recfilter.h"

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

    RecFilterDim x("x", width);
    RecFilterDim y("y", height);

    RecFilter filter;

    filter(x,y) = image(x,y);

    filter.add_filter(+x, {1.0f, W(0,0), W(0,1), W(0,2)});
    filter.add_filter(+x, {1.0f, W(1,0), W(1,1), W(1,2)});
    filter.add_filter(+y, {1.0f, W(2,0), W(2,1), W(2,2)});
    filter.add_filter(+y, {1.0f, W(3,0), W(3,1), W(3,2)});

    filter.split(x, tile, y, tile);

    // ----------------------------------------------------------------------------------------------

    Realization out = filter.realize();

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
                (y>0 ? W(2,0)*ref(x,y-1) : 0.0f) +
                (y>1 ? W(2,1)*ref(x,y-2) : 0.0f) +
                (y>2 ? W(2,2)*ref(x,y-3) : 0.0f);
        }
    }
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            ref(x,y) +=
                (y>0 ? W(3,0)*ref(x,y-1) : 0.0f) +
                (y>1 ? W(3,1)*ref(x,y-2) : 0.0f) +
                (y>2 ? W(3,2)*ref(x,y-3) : 0.0f);
        }
    }

    cerr << CheckResultVerbose<float>(ref, hl_out) << endl;

    return 0;
}
