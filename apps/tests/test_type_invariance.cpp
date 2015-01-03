#include <iostream>
#include <Halide.h>

#include "recfilter.h"

using namespace Halide;

using std::cerr;
using std::endl;
using std::vector;

int main(int argc, char **argv) {
    int width  = 20;
    int height = 20;
    int tile   = 4;

    Image<int16_t> random_image = generate_random_image<int16_t>(width,height);
    ImageParam image(type_of<int16_t>(), 2);
    image.set(random_image);

    vector<double> W = {1.0, -1.0};

    RecFilterDim x("x", width), y("y", height);

    RecFilter filter;

    filter(x,y) = image(x,y);

    filter.add_filter(+x, {1.0, W[0], W[1]});
    filter.add_filter(+y, {1.0, W[0], W[1]});

    filter.split(x, tile, y, tile);

    // ----------------------------------------------------------------------------------------------

    Buffer out(type_of<int16_t>(),width,height);
    filter.realize(out);

    // ----------------------------------------------------------------------------------------------

    cerr << "\nChecking difference ... " << endl;
    Image<int16_t> hl_out(out);
    Image<int16_t> diff(width,height);
    Image<int16_t> ref(width,height);

    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            ref(x,y) = random_image(x,y);
        }
    }
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            ref(x,y) +=
                (x>0 ? int16_t(W[0])*ref(x-1,y) : 0) +
                (x>1 ? int16_t(W[1])*ref(x-2,y) : 0);
        }
    }
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            ref(x,y) +=
                (y>0 ? int16_t(W[0])*ref(x,y-1) : 0) +
                (y>1 ? int16_t(W[1])*ref(x,y-2) : 0);
        }
    }

    cerr << CheckResultVerbose<int16_t>(ref, hl_out) << endl;

    return 0;
}

