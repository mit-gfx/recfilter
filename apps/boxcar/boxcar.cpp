#include <iostream>
#include <Halide.h>

#include "recfilter.h"

#define MAX_THREAD        192
#define BOX_FILTER_FACTOR 16

using namespace Halide;

using std::vector;
using std::string;
using std::cerr;
using std::endl;


int main(int argc, char **argv) {
    Arguments args(argc, argv);

    bool  nocheck = args.nocheck;
    int   width   = args.width;
    int   height  = args.width;
    int   tile_width = args.block;
    int   iterations = args.iterations;

    Image<float> random_image = generate_random_image<float>(width,height);

    ImageParam image(type_of<float>(), 2);
    image.set(random_image);

    // ----------------------------------------------------------------------------------------------

    int box = BOX_FILTER_FACTOR;        // radius of box filter

    RecFilterDim x("x", width);
    RecFilterDim y("y", height);

    RecFilter filter;
    filter(x,y) = image(x,y);

    filter.add_filter(+x, {1.0, 1.0});
    filter.add_filter(+y, {1.0, 1.0});

    Func S = filter.func();

    Func B("Boxcar");
    B(x,y) = S(min(x+box,image.width()-1), min(y+box,image.height()-1))
           + select(x-box-1<0 || y-box-1<0, 0, S(max(x-box-1,0), max(y-box-1,0)))
           - select(y-box-1<0, 0, S(min(x+box,image.width()-1), max(y-box-1,0)))
           - select(x-box-1<0, 0, S(max(x-box-1,0), min(y+box,image.height()-1)));

    filter.split(x, tile_width, y, tile_width);

    cerr << filter << endl;

    // ----------------------------------------------------------------------------------------------

    {
    }

    Realization out = B.realize();

    // ----------------------------------------------------------------------------------------------

    if (!nocheck) {
        cerr << "\nChecking difference ... " << endl;
        Image<float> hl_out(out);
        Image<float> ref(width,height);

        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {
                ref(x,y) = 0;
                for (int v=y-box; v<=y+box; v++) {
                    for (int u=x-box; u<=x+box; u++) {
                        if (v>=0 && u>=0 && v<height && u<width)
                            ref(x,y) += random_image(u,v);
                    }
                }
            }
        }
        cerr << CheckResult<float>(ref,hl_out) << endl;
    }

    return 0;
}
