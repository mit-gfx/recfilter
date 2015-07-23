#include <iostream>
#include <Halide.h>
#include <recfilter.h>

using namespace Halide;

int main(int argc, char **argv) {
    int width  = 20;
    int height = 20;
    int tile   = 4;

    Image<float> image = generate_random_image<float>(width,height);

    RecFilterDim x("x", width), y("y", height);

    RecFilter filter;
    filter(x,y) = image(x,y);

    filter.add_filter(+x, {1.0, 1.0});
    filter.add_filter(+y, {1.0, 1.0});

    filter.split(x, tile, y, tile);

    //filter.intra_schedule().compute_globally();
    //filter.inter_schedule().compute_globally();

    std::cerr << filter << std::endl;

    Realization out = filter.realize();
    Image<float> hl_out(out);
    std::cerr << hl_out << std::endl;

    return 0;
}
