#include <iostream>
#include <Halide.h>

#include "recfilter.h"

using namespace Halide;

using std::cerr;
using std::endl;
using std::vector;

int main(int argc, char **argv) {
    int width  = 12;
    int height = 12;

    Image<float> image = generate_random_image<float>(width,height);

    vector<float> W = {1.0, 2.0f, 0.4f};

    RecFilterDim x("x", width), y("y", height);

    // two filters
    RecFilter f1("R1");
    f1(x,y) = image(x,y);
    f1.add_filter(+x, {1.0, 2.0f, -1.0f});
    f1.add_filter(+y, {1.0, 1.0f});

    RecFilter f2("R2");
    f2(x,y) = f1.as_func()(x,y);
    f2.add_filter(+x, {1.0, 1.0f});
    f2.add_filter(+y, {1.0, 2.0f, -1.0f});

    // overlapped version
    RecFilter f3 = f1.overlap_to_higher_order_filter(f2, "O");

    cerr << "Cascaded def" << f1 << f2 << endl;
    cerr << "Overlapped" << f3 << endl;

    cerr << "\nChecking difference ... " << endl;

    Image<float> ref(f2.realize());
    Image<float> out(f3.realize());

    cerr << CheckResultVerbose<float>(ref, out) << endl;

    return 0;
}
