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

    RecFilterDim x("x", width), y("y", height);

    RecFilter f("R");
    f(x,y) = image(x,y);
    f.add_filter(+x, {1.0, 3.0f, -3.0f, 0.5f, 1.0f});
    f.add_filter(+y, {1.0, 3.0f, -3.0f, 0.5f, 1.0f});

    cerr << "Overlapped" << f << endl;

    // cascasded version
    vector<RecFilter> fc = f.cascade_by_order(2,2);

    cerr << "Cascaded def";
    for (int i=0; i<fc.size(); i++) {
        cerr << fc[i] << endl;
    }

    cerr << "\nChecking difference ... " << endl;

    Image<float> ref(f.realize());
    Image<float> out(fc[fc.size()-1].realize());

    cerr << CheckResultVerbose<float>(ref, out) << endl;

    return 0;
}

