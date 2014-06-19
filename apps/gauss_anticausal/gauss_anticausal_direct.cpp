#include <iostream>
#include <cstdio>
#include <cmath>

#include <Halide.h>

#include "../../lib/gaussian_weights.h"
#include "../../lib/split.h"

#define MAX_THREAD 192

using namespace Halide;

using std::vector;
using std::string;
using std::cerr;
using std::endl;

Image<float> reference_gaussian(Image<float> in, float sigma);

int main(int argc, char **argv) {

    return 0;
}


Image<float> reference_gaussian(Image<float> in, float sigma) {
    int width = in.width();
    int height= in.height();
    Image<float> ref(width,height);
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            float a = 0.0f;
            float w = 0.0f;
            for (int j=0; j<height; j++) {
                for (int i=0; i<width; i++) {
                    float d = (x-i)*(x-i) + (y-j)*(y-j);
                    float g = gaussian(std::sqrt(d), 0.0f, sigma);
                    a += g * in(i,j);
                    w += g;
                }
            }
            ref(x,y) = a/w;
        }
    }
    return ref;
}
