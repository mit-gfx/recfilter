#include <iostream>
#include <cstdio>
#include <cmath>

#include <Halide.h>

#include "../../lib/split.h"

#define MAX_THREAD 192

using namespace Halide;

using std::vector;
using std::string;
using std::cerr;
using std::endl;

template<typename T>
Image<T> reference_recursive_filter(Image<T> in, Image<T> weights);


int main(int argc, char **argv) {
    Image<float> random_image = generate_random_image<float>(5,5);

    ImageParam image(type_of<float>(), 2);
    image.set(random_image);

    // ----------------------------------------------------------------------------------------------

    int filter_order_x = 3;
    int filter_order_y = 3;

    Image<float> weights(2,3);
    weights(0,0) = 0.125f; // x dimension filtering weights
    weights(0,1) = 0.0625f;
    weights(0,2) = 0.03125f;
    weights(1,0) = 0.125f; // y dimension filtering weights
    weights(1,1) = 0.0625f;
    weights(1,2) = 0.03125f;

    Func I("Input");
    Func W("Weight");
    Func S("S");

    Var x("x");
    Var y("y");

    RDom rx(1, image.width()-1, "rx");
    RDom ry(1, image.height()-1,"ry");
    RDom rz(1, image.width()-1, "rz");
    RDom rw(1, image.height()-1,"rw");

    I(x,y) = select((x<0 || y<0 || x>image.width()-1 || y>image.height()-1), 0.0f, image(clamp(x,0,image.width()-1),clamp(y,0,image.height()-1)));

    W(x, y) = weights(x,y);

    S(x, y) = I(x,y);
    S(rx,y) +=
        select(rx>0, W(0,0)*S(rx-1,y), 0.0f) +
        select(rx>1, W(0,1)*S(rx-2,y), 0.0f) +
        select(rx>2, W(0,2)*S(rx-3,y), 0.0f);

    S(x,ry) +=
        select(ry>0, W(1,0)*S(x,ry-1), 0.0f) +
        select(ry>1, W(1,1)*S(x,ry-2), 0.0f) +
        select(ry>2, W(1,2)*S(x,ry-3), 0.0f);

    S(image.width()-1-rz, y) +=
        select(rz<image.width()-1,   W(0,0)*S(image.width()-1-(rz-1),y), 0.0f) +
        select(rz<image.width()-1-1, W(0,1)*S(image.width()-1-(rz-2),y), 0.0f) +
        select(rz<image.width()-1-2, W(0,2)*S(image.width()-1-(rz-3),y), 0.0f);

    S(x,image.height()-1-rw) +=
        select(rw<image.height()-1,   W(1,0)*S(x,image.height()-1-(rw-1)), 0.0f) +
        select(rw<image.height()-1-1, W(1,1)*S(x,image.height()-1-(rw-2)), 0.0f) +
        select(rw<image.height()-1-2, W(1,2)*S(x,image.height()-1-(rw-3)), 0.0f);

    return EXIT_SUCCESS;
}
