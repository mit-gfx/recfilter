#include <iostream>
#include <Halide.h>

#include "../../lib/split.h"

using namespace Halide;

using std::cerr;
using std::endl;

int main(int argc, char **argv) {
    Arguments args("test_correctness", argc, argv);

    int width  = 20;
    int height = 1;
    int tile   = 4;

    Image<int> random_image = generate_random_image<int>(width,height);

    ImageParam image(type_of<int>(), 2);
    image.set(random_image);

    // ----------------------------------------------------------------------------------------------

    Func I("Input");
    Func S("S");

    Var x("x");
    Var y("y");

    RDom rx(0, image.width(),"rx");
    RDom ry(0, image.width(),"ry");
    RDom rz(0, image.width(),"rz");
    RDom rw(0, image.width(),"rw");

    I(x,y) = select((x<0 || y<0 || x>image.width()-1 || y>image.height()-1), 0, image(clamp(x,0,image.width()-1),clamp(y,0,image.height()-1)));

    S(x, y) = I(x,y);
    S(rx,y) = S(rx,y) + select(rx>0, S(max(0,rx-1),y), 0);
    S(ry,y) = S(ry,y) + select(ry>0, S(max(0,ry-1),y), 0);
    S(rz,y) = S(rz,y) + select(rz>0, S(max(0,rz-1),y), 0);
    S(rw,y) = S(rw,y) + select(rw>0, S(max(0,rw-1),y), 0);

    // ----------------------------------------------------------------------------------------------

    Var xi("xi"), yi("yi");
    Var xo("xo"), yo("yo");

    RDom rxi(0, tile, "rxi");
    RDom ryi(0, tile, "ryi");
    RDom rzi(0, tile, "rzi");
    RDom rwi(0, tile, "rwi");

    split(S, Internal::vec(0,0,0,0), Internal::vec(x,x,x,x), Internal::vec(xi,xi,xi,xi),
            Internal::vec(xo,xo,xo,xo), Internal::vec(rx,ry,rz,rw), Internal::vec(rxi,ryi,rzi,rwi));

    // ----------------------------------------------------------------------------------------------

    std::vector<Func> func_list;
    extract_func_calls(S, func_list);
    for (size_t i=0; i<func_list.size(); i++) {
        cerr << func_list[i] << endl;
        func_list[i].compute_root();
    }

    Image<int> hl_out = S.realize(width,height);

    // ----------------------------------------------------------------------------------------------

    cerr << "\nChecking difference ... " << endl;
    Image<int> diff(width,height);
    Image<int> ref(width,height);

    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            ref(x,y) = random_image(x,y);
        }
    }
    for (int y=0; y<height; y++) {
        for (int x=1; x<width; x++) {
            ref(x,y) += ref(x-1,y);
        }
    }
    for (int y=0; y<height; y++) {
        for (int x=1; x<width; x++) {
            ref(x,y) += ref(x-1,y);
        }
    }
    for (int y=0; y<height; y++) {
        for (int x=1; x<width; x++) {
            ref(x,y) += ref(x-1,y);
        }
    }
    for (int y=0; y<height; y++) {
        for (int x=1; x<width; x++) {
            ref(x,y) += ref(x-1,y);
        }
    }

    int diff_sum = 0;
    int all_sum = 0;
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            diff(x,y) = ref(x,y) - hl_out(x,y);
            diff_sum += std::abs(diff(x,y));
            all_sum += ref(x,y);
        }
    }
    float diff_ratio = 100.0f * float(diff_sum) / float(all_sum);

    cerr << "Reference" << endl << ref << endl;
    cerr << "Halide output" << endl << hl_out << endl;
    cerr << "Difference " << endl << diff << endl;
    cerr << "\nError = " << diff_sum << " ~ " << diff_ratio << "%" << endl;

    return 0;
}
