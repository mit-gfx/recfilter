#include <iostream>
#include <iomanip>
#include <Halide.h>

int main(int argc, char **argv) {
    int n = 4;
    int max_r = 5;

    Halide::Var i("i"), j("j"), r("r");

    Halide::RDom m(0, n, 0, n, 0, n, 1, max_r-1, "m");

    Halide::Func A("A");
    Halide::Func Ar("Ar");

    A(i,j) = j*n+i;

    Ar(i, j, r)    = select(r==0, A(i,j), 0);
    Ar(m.y, m.z, m.w) += Ar(m.y, m.x, m.w-1) * A(m.x, m.z);

    Ar.compute_root();
    Ar.bound(i,0,n).bound(j,0,n).bound(r,0,max_r);

    Ar.trace_realizations();

    Halide::Image<int> img = Ar.realize(n,n, max_r);

    // print the matrices
    for (size_t z=0; z<max_r; z++) {
        std::cout << "A^" << z+1 << std::endl;
        for (size_t y=0; y<n; y++) {
            for (size_t x=0; x<n; x++) {
                std::cout << std::setw(3) << img(x,y,z) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    return EXIT_SUCCESS;
}
