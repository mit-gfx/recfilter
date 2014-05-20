#include <iostream>
#include <Halide.h>

int main(int argc, char **argv) {
    Halide::Var i("i"), j("j"), r("r");

    Halide::Func A("A");

    A(i,j) = 1;

    return 0;
}
