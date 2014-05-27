#ifndef _SPLITLIB_MATRIX_UTILS_H_
#define _SPLITLIB_MATRIX_UTILS_H_

#include <Halide.h>

using namespace Halide;

namespace WeightMatrix {

Func add_matrix(Func A, Func B) {
    assert(A.defined());
    assert(B.defined());

    Var i, j, r;

    Func AB;
    AB(i, j, r) = A(i, j, r) + B(i, j, r);

    AB.compute_root();

    return AB;
}

Func mult_matrix(Func A, Func B, Expr tile_width, int filter_order) {
    assert(A.defined());
    assert(B.defined());
    assert(A.output_types()[0] == B.output_types()[0]);

    Var i, j, r;
    RDom m(0, filter_order, 0, filter_order, 0, filter_order);

    Type type = A.output_types()[0];

    Func AB;

    AB(i,   j,   r)  = Internal::Cast::make(type,0);
    AB(m.y, m.z, r) += A(m.y, m.x, r) * B(m.x, m.z, r);

    AB.compute_root();
    AB.bound(i,0,filter_order).bound(j,0,filter_order).bound(r,0,tile_width);

    return AB;
}

Func A(Func filter_weights, Expr tile_width, int scan_id, int filter_order) {
    assert(filter_weights.defined());

    Type type = filter_weights.output_types()[0];

    Var i, j;

    Func A;

    A(i,j) = Internal::Cast::make(type,0);
    A(i,j) = select(j==filter_order-1,
            filter_weights(scan_id, filter_order-1-i),
            select(i==j+1, 1, A(i,j)));

    A.compute_root();
    A.bound(i,0,filter_order).bound(j,0,filter_order);

    return A;
}

Func Ar(Func A, Expr tile_width, int filter_order) {
    assert(A.defined());

    Var i, j, r;
    RDom m(0, filter_order, 0, filter_order, 0, filter_order, 1, tile_width-1);

    Func Ar;

    // filter matrix to power r= A^r_F [Nehab et al 2011, appendix A]
    Ar(i, j, r) = select(r==0, A(i,j), 0);
    Ar(m.y, m.z, m.w) += Ar(m.y, m.x, m.w-1) * A(m.x, m.z);

    Ar.compute_root();
    Ar.bound(i,0,filter_order).bound(j,0,filter_order).bound(r,0,tile_width);

    return Ar;
}

Func Ar_sum_r(Func A, Expr tile_width, int filter_order) {
    assert(A.defined());

    Var i, j, r;
    RDom m(1, tile_width-1);

    Func As;

    As(i,j,r)  = A(i,j,r);
    As(i,j,m) += As(i,j,m-1);

    As.compute_root();
    As.bound(i,0,filter_order).bound(j,0,filter_order).bound(r,0,tile_width);

    return As;
}

Func matrix_last_row(Func A, Expr tile_width, int filter_order) {
    assert(A.defined());

    Var r, j;

    RDom c(0, filter_order);

    Func W("W_NO_REVEAL_");

    W(r,j) = A(filter_order-1-j, filter_order-1, r);

    W.compute_root();
    W.bound(r,0,tile_width).bound(j,0,filter_order);

    return W;
}

}

#endif // _SPLITLIB_MATRIX_UTILS_H_
