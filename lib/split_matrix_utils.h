#ifndef _SPLITLIB_WEIGHT_MATRICES_H_
#define _SPLITLIB_WEIGHT_MATRICES_H_

#include <Halide.h>

using namespace Halide;


namespace WeightMatrix {

Image<float> mult_matrix(Image<float> A, Image<float> B) {
    assert(A.width() == B.height());

    int num_rows = A.height();
    int num_cols = B.width();
    int num_common = A.width();

    Image<float> C(num_cols, num_rows);

    for (size_t i=0; i<C.width(); i++) {
        for (size_t j=0; j<C.height(); j++) {
            C(i,j) = 0.0f;
        }
    }
    for (size_t i=0; i<C.height(); i++) {
        for (size_t j=0; j<C.width(); j++) {
            for (size_t k=0; k<num_common; k++) {
                C(j,i) += A(k,i) * B(j,k);
            }
        }
    }
    return C;
}

Image<float> matrix_pow(Image<float> A, int power) {
    assert(power > 0);

    Image<float> C(A.width(), A.height());

    for (size_t i=0; i<A.width(); i++) {
        for (size_t j=0; j<A.height(); j++) {
            C(i,j) = A(i,j);
        }
    }

    for (size_t i=1; i<power; i++) {
        C = mult_matrix(C,A);
    }
    return C;
}

/** Computes the filter matrix similar to A_F Nehab et al for a particular scan.
 *  Input: Matrix filter_weights (scan_id x filter_order)
 *  Output:
 *   1     1       0       0     ...  0
 *   1     0       1       0     ...  0
 *   1     0       0       1     ...  0
 *   1     0       0       0     ...  .
 *   .     .       .       .     ...  .
 *   1     0       0       0          1
 *  w_k  w_{k-1} w_{k-1} w_{k-2} ... w_{0}
 *
 *  where k = filter_order and w_i = filter_weight(scan_id, i)
 */
Image<float> A(Image<float> filter_weights, int scan_id) {
    int filter_order = filter_weights.height();

    Image<float> w_matrix(filter_order, filter_order);

    for (size_t j=0; j<filter_order; j++) {
        for (size_t i=0; i<filter_order; i++) {
            if (j==i-1) {
                w_matrix(i,j) = 1.0f;
            } else {
                w_matrix(i,j) = 0.0f;
            }
        }
    }
    for (size_t i=0; i<filter_order; i++) {
        w_matrix(i,filter_order-1) = filter_weights(scan_id, filter_order-1-i);
    }
    return w_matrix;
}

/** Compute all powers upto tile_width of a matrix.
 * Input matrix: A (filter_order x filter_order)
 * Output matrix: Ar (filter_order x filter_order x tile_width)
 * such that Ar(:,:,i) = A(:,:)^i
 */
Image<float> Ar(Image<float> A, int tile_width) {
    assert(A.width() == A.height());

    int filter_order = A.width();

    Image<float> Ar(filter_order, filter_order, tile_width);

    for (size_t z=0; z<tile_width; z++) {
        Image<float> C = matrix_pow(A, z+1);
        for (size_t y=0; y<filter_order; y++) {
            for (size_t x=0; x<filter_order; x++) {
                Ar(x,y,z) = C(x,y);
            }
        }
    }
    return Ar;
}

/** Accumulate weight coefficients from other scans.
 *  Input matrix:
 *    A (filter_order x filter_order x tile_width)
 *    B (filter_order x filter_order)
 *  Output matrix
 *    C (filter_order x filter_order x tile_width)
 *  such that
 *    C(:,:,i) = \sum_{k=0}^{i} { A(:,:,k)*B(:,:)^(i-k) }
 */
Image<float> accumulate_weights(Image<float> A, Image<float> B) {
    assert(A.width() == A.height());
    assert(B.width() == B.height());
    assert(A.width() == B.height());

    int filter_order = A.width();
    int tile_width   = A.channels();

    Image<float> C(filter_order, filter_order, tile_width);

    for (size_t z=1; z<tile_width; z++) {
        for (size_t y=0; y<filter_order; y++) {
            for (size_t x=0; x<filter_order; x++) {
                C(x,y,z) = 0.0f;
            }
        }
    }
    for (size_t y=0; y<filter_order; y++) {
        for (size_t x=0; x<filter_order; x++) {
            C(x,y,0) = A(x,y,0);
        }
    }
    for (size_t z=1; z<tile_width; z++) {
        Image<float> Ck(filter_order, filter_order);
        for (size_t y=0; y<filter_order; y++) {
            for (size_t x=0; x<filter_order; x++) {
                Ck(x,y) = C(x,y,z-1);
            }
        }
        Image<float> Z = mult_matrix(B,Ck);
        for (size_t y=0; y<filter_order; y++) {
            for (size_t x=0; x<filter_order; x++) {
                C(x,y,z) = Z(x,y) + A(x,y,z);
            }
        }
    }
    return C;
}

/** Extract the last row of all matrices in a series of 2D matrices.
 *
 * Input matrix:  A (filter_order x filter_order x tile_width)
 * Output matrix: B (tile_width x filter_order)
 * such that
 * B(r) is the last row of the r-th matrix of A, i.e. A(:,:,r)
 */
Image<float> matrix_last_row(Image<float> A) {
    assert(A.width() == A.height());

    int filter_order = A.width();
    int tile_width   = A.channels();

    Image<float> W(tile_width, filter_order);

    for (size_t r=0; r<tile_width; r++) {
        for (size_t j=0; j<filter_order; j++) {
            W(r,j) = A(filter_order-1-j, filter_order-1, r);
        }
    }
    return W;
}

}

#endif // _SPLITLIB_WEIGHT_MATRICES_H_
