#include "split_utils.h"

using namespace Halide;


Image<float> weight_matrix_A_FB(Image<float> filter_weights,
        int scan_id, int tile_width)
{
    int filter_order = filter_weights.height();

    vector<float> weights(filter_order);
    for (int i=0; i<filter_order; i++) {
        weights[i] = filter_weights(scan_id, i);
    }

    Image<float> C(tile_width, tile_width);

    // initialize as identity matrix
    for (int x=0; x<tile_width; x++) {
        for (int y=0; y<tile_width; y++) {
            C(x,y) = (x==y ? 1.0f : 0.0f);
        }
    }

    // update one row at a time from bottom to up
    for (int y=0; y<tile_width; y++) {
        for (int x=0; x<tile_width; x++) {
            for (int j=0; y-j-1>=0 && j<filter_order; j++) {
                C(x,y) += C(x,y-j-1) * weights[j];
            }
        }
    }

    return C;
}

Image<float> weight_matrix_A_FP(Image<float> filter_weights,
        int scan_id, int tile_width)
{
    int filter_order = filter_weights.height();

    vector<float> weights(filter_order);
    for (int i=0; i<filter_order; i++) {
        weights[i] = filter_weights(scan_id, i);
    }

    Image<float> C(filter_order, tile_width);

    for (int x=0; x<filter_order; x++) {
        for (int y=0; y<tile_width; y++) {
            C(x,y) = 0.0f;
        }
    }

    for (int y=0; y<tile_width; y++) {
        for (int x=0; x<filter_order; x++) {
            if (y<filter_order) {
               C(x,y) = (x+y<filter_order ? weights[x+y] : 0.0f);
            }
            for (int j=0; y-j-1>=0 && j<filter_order; j++) {
                C(x,y) += C(x,y-j-1) * weights[j];
            }
        }
    }

    return C;
}

Image<float> weight_matrix_transpose(Image<float> A) {
    Image<float> B(A.height(),A.width());
    for (int y=0; y<B.height(); y++) {
        for (int x=0; x<B.width(); x++) {
            B(x,y) = A(y,x);
        }
    }
    return B;
}

Image<float> weight_matrix_mult(Image<float> A, Image<float> B) {
    assert(A.width() == B.height());

    int num_rows = A.height();
    int num_cols = B.width();
    int num_common = A.width();

    Image<float> C(num_cols, num_rows);

    for (int i=0; i<C.width(); i++) {
        for (int j=0; j<C.height(); j++) {
            C(i,j) = 0.0f;
        }
    }
    for (int i=0; i<C.height(); i++) {
        for (int j=0; j<C.width(); j++) {
            for (int k=0; k<num_common; k++) {
                C(j,i) += A(k,i) * B(j,k);
            }
        }
    }
    return C;
}
