#include "recfilter.h"
#include "recfilter_internals.h"
#include "coefficients.h"
#include "modifiers.h"

using namespace Halide;

Image<float> matrix_B(
        Image<float> feedfwd_coeff,
        Image<float> feedback_coeff,
        int scan_id,
        int tile_width,
        bool clamp_border)
{
    int filter_order = feedback_coeff.height();

    float feedfwd = feedfwd_coeff(scan_id);

    std::vector<float> feedback(filter_order);
    for (int i=0; i<filter_order; i++) {
        feedback[i] = feedback_coeff(scan_id, i);
    }

    Image<float> C(tile_width, tile_width);

    // initialize
    for (int x=0; x<tile_width; x++) {
        for (int y=0; y<tile_width; y++) {
            C(x,y) = (x==y ? feedfwd : 0.0f);
        }
    }

    // update one row at a time from bottom to up
    for (int y=0; y<tile_width; y++) {
        for (int x=0; x<tile_width; x++) {
            for (int j=0; j<filter_order; j++) {
                float a = 0.0f;
                if (clamp_border) {
                    a = (y-j-1>=0 ? C(x,y-j-1)*feedback[j] : (x==0 ? feedback[j] : 0.0f));
                } else {
                    a = (y-j-1>=0 ? C(x,y-j-1)*feedback[j] : 0.0f);
                }
                C(x,y) += a;
            }
        }
    }

    return C;
}

Image<float> matrix_R(
        Image<float> feedback_coeff,
        int scan_id,
        int tile_width)
{
    int filter_order = feedback_coeff.height();

    std::vector<float> weights(filter_order);
    for (int i=0; i<filter_order; i++) {
        weights[i] = feedback_coeff(scan_id, i);
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

Image<float> matrix_transpose(Image<float> A) {
    Image<float> B(A.height(),A.width());
    for (int y=0; y<B.height(); y++) {
        for (int x=0; x<B.width(); x++) {
            B(x,y) = A(y,x);
        }
    }
    return B;
}

Image<float> matrix_mult(Image<float> A, Image<float> B) {
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

Image<float> matrix_antidiagonal(int size) {
    Image<float> C(size, size);

    for (int i=0; i<C.width(); i++) {
        for (int j=0; j<C.height(); j++) {
            C(i,j) = (i==size-1-j ? 1.0f : 0.0f);
        }
    }
    return C;
}
