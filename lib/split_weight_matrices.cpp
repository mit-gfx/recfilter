#include "split_utils.h"

using namespace Halide;

static Image<float> matrix_B(
        Image<float> feedfwd_coeff,
        Image<float> feedback_coeff,
        int scan_id,
        int tile_width,
        bool clamp_border)
{
    int filter_order = feedback_coeff.height();

    float feedfwd = feedfwd_coeff(scan_id);

    vector<float> feedback(filter_order);
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

static Image<float> matrix_R(
        Image<float> feedback_coeff,
        int scan_id,
        int tile_width)
{
    int filter_order = feedback_coeff.height();

    vector<float> weights(filter_order);
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

static Image<float> matrix_transpose(Image<float> A) {
    Image<float> B(A.height(),A.width());
    for (int y=0; y<B.height(); y++) {
        for (int x=0; x<B.width(); x++) {
            B(x,y) = A(y,x);
        }
    }
    return B;
}

static Image<float> matrix_mult(Image<float> A, Image<float> B) {
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

static Image<float> matrix_antidiagonal(int size) {
    Image<float> C(size, size);

    for (int i=0; i<C.width(); i++) {
        for (int j=0; j<C.height(); j++) {
            C(i,j) = (i==size-1-j ? 1.0f : 0.0f);
        }
    }
    return C;
}


/** Weight coefficients (tail_size x tile_width) for
 * applying scan's corresponding to split indices split_id1 to
 * split_id2 in the SplitInfo struct.
 * It is meaningful to apply subsequent scans on the tail of any scan
 * as it undergoes other scans only if they happen after the first
 * scan. The SpliInfo object stores the scans in reverse order, hence indices
 * into the SplitInfo object split_id1 and split_id2 must be decreasing
 */
Image<float> tail_weights(SplitInfo s, int split_id1, int split_id2, bool clamp_border) {
    assert(split_id1 >= split_id2);

    const int* tile_width_ptr = as_const_int(s.tile_width);
    assert(tile_width_ptr &&
            "Could not convert tile width expression to integer");

    int  tile_width  = *tile_width_ptr;
    int  scan_id     = s.scan_id[split_id1];
    bool scan_causal = s.scan_causal[split_id1];

    Image<float> R = matrix_R(s.feedback_coeff, scan_id, tile_width);

    // accummulate weight coefficients because of all subsequent scans
    // traversal is backwards because SplitInfo contains scans in the
    // reverse order
    for (int j=split_id1-1; j>=split_id2; j--) {
        if (scan_causal != s.scan_causal[j]) {
            Image<float> B = matrix_B(s.feedfwd_coeff,
                    s.feedback_coeff, s.scan_id[j], tile_width, clamp_border);
            Image<float> I = matrix_antidiagonal(R.height());
            R = matrix_mult(I, R);
            R = matrix_mult(B, R);
            R = matrix_mult(I, R);
        }
        else {
            Image<float> B = matrix_B(s.feedfwd_coeff,
                    s.feedback_coeff, s.scan_id[j], tile_width, false);
            R = matrix_mult(B, R);
        }
    }

    return matrix_transpose(R);
}

/** Weight coefficients (tail_size x tile_width) for
 * applying scan's corresponding to split indices split_id1
 */
Image<float> tail_weights(SplitInfo s, int split_id1, bool clamp_border) {
    return tail_weights(s, split_id1, split_id1, clamp_border);
}
