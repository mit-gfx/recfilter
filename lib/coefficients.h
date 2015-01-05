#ifndef _COEFFICIENTS_H_
#define _COEFFICIENTS_H_

Halide::Image<float> matrix_B(
        Halide::Image<float> feedfwd_coeff,
        Halide::Image<float> feedback_coeff,
        int scan_id,
        int tile_width,
        bool clamp_border);

Halide::Image<float> matrix_R(
        Halide::Image<float> feedback_coeff,
        int scan_id,
        int tile_width);

Halide::Image<float> matrix_transpose(Halide::Image<float> A);
Halide::Image<float> matrix_mult(Halide::Image<float> A, Halide::Image<float> B);
Halide::Image<float> matrix_antidiagonal(int size);


#endif // _COEFFICIENTS_H_
