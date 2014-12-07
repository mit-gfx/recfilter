#ifndef _COEFFICIENTS_H_
#define _COEFFICIENTS_H_

Halide::Image<double> matrix_B(
        Halide::Image<double> feedfwd_coeff,
        Halide::Image<double> feedback_coeff,
        int scan_id,
        int tile_width,
        bool clamp_border);

Halide::Image<double> matrix_R(
        Halide::Image<double> feedback_coeff,
        int scan_id,
        int tile_width);

Halide::Image<double> matrix_transpose(Halide::Image<double> A);
Halide::Image<double> matrix_mult(Halide::Image<double> A, Halide::Image<double> B);
Halide::Image<double> matrix_antidiagonal(int size);


#endif // _COEFFICIENTS_H_
