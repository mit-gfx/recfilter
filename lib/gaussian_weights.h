#ifndef _GAUSSIAN_WEIGHTS_H_
#define _GAUSSIAN_WEIGHTS_H_

#include <cmath>
#include <complex>
#include <iostream>
#include <Halide.h>

/**@name Compute the Gaussian, derivative of Gaussian and integral of Gaussian
 * @param[in] x input (double or Halide::Expr)
 * @param[in] mu mean of the Gaussian function
 * @param[in] sigma sigma support of the true Gaussian filter
 */
// @ {
double gaussian       (double x, double mu, double sigma);
double gaussDerivative(double x, double mu, double sigma);
double gaussIntegral  (double x, double mu, double sigma);

Halide::Expr gaussian       (Halide::Expr x, double mu, double sigma);
Halide::Expr gaussDerivative(Halide::Expr x, double mu, double sigma);
Halide::Expr gaussIntegral  (Halide::Expr x, double mu, double sigma);
// @}


/**
 * @brief Wrapper to compute third order recursive filter weights for Gaussian blur.
 * Third order filter can be approximated by cascaded first and second order filters
 * @param[in] order order of recursive filter for approximation
 * @param[in] sigma sigma of Gaussian filter
 *
 * @return vector with feedforward coeff as first element and rest feedback coeff
 */
std::vector<double> gaussian_weights(double sigma, int order);

/**
 * @brief Compute the size of a box filter that approximates a Gaussian
 *
 * Source: "Efficient Approximation of Gaussian Filters"
 * Rau and McClellan, IEEE Trans. on Signal Processing 1997
 * http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=554310
 *
 * @param[in] k number of repeated applications of box filter
 * @param[in] sigma sigma support of the true Gaussian filter
 * @return box filter width
 */
int gaussian_box_filter(int k, double sigma);


/** @brief Apply Gaussian filter on an input image
 *
 * @param[in] in input single channel image
 * @param[in] sigma sigma support of the true Gaussian filter
 * @return filtered image
 */
Halide::Image<double> reference_gaussian(Halide::Image<double> in, double sigma);


#endif // _GAUSSIAN_WEIGHTS_H_
