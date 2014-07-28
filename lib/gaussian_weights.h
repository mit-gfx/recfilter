#ifndef _GAUSSIAN_WEIGHTS_H_
#define _GAUSSIAN_WEIGHTS_H_

#include <cmath>
#include <complex>
#include <iostream>
#include <Halide.h>

/**@name Compute the Gaussian, derivative of Gaussian and integral of Gaussian
 * @param[in] x input (float or Halide::Expr)
 * @param[in] mu mean of the Gaussian function
 * @param[in] sigma sigma support of the true Gaussian filter
 */
// @ {
float gaussian       (float x, float mu, float sigma);
float gaussDerivative(float x, float mu, float sigma);
float gaussIntegral  (float x, float mu, float sigma);

Halide::Expr gaussian       (Halide::Expr x, float mu, float sigma);
Halide::Expr gaussDerivative(Halide::Expr x, float mu, float sigma);
Halide::Expr gaussIntegral  (Halide::Expr x, float mu, float sigma);
// @}


/**
 * @brief Wrapper to compute third order recursive filter weights for Gaussian blur.
 * Third order filter can be approximated by cascaded first and second order filters
 *
 * @return feedforward coeff and vector of feedback coeff
 */
std::pair<float, vector<float> >
gaussian_weights(
        float sigma,   ///< Gaussian sigma
        int order      ///< recursive filter order for approximating Gaussian
        );

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
int gaussian_box_filter(int k, float sigma);


/** @brief Apply Gaussian filter on an input image
 *
 * @param[in] in input single channel image
 * @param[in] sigma sigma support of the true Gaussian filter
 * @return filtered image
 */
Halide::Image<float> reference_gaussian(Halide::Image<float> in, float sigma);


#endif // _GAUSSIAN_WEIGHTS_H_
