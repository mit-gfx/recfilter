#ifndef _GAUSSIAN_WEIGHTS_H_
#define _GAUSSIAN_WEIGHTS_H_

#include <cmath>
#include <complex>
#include <iostream>
#include <Halide.h>

/**
 * @brief Compute the Gaussian, derivative of Gaussian and integral of Gaussian
 * @param[in] x Input (float or Halide::Expr)
 * @param[in] mu Mean of the Gaussian function
 * @param[in] sigma Standard deviation of the Gaussian
 */
// @ {
//Expr gaussian(Expr x, float mu, float sigma);
//Expr gaussDerivative(Expr x, float mu, float sigma);
//Expr gaussIntegral(float x, float mu, float sigma);
float gaussian(float x, float mu, float sigma);
float gaussDerivative(float x, float mu, float sigma);
float gaussIntegral(float x, float mu, float sigma);
// @}


/**
 * @brief Wrapper to compute third order recursive filter weights for Gaussian blur.
 * Third order filter can be approximated by cascaded first and second order filters
 *
 * @return Images containing causal x, anticausal x,
 * causal y and anticausal y feedforward and feedback recursive filter weights
 */
std::pair<Halide::Image<float>, Halide::Image<float> > gaussian_weights(
        float sigma,   /// Gaussian sigma
        int order,     /// Recursive filter order for approximating Gaussian
        int num_scans  /// Number of scans in the recursive filter
        );

#endif // _GAUSSIAN_WEIGHTS_H_
