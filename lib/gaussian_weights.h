#ifndef _GAUSSIAN_WEIGHTS_H_
#define _GAUSSIAN_WEIGHTS_H_

#include <cmath>
#include <complex>
#include <iostream>
#include <Halide.h>

/**
 * Feed forward and feedback coefficients for computing n integrals of image,
 * single intergal is summed area table and multiple integrals are multiple
 * applications of summed area table
 *
 * \param[in] iterations number of integrals
 * \returns coeff of IIR filter that computes multiple integrals
 */
std::vector<float> integral_image_coeff(int iterations);

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
 * @param[in] order order of recursive filter for approximation
 * @param[in] sigma sigma of Gaussian filter
 *
 * @return vector with feedforward coeff as first element and rest feedback coeff
 */
std::vector<float> gaussian_weights(float sigma, int order);

/**
 * @brief Compute the size of a box filter that approximates a Gaussian
 *
 * Source: "Efficient Approximation of Gaussian Filters"
 * Rau and McClellan, IEEE Trans. on Signal Processing 1997
 * http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=554310
 *
 * @param[in] iterations number of repeated applications of box filter
 * @param[in] sigma sigma support of the true Gaussian filter
 * @return box filter width
 */
int gaussian_box_filter(int iterations, float sigma);


/** @brief Apply Gaussian filter on an input image
 *
 * @param[in] in input single channel image
 * @param[in] sigma sigma support of the true Gaussian filter
 * @return filtered image
 */
template <typename T>
Halide::Image<T> reference_gaussian(Halide::Image<T> in, T sigma) {
    int width = in.width();
    int height= in.height();
    Halide::Image<T> ref(width,height);
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            float a = 0.0;
            float w = 0.0;
            for (int j=0; j<height; j++) {
                for (int i=0; i<width; i++) {
                    float d = (x-i)*(x-i) + (y-j)*(y-j);
                    float g = gaussian(std::sqrt(d), 0.0, sigma);
                    a += g * in(i,j);
                    w += g;
                }
            }
            ref(x,y) = a/w;
        }
    }
    return ref;
}

#endif // _GAUSSIAN_WEIGHTS_H_
