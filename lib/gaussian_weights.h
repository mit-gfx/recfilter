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
template <typename T>
Halide::Image<T> reference_gaussian(Halide::Image<T> in, T sigma) {
    int width = in.width();
    int height= in.height();
    Halide::Image<T> ref(width,height);
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            double a = 0.0;
            double w = 0.0;
            for (int j=0; j<height; j++) {
                for (int i=0; i<width; i++) {
                    double d = (x-i)*(x-i) + (y-j)*(y-j);
                    double g = gaussian(std::sqrt(d), 0.0, sigma);
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
