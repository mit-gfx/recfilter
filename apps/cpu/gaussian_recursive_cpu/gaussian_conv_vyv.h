/**
 * \file gaussian_conv_vyv.h
 * \brief Vliet-Young-Verbeek approximate Gaussian convolution
 * \author Pascal Getreuer <getreuer@cmla.ens-cachan.fr>
 *
 * Copyright (c) 2012, Pascal Getreuer
 * All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under, at your option, the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version, or the terms of the
 * simplified BSD license.
 *
 * You should have received a copy of these licenses along this program.
 * If not, see <http://www.gnu.org/licenses/> and
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

/**
 * \defgroup vyv_gaussian Vliet-Young-Verbeek Gaussian convolution
 * \brief An accurate recursive filter approximation, computed in-place.
 *
 * This code implements the recursive filter approximation of Gaussian
 * convolution proposed by Vliet, Young, and Verbeek. The Gaussian is
 * approximated by a cascade of a causal filter and an anticausal filter,
 * \f[ H(z) = G(z) G(z^{-1}), \quad
       G(z) = \frac{b_0}{1 + a_1 z^{-1} + \cdots + a_K z^{-K}}. \f]
 *
 * The process to use these functions is the following:
 *    -# vyv_precomp() to precompute coefficients for the convolution
 *    -# vyv_gaussian_conv() or vyv_gaussian_conv_image() to perform
 *       the convolution itself (may be called multiple times if desired)
 *
 * \par Example
\code
    vyv_coeffs c;
    
    vyv_precomp(&c, sigma, K, tol);
    vyv_gaussian_conv(c, dest, src, N, stride);
\endcode
 *
 * \note When the #num typedef is set to single-precision arithmetic,
 * vyv_gaussian_conv() may be inaccurate for large values of sigma.
 *
 * \par References
 *  - I.T. Young, L.J. van Vliet, "Recursive implementation of the Gaussian
 *    filter," Signal Processing, vol. 44, no. 2, pp. 139-151, 1995.
 *    http://dx.doi.org/10.1016/0165-1684(95)00020-E
 *  - L.J. van Vliet, I.T. Young, P.W. Verbeek, "Recursive Gaussian
 *    derivative filters," Proceedings of the 14th International Conference
 *    on Pattern Recognition, vol. 1, pp. 509-514, 1998.
 *    http://dx.doi.org/10.1109/ICPR.1998.711192
 *
 * \{
 */

#ifndef _GAUSSIAN_CONV_VYV_H_
#define _GAUSSIAN_CONV_VYV_H_

#include "num.h"

/** \brief Minimum valid VYV filter order. */
#define VYV_MIN_K       3
/** \brief Maximum valid VYV filter order. */
#define VYV_MAX_K       5
/** \brief Test whether a given K value is a valid VYV filter order. */
#define VYV_VALID_K(K)  (VYV_MIN_K <= (K) && (K) <= VYV_MAX_K)

/**
 * \brief Coefficients for Vliet-Young-Verbeek Gaussian approximation.
 *
 * The vyv_coeffs struct stores the coefficients for the recursive filter of
 * order K. This struct allows to precompute these filter coefficients
 * separately from actually performing the filtering so that filtering may be
 * performed multiple times using the same precomputed coefficients.
 *
 * This coefficients struct is precomputed by vyv_precomp() and then used
 * by vyv_gaussian_conv() or vyv_gaussian_conv_image().
 */
typedef struct vyv_coeffs_
{
    num filter[VYV_MAX_K + 1];     /**< Recursive filter coefficients       */
    num M[VYV_MAX_K * VYV_MAX_K];  /**< Matrix for handling right boundary  */
    num sigma;                     /**< Gaussian standard deviation         */
    num tol;                       /**< Boundary accuracy                   */
    int K;                         /**< Filter order                        */
    long max_iter;                 /**< Max iterations for left boundary    */
} vyv_coeffs;

void vyv_precomp(vyv_coeffs *c, double sigma, int K, num tol);
void vyv_gaussian_conv(vyv_coeffs c,
    num *dest, const num *src, long N, long stride);
void vyv_gaussian_conv_image(vyv_coeffs c, num *dest, const num *src,
    int width, int height, int num_channels);

/** \} */
#endif /* _GAUSSIAN_CONV_VYV_H_ */
