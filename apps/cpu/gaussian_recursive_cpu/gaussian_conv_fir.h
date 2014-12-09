/**
 * \file gaussian_conv_fir.h
 * \brief Gaussian convolution using FIR filters
 * \author Pascal Getreuer <getreuer@cmla.ens-cachan.fr>
 *
 * Copyright (c) 2012-2013, Pascal Getreuer
 * All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under, at your option, the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version, or the terms of the
 * simplified BSD license.
 *
 * You should have received a copy of these licenses along with this program.
 * If not, see <http://www.gnu.org/licenses/> and
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

/**
 * \defgroup fir_gaussian FIR Gaussian convolution
 * \brief Approximation by finite impulse response filtering.
 *
 * This code approximates Gaussian convolution with finite impulse response
 * (FIR) filtering. By truncating the Gaussian to a finite support, Gaussian
 * convolution is approximated by
 * \f[ H(z) = \tfrac{1}{s(r)}\sum_{n=-r}^r G_\sigma(n) z^{-n}, \quad
       s(r) = \sum_{n=-r}^{r} G_\sigma(n). \f]
 *
 * The process to use these functions is the following:
 *    -# fir_precomp() to precompute filter coefficients for the convolution
 *    -# fir_gaussian_conv() or fir_gaussian_conv_image() to perform
 *       the convolution itself (may be called multiple times if desired)
 *    -# fir_free() to clean up
 *
 * \par Example
\code
    fir_coeffs c;
    
    fir_precomp(&c, sigma, tol);
    fir_gaussian_conv(c, dest, src, N, stride);
    fir_free(&c);
\endcode
 *
 * \{
 */

#ifndef _GAUSSIAN_CONV_FIR_H_
#define _GAUSSIAN_CONV_FIR_H_

#include "num.h"

/** \brief Coefficients for FIR Gaussian approximation */
typedef struct fir_coeffs_
{
    num *g_trunc;   /**< FIR filter coefficients            */
    long radius;    /**< The radius of the filter's support */
} fir_coeffs;

int fir_precomp(fir_coeffs *c, double sigma, num tol);
void fir_gaussian_conv(fir_coeffs c, num *dest, const num *src,
    long N, long stride);
void fir_gaussian_conv_image(fir_coeffs c, num *dest, num *buffer,
    const num *src, int width, int height, int num_channels);
void fir_free(fir_coeffs *c);

/** \} */
#endif /* _GAUSSIAN_CONV_FIR_H_ */
