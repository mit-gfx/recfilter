/**
 * \file gaussian_conv_am.h
 * \brief Alvarez-Mazorra approximate Gaussian convolution
 * \author Pascal Getreuer <getreuer@cmla.ens-cachan.fr>
 *
 * Copyright (c) 2011-2013, Pascal Getreuer
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
 * \defgroup am_gaussian Alvarez-Mazorra Gaussian convolution
 * \brief A first-order recursive filter approximation, computed in-place.
 *
 * This code implements Alvarez and Mazorra's recursive filter approximation
 * of Gaussian convolution. The Gaussian is approximated by a cascade of
 * first-order recursive filters,
 * \f[ H(z) = \left(\nu/\lambda\right)^K
       \left( \frac{1}{1 - \nu z^{-1}} \frac{1}{1 - \nu z} \right)^K. \f]
 *
 * \par References
 *  - Alvarez, Mazorra, "Signal and Image Restoration using Shock Filters and
 *    Anisotropic Diffusion," SIAM J. on Numerical Analysis, vol. 31, no. 2,
 *    pp. 590-605, 1994. http://www.jstor.org/stable/2158018
 *
 * \{
 */
#ifndef _GAUSSIAN_CONV_AM_H_
#define _GAUSSIAN_CONV_AM_H_

#include "num.h"

void am_gaussian_conv(num *dest, const num *src, long N, long stride,
    double sigma, int K, num tol, int use_adjusted_q);

void am_gaussian_conv_image(num *dest, const num *src,
    int width, int height, int num_channels,
    num sigma, int K, num tol, int use_adjusted_q);

/** \} */
#endif /* _GAUSSIAN_CONV_AM_H_ */
