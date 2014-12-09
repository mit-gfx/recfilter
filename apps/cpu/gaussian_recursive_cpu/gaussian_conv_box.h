/**
 * \file gaussian_conv_box.h
 * \brief Box filtering approximation of Gaussian convolution
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
 * \defgroup box_gaussian Box filter Gaussian convolution
 * \brief A fast low-accuracy approximation of Gaussian convolution.
 *
 * This code implements the basic iterated box filter approximation of
 * Gaussian convolution as developed by Wells. This approach is based on the
 * efficient recursive implementation of the box filter as
 * \f[ H(z) = \frac{1}{2r + 1}\frac{z^r - z^{-r-1}}{1 - z^{-1}}, \f]
 * where r is the box radius.
 *
 * While box filtering is very efficient, it has the limitation that only a
 * quantized set of \f$ \sigma \f$ values can be approximated because the box
 * radius r is integer. \ref ebox_gaussian and \ref sii_gaussian are
 * extensions of box filtering that allow \f$ \sigma \f$ to vary continuously.
 *
 * \par Example
\code
    num *buffer;
    
    buffer = (num *)malloc(sizeof(num) * N);
    box_gaussian_conv(dest, buffer, src, N, stride, sigma, K);
    free(buffer);
\endcode
 *
 * \par Reference
 *  - W.M. Wells, "Efficient synthesis of Gaussian filters by cascaded
 *    uniform filters," IEEE Transactions on Pattern Analysis and Machine
 *    Intelligence, vol. 8, no. 2, pp. 234-239, 1986.
 *    http://dx.doi.org/10.1109/TPAMI.1986.4767776
 *
 * \{
 */
#ifndef _GAUSSIAN_CONV_BOX_H_
#define _GAUSSIAN_CONV_BOX_H_

#include "num.h"

void box_gaussian_conv(num *dest, num *buffer, const num *src,
    long N, long stride, num sigma, int K);
void box_gaussian_conv_image(num *dest, num *buffer, const num *src,
    int width, int height, int num_channels, num sigma, int K);

/** \} */
#endif /* _GAUSSIAN_CONV_BOX_H_ */
