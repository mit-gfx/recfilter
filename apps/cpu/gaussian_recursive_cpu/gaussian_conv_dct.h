/**
 * \file gaussian_conv_dct.h
 * \brief Gaussian convolution via DCT
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
 * You should have received a copy of these licenses along this program.
 * If not, see <http://www.gnu.org/licenses/> and
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

/**
 * \defgroup dct_gaussian DCT-based Gaussian convolution
 * \brief Convolution via multiplication in the DCT domain.
 *
 * Via the convolution-multiplication property, discrete cosine transforms
 * (DCTs) are an effective way to implement Gaussian convolution. We follow
 * Martucci's use of DCTs to perform convolution with half-sample symmetric
 * boundary handling,
 * \f[ G_\sigma\!\stackrel{\text{sym}}{*}\!f=\mathcal{C}_\mathrm{2e}^{-1}\bigl
(\mathcal{C}_\mathrm{1e}(G_\sigma)\cdot\mathcal{C}_\mathrm{2e}(f)\bigr), \f]
 * where \f$ \mathcal{C}_\mathrm{1e} \f$ and \f$ \mathcal{C}_\mathrm{2e} \f$
 * denote respectively the DCT-I and DCT-II transforms of the same period
 * length. This DCT-based convolution is equivalent to (but is faster and
 * uses less memory) than FFT-based convolution with the symmetrized signal
 * \f$ (f_0,\ldots,f_{N-1},f_{N-1},\ldots,f_0) \f$.
 *
 * The FFTW library is used to compute the DCT transforms.
 *
 * The process to use these functions is the following:
 *    -# dct_precomp() or dct_precomp_image() to set up the convolution
 *    -# dct_gaussian_conv() to perform the convolution itself (it may
 *       be called multiple times if desired)
 *    -# dct_free() to clean up
 *
 * \par Example
\code
    dct_coeffs c;
    
    dct_precomp(&c, dest, src, N, stride, sigma);
    dct_gaussian_conv(c);
    dct_free(&c);
\endcode
 *
 * \par Reference
 *  - S. Martucci, "Symmetric convolution and the discrete sine and cosine
 *    transforms," IEEE Transactions on Signal Processing SP-42,
 *    pp. 1038-1051, 1994. http://dx.doi.org/10.1109/78.295213
 *
 * \{
 */

#ifndef _GAUSSIAN_CONV_DCT_H_
#define _GAUSSIAN_CONV_DCT_H_

#include <fftw3.h>
#include "num.h"

/** \brief FFTW plans and coefficients for DCT-based Gaussian convolution */
typedef struct dct_coeffs_
{
    FFT(plan) forward_plan;     /**< forward DCT plan   */
    FFT(plan) inverse_plan;     /**< inverse DCT plan   */
    num *dest;                  /**< destination array  */
    const num *src;             /**< source array       */
    
    enum
    {
        DCT_GAUSSIAN_1D,
        DCT_GAUSSIAN_IMAGE
    } conv_type;                /**< flag to switch between 1D vs. 2D  */
    union
    {
        struct
        {
            num alpha;          /**< exponent coefficient              */
            long N;             /**< signal length                     */
            long stride;        /**< stride between successive samples */
        } one;                  /**< 1D convolution parameters         */
        struct
        {
            num alpha_x;        /**< exponent coefficient              */
            num alpha_y;        /**< exponent coefficient              */
            int width;          /**< image width                       */
            int height;         /**< image height                      */
            int num_channels;   /**< number of image channels          */
        } image;                /**< 2D convolution parameters         */
    } dims;
} dct_coeffs;

int dct_precomp(dct_coeffs *c, num *dest, const num *src,
    long N, long stride, double sigma);
int dct_precomp_image(dct_coeffs *c, num *dest, const num *src,
    int width, int height, int num_channels, double sigma);
void dct_free(dct_coeffs *c);
void dct_gaussian_conv(dct_coeffs c);

/** \} */
#endif /* _GAUSSIAN_CONV_DCT_H_ */
