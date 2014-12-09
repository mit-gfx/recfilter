/**
 * \file strategy_gaussian_conv.h
 * \brief Suite of different 1D Gaussian convolution methods
 * \author Pascal Getreuer <getreuer@cmla.ens-cachan.fr>
 *
 * Copyright (c) 2013, Pascal Getreuer
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

#ifndef _STRATEGY_GAUSSIAN_CONV_H_
#define _STRATEGY_GAUSSIAN_CONV_H_

#include "num.h"

/**
 * \brief Plan to perform a 1D Gaussian convolution for any algorithm
 *
 * The gconv* object represents a generic plan to perform a 1D Gaussian
 * convolution using any of the implemented algorithms. It is an application
 * of the strategy design pattern to allow the different Gaussian convolution
 * algorithms to be easily interchangeable in the benchmarking code.
 *
 * Use is similar to FFTW's fftw_plan. A gconv* object is first created by
 * gconv_plan(), then the 1D Gaussian convolution is performed by
 * gconv_execute(), and finally gconv_free() cleans up. The same Gaussian
 * convolution may be executed multiple times by multiple calls to
 * gconv_execute(). This interface allows to separate precomputation and
 * actual execution in timing tests.
 *
 * Most of the Gaussian convolution algorithm implementations follow this
 * structure of plan, execute, free, but there are variations in the
 * number of algorithm parameters and whether additional workspace memory
 * is required. The implementation of gconv* is simply a set of wrapper
 * functions to unify all algorithms to the same interface.
 *
 * \par Example
\code
    gconv *g = gconv_plan(dest, src, N, stride,
                          algorithm_name, sigma, K, tol);
    gconv_execute(g);
    gconv_free(g);
\endcode
 */
typedef struct gconv_ gconv;

gconv* gconv_plan(num *dest, const num *src, long N, long stride,
    const char *algo, double sigma, int K, num tol);
gconv* gconv_plan_2D(num *dest, const num *src, long N, long stride,
    const char *algo, double sigma, int K, num tol);
void gconv_execute(gconv *g);
void gconv_execute_2D(gconv *g);
void gconv_free(gconv *g);

#endif /* _STRATEGY_GAUSSIAN_CONV_H_ */
