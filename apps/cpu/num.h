/**
 * \file num.h
 * \brief num typedef
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
#ifndef _NUM_H_
#define _NUM_H_

/** \typedef num
 * We define a typedef "num" for the numeric datatype to use for computation.
 * If the preprocessing symbol `NUM_SINGLE` is defined, then `num` is defined
 * as a float (i.e., single precision). Otherwise, `num` is a double.
 */
#ifdef NUM_SINGLE
typedef float num;
#else
typedef double num;
#endif

/** \def FFT
 * For use with the FFTW libary, the macro `FFT(functionname)` is defined
 * such that it expands to
 *    `fftwf_functionname`  if #num is single,
 * or
 *    `fftw_functionname`   if #num is double.
 */
/** \brief Token-pasting macro */
#define _FFTW_CONCAT(A,B)    A ## B
#ifdef NUM_SINGLE
#define FFT(S)      _FFTW_CONCAT(fftwf_,S)
#else
#define FFT(S)      _FFTW_CONCAT(fftw_,S)
#endif

/** \def IMAGEIO_NUM
 * For use with imageio.c, define `IMAGEIO_NUM` to be either `IMAGEIO_SINGLE`
 * or `IMAGEIO_DOUBLE`, depending on whether `NUM_SINGLE` is defined.
 */
#ifdef NUM_SINGLE
#define IMAGEIO_NUM     IMAGEIO_SINGLE
#else
#define IMAGEIO_NUM     IMAGEIO_DOUBLE
#endif

#endif
