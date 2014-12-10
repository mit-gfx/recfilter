/**
 * \file complex_arith.h
 * \brief Complex arithmetic
 * \author Pascal Getreuer <getreuer@cmla.ens-cachan.fr>
 *
 * This file is a light C89-compatible implementation of complex arithmetic.
 * Functions are defined for addition, subtraction, multiplication, division,
 * magnitude, argument (angle), power, sqrt, exp, and logarithm.
 *
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

#ifndef _COMPLEX_ARITH_H_
#define _COMPLEX_ARITH_H_

#include <math.h>

/** \brief Complex double data type. */
typedef struct _complex_type
{
    double real;    /**< real part      */
    double imag;    /**< imaginary part */
} _complex_type;

/** \brief Short alias for _complex_type */
#define complex     _complex_type

#ifdef __GNUC__
__attribute__((pure,unused))
#endif
/** \brief Make a complex number z = a + ib. */
static complex make_complex(double a, double b)
{
    complex z;
    z.real = a;
    z.imag = b;
    return z;
}

#ifdef __GNUC__
__attribute__((pure,unused))
#endif
/** \brief Complex conjugate z*. */
static complex c_conj(complex z)
{
    complex result;
    result.real = z.real;
    result.imag = -z.imag;
    return result;
}

#ifdef __GNUC__
__attribute__((pure,unused))
#endif
/** \brief Complex addition. */
static complex c_add(complex w, complex z)
{
    complex result;
    result.real = w.real + z.real;
    result.imag = w.imag + z.imag;
    return result;
}

#ifdef __GNUC__
__attribute__((pure,unused))
#endif
/** \brief Complex negation -z. */
static complex c_neg(complex z)
{
    complex result;
    result.real = -z.real;
    result.imag = -z.imag;
    return result;
}

#ifdef __GNUC__
__attribute__((pure,unused))
#endif
/** \brief Complex subtraction. */
static complex c_sub(complex w, complex z)
{
    complex result;
    result.real = w.real - z.real;
    result.imag = w.imag - z.imag;
    return result;
}

#ifdef __GNUC__
__attribute__((pure,unused))
#endif
/** \brief Complex multiplication. */
static complex c_mul(complex w, complex z)
{
    complex result;
    result.real = w.real * z.real - w.imag * z.imag;
    result.imag = w.real * z.imag + w.imag * z.real;
    return result;
}

#ifdef __GNUC__
__attribute__((pure,unused))
#endif
/** \brief Complex multiplicative inverse 1/z. */
static complex c_inv(complex z)
{
    complex result;
    
    /* There are two mathematically-equivalent formulas for the inverse. For
     * accuracy, choose the formula with the smaller value of |ratio|.
     */
    if (fabs(z.real) >= fabs(z.imag))
    {
        double ratio = z.imag / z.real;
        double denom = z.real + z.imag * ratio;
        result.real = 1 / denom;
        result.imag = -ratio / denom;
    }
    else
    {
        double ratio = z.real / z.imag;
        double denom = z.real * ratio + z.imag;
        result.real = ratio / denom;
        result.imag = -1 / denom;
    }
    
    return result;
}

#ifdef __GNUC__
__attribute__((pure,unused))
#endif
/** \brief Complex division w/z. */
static complex c_div(complex w, complex z)
{
    complex result;
    
    /* For accuracy, choose the formula with the smaller value of |ratio|. */
    if (fabs(z.real) >= fabs(z.imag))
    {
        double ratio = z.imag / z.real;
        double denom = z.real + z.imag * ratio;
        result.real = (w.real + w.imag * ratio) / denom;
        result.imag = (w.imag - w.real * ratio) / denom;
    }
    else
    {
        double ratio = z.real / z.imag;
        double denom = z.real * ratio + z.imag;
        result.real = (w.real * ratio + w.imag) / denom;
        result.imag = (w.imag * ratio - w.real) / denom;
    }
    
    return result;
}

#ifdef __GNUC__
__attribute__((pure,unused))
#endif
/** \brief Complex magnitude. */
static double c_mag(complex z)
{
    z.real = fabs(z.real);
    z.imag = fabs(z.imag);
    
    /* For accuracy, choose the formula with the smaller value of |ratio|. */
    if (z.real >= z.imag)
    {
        double ratio = z.imag / z.real;
        return z.real * sqrt(1.0 + ratio * ratio);
    }
    else
    {
        double ratio = z.real / z.imag;
        return z.imag * sqrt(1.0 + ratio * ratio);
    }
}

#ifdef __GNUC__
__attribute__((pure,unused))
#endif
/** \brief Complex argument (angle) in [-pi,+pi]. */
static double c_arg(complex z)
{
    return atan2(z.imag, z.real);
}

#ifdef __GNUC__
__attribute__((pure,unused))
#endif
/** \brief Complex power w^z. */
static complex c_pow(complex w, complex z)
{
    complex result;
    double mag_w = c_mag(w);
    double arg_w = c_arg(w);
    double mag = pow(mag_w, z.real) * exp(-z.imag * arg_w);
    double arg = z.real * arg_w + z.imag * log(mag_w);
    result.real = mag * cos(arg);
    result.imag = mag * sin(arg);
    return result;
}

#ifdef __GNUC__
__attribute__((pure,unused))
#endif
/** \brief Complex power w^x with real exponent. */
static complex c_real_pow(complex w, double x)
{
    complex result;
    double mag = pow(c_mag(w), x);
    double arg = c_arg(w) * x;
    result.real = mag * cos(arg);
    result.imag = mag * sin(arg);
    return result;
}

#ifdef __GNUC__
__attribute__((pure,unused))
#endif
/** \brief Complex square root (principal branch). */
static complex c_sqrt(complex z)
{
    double r = c_mag(z);
    complex result;
    result.real = sqrt((r + z.real) / 2);
    result.imag = sqrt((r - z.real) / 2);
    
    if (z.imag < 0)
        result.imag = -result.imag;
    
    return result;
}

#ifdef __GNUC__
__attribute__((pure,unused))
#endif
/** \brief Complex exponential. */
static complex c_exp(complex z)
{
    double r = exp(z.real);
    complex result;
    result.real = r * cos(z.imag);
    result.imag = r * sin(z.imag);
    return result;
}

#ifdef __GNUC__
__attribute__((pure,unused))
#endif
/** \brief Complex logarithm (principal branch). */
static complex c_log(complex z)
{
    complex result;
    result.real = log(c_mag(z));
    result.imag = c_arg(z);
    return result;
}

#endif /* _COMPLEX_ARITH_H_ */
