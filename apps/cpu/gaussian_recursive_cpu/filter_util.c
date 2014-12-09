/**
 * \file filter_util.c
 * \brief Filtering utility functions
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

#include "filter_util.h"
#include <assert.h>
#include <math.h>
#include <string.h>

/** \brief Maximum possible value of q in init_recursive_filter() */
#define MAX_Q       7

/**
 * \brief Compute taps of the impulse response of a causal recursive filter
 * \param h     destination array of size N
 * \param N     number of taps to compute (beginning from n = 0)
 * \param b     numerator coefficients
 * \param p     largest delay of the numerator
 * \param a     denominator coefficients
 * \param q     largest delay of the denominator
 *
 * Computes taps \f$ h_0, \ldots, h_{N-1} \f$ of the impulse response of the
 * recursive filter
 * \f[ H(z) = \frac{b[0] + b[1]z^{-1} + \cdots + b[p]z^{-p}}
 *            {1 + a[1]z^{-1} + \cdots + a[q]z^{-q}}, \f]
 * or equivalently
 * \f[ \begin{aligned} h[n] = & b[0]\delta_n + b[1]\delta_{n-1}+\cdots +
 * b[p]\delta_{n-p}\\ & -a[1]h[n-1]-\cdots -a[q]h[n-q], \end{aligned} \f]
 * for n = 0, ..., N-1 where \f$ \delta \f$ is the unit impulse. In the
 * denominator coefficient array, element a[0] is not used.
 */
void recursive_filter_impulse(num *h, long N,
    const num *b, int p, const num *a, int q)
{
    long m, n;
    
    assert(h && N > 0 && b && p >= 0 && a && q > 0);
    
    for (n = 0; n < N; ++n)
    {
        h[n] = (n <= p) ? b[n] : 0;
        
        for (m = 1; m <= q && m <= n; ++m)
            h[n] -= a[m] * h[n - m];
    }
    
    return;
}

/**
 * \brief Initialize a causal recursive filter with boundary extension
 * \param dest      destination array with size of at least q
 * \param src       input signal of length N
 * \param N         length of src
 * \param stride    the stride between successive samples of src
 * \param b         numerator coefficients
 * \param p         largest delay of the numerator
 * \param a         denominator coefficients
 * \param q         largest delay of the denominator
 * \param sum       the L^1 norm of the impulse response
 * \param tol       accuracy tolerance
 * \param max_iter  maximum number of samples to use for approximation
 *
 * This routine initializes a recursive filter,
 * \f[  \begin{aligned} u_n &= b_0 f_n + b_1 f_{n-1} + \cdots + b_p f_{n-p}
&\quad - a_1 u_{n-1} - a_2 u_{n-2} - \cdots - a_q u_{n-q}, \end{aligned} \f]
 * with boundary extension by approximating the infinite sum
 * \f[ u_m=\sum_{n=-m}^\infty h_{n+m}\Tilde{f}_{-n} \approx \sum_{n=-m}^{k-1}
           h_{n+m} \Tilde{f}_{-n}, \quad m = 0, \ldots, q - 1. \f]
 */
void init_recursive_filter(num *dest, const num *src, long N, long stride,
    const num *b, int p, const num *a, int q,
    num sum, num tol, long max_iter)
{
    num h[MAX_Q + 1];
    long n;
    int m;
    
    assert(dest && src && N > 0 && stride != 0
        && b && p >= 0 && a && 0 < q && q <= MAX_Q
        && tol > 0 && max_iter > 0);
    
    /* Compute the first q taps of the impulse response, h_0, ..., h_{q-1} */
    recursive_filter_impulse(h, q, b, p, a, q);
    
    /* Compute dest_m = sum_{n=1}^m h_{m-n} src_n, m = 0, ..., q-1 */
    for (m = 0; m < q; ++m)
        for (dest[m] = 0, n = 1; n <= m; ++n)
            dest[m] += h[m - n] * src[stride * extension(N, n)];

    for (n = 0; n < max_iter; ++n)
    {
        num cur = src[stride * extension(N, -n)];
        
        /* dest_m = dest_m + h_{n+m} src_{-n} */
        for (m = 0; m < q; ++m)
            dest[m] += h[m] * cur;
        
        sum -= fabs(h[0]);
        
        if (sum <= tol)
            break;
        
        /* Compute the next impulse response tap, h_{n+q} */
        h[q] = (n + q <= p) ? b[n + q] : 0;
        
        for (m = 1; m <= q; ++m)
            h[q] -= a[m] * h[q - m];
        
        /* Shift the h array for the next iteration */
        for (m = 0; m < q; ++m)
            h[m] = h[m + 1];
    }
    
    return;
}
