/**
 * \file filter_util.h
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

#ifndef _FILTER_UTIL_H_
#define _FILTER_UTIL_H_

#include "num.h"

#ifdef __GNUC__
__attribute__((pure,unused))
#endif
/**
 * \brief Half-sample symmetric boundary extension
 * \param N     signal length
 * \param n     requested sample, possibly outside {0,...,`N`-1}
 * \return reflected sample in {0,...,`N`-1}
 *
 * This function is used for boundary handling. Suppose that `src` is an array
 * of length `N`, then `src[extension(N, n)]` evaluates the symmetric
 * extension of `src` at location `n`.
 *
 * Half-sample symmetric extension is implemented by the pseudocode
\verbatim
    repeat
        if n < 0, reflect n over -1/2
        if n >= N, reflect n over N - 1/2
    until 0 <= n < N
\endverbatim
 * The loop is necessary as some `n` require multiple reflections to bring
 * them into the domain {0,...,`N`-1}.
 *
 * This function is used by all of the Gaussian convolution algorithms
 * included in this work except for DCT-based convolution (where symmetric
 * boundary handling is performed implicitly by the transform). For FIR, box,
 * extended box, SII, and Deriche filtering, this function could be replaced
 * to apply some other boundary extension (e.g., periodic or constant
 * extrapolation) without any further changes. However, Alvarez-Mazorra and
 * Vliet-Young-Verbeek are hard-coded for symmetric extension on the right
 * boundary, and would require specific modification to change the handling
 * on the right boundary.
 *
 * \par A note on efficiency
 * This function is a computational bottleneck, as it is used within core
 * filtering loops. As a small optimization, we encourage inlining by defining
 * the function as `static`. We refrain from further optimization since this
 * is a pedagogical implementation, and code readability is more important.
 * Ideally, filtering routines should take advantage of algorithm-specific
 * properties such as exploiting sequential sample locations (to update the
 * extension cheaply) and samples that are provably in the interior (where
 * boundary checks may omitted be entirely).
 */
static long extension(long N, long n)
{
    while (1)
        if (n < 0)
            n = -1 - n;         /* Reflect over n = -1/2.    */
        else if (n >= N)
            n = 2 * N - 1 - n;  /* Reflect over n = N - 1/2. */
        else
            break;
        
    return n;
}

void recursive_filter_impulse(num *h, long N,
    const num *b, int p, const num *a, int q);

void init_recursive_filter(num *dest, const num *src, long N, long stride,
    const num *b, int p, const num *a, int q,
    num sum, num tol, long max_iter);

#endif /* _FILTER_UTIL_H_ */
