/**
 * \file erfc_cody.c
 * \brief W.J. Cody's approximation of the complementary error function (erfc)
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
 * You should have received a copy of these licenses along with this program.
 * If not, see <http://www.gnu.org/licenses/> and
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

#include "erfc_cody.h"

/**
 * \brief Evaluate rational polynomial for erfc approximation
 * \param P     numerator coefficients
 * \param Q     denominator coefficients
 * \param N     order
 * \param x     x-value at which to evaluate the rational polynomial
 *
 * Evaluates rational polynomial
 * \f[ \frac{P[N+1] x^{N+1} + P[0] x^N + \cdots + P[N-1] x + P[N]}
       {x^{N+1} + Q[0] x^N + \cdots + Q[N-1] x + Q[N]}. \f]
 */
static double erfc_cody_rpoly(const double *P,
    const double *Q, int N, double x)
{
    double xnum = P[N + 1] * x, xden = x;
    int n;
    
    for (n = 0; n < N; ++n)
    {
        xnum = (xnum + P[n]) * x;
        xden = (xden + Q[n]) * x;
    }
    
    return (xnum + P[N]) / (xden + Q[N]);
}

/**
 * \brief Complementary error function
 *
 * Based on the public domain NETLIB (Fortran) code by W. J. Cody
 * Applied Mathematics Division
 * Argonne National Laboratory
 * Argonne, IL 60439
 *
 * From the original documentation:
 * The main computation evaluates near-minimax approximations from "Rational
 * Chebyshev approximations for the error function" by W. J. Cody, Math.
 * Comp., 1969, PP. 631-638. This transportable program uses rational
 * functions that theoretically approximate erf(x) and erfc(x) to at least 18
 * significant decimal digits. The accuracy achieved depends on the
 * arithmetic system, the compiler, the intrinsic functions, and proper
 * selection of the machine-dependent constants.
 */
double erfc_cody(double x)
{
    static const double P1[5] = { 3.16112374387056560e0,
        1.13864154151050156e2, 3.77485237685302021e2,
        3.20937758913846947e3, 1.85777706184603153e-1 };
    static const double Q1[4] = { 2.36012909523441209e1,
        2.44024637934444173e2, 1.28261652607737228e3,
        2.84423683343917062e3 };
    static const double P2[9] = { 5.64188496988670089e-1,
        8.88314979438837594e0, 6.61191906371416295e1,
        2.98635138197400131e2, 8.81952221241769090e2,
        1.71204761263407058e3, 2.05107837782607147e3,
        1.23033935479799725e3, 2.15311535474403846e-8 };
    static const double Q2[8] = { 1.57449261107098347e1,
        1.17693950891312499e2, 5.37181101862009858e2,
        1.62138957456669019e3, 3.29079923573345963e3,
        4.36261909014324716e3, 3.43936767414372164e3,
        1.23033935480374942e3 };
    static const double P3[6] = { 3.05326634961232344e-1,
        3.60344899949804439e-1, 1.25781726111229246e-1,
        1.60837851487422766e-2, 6.58749161529837803e-4,
        1.63153871373020978e-2 };
    static const double Q3[5] = { 2.56852019228982242e0,
        1.87295284992346047e0, 5.27905102951428412e-1,
        6.05183413124413191e-2, 2.33520497626869185e-3 };
    double y, result;
    
    y = fabs(x);
    
    if (y <= 0.46875)
        return 1 - x * erfc_cody_rpoly(P1, Q1, 3, (y > 1.11e-16) ? y*y : 0);
    else if (y <= 4)
        result = exp(-y*y) * erfc_cody_rpoly(P2, Q2, 7, y);
    else if (y >= 26.543)
        result = 0;
    else
        result = exp(-y*y) * ( (M_1_SQRTPI
            - y*y * erfc_cody_rpoly(P3, Q3, 4, 1.0/(y*y))) / y );
    
    return (x < 0) ? (2 - result) : result;
}
