/**
 * \file inverfc_acklam.c
 * \brief Acklam's algorithm for the inverse complementary error function
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

#include "inverfc_acklam.h"

/**
 * \brief Inverse complementary error function \f$\mathrm{erfc}^{-1}(x)\f$
 * \param x     input argument
 *
 * Reference: P.J. Acklam, "An algorithm for computing the inverse normal
 * cumulative distribution function," 2010, online at
 * http://home.online.no/~pjacklam/notes/invnorm/
 */
double inverfc_acklam(double x)
{
    static const double a[] = { -3.969683028665376e1, 2.209460984245205e2,
        -2.759285104469687e2, 1.383577518672690e2, -3.066479806614716e1,
        2.506628277459239};
    static const double b[] = { -5.447609879822406e1, 1.615858368580409e2,
        -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1};
    static const double c[] = { -7.784894002430293e-3, -3.223964580411365e-1,
        -2.400758277161838, -2.549732539343734, 4.374664141464968,
        2.938163982698783};
    static const double d[] = { 7.784695709041462e-3, 3.224671290700398e-1,
        2.445134137142996, 3.754408661907416};
    double y, e, u;
    
    x /= 2.0;
    
    if (0.02425 <= x && x <= 0.97575)
    {
        double q = x - 0.5;
        double r = q * q;
        y = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q
            / (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1);
    }
    else
    {
        double q = sqrt(-2.0 * log((x > 0.97575) ? (1.0 - x) : x));
        y = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
            / ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1);
        
        if (x > 0.97575)
            y = -y;
    }
    
    e = 0.5 * erfc_cody(-y/M_SQRT2) - x;
    u = e * M_SQRT2PI * exp(0.5 * y * y);
    y -= u / (1.0 + 0.5 * y * u);
    return -y / M_SQRT2;
}
