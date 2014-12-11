/**
 * \file gaussian_short_conv.h
 * \brief Gaussian convolution for short signals (N <= 4)
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
 * You should have received a copy of these licenses along this program.
 * If not, see <http://www.gnu.org/licenses/> and
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

#ifndef GAUSSIAN_SHORT_CONV_H
#define GAUSSIAN_SHORT_CONV_H
#include "num.h"

void gaussian_short_conv(num *dest, const num *src,
    long N, long stride, num sigma);

#endif /* GAUSSIAN_SHORT_CONV_H */
