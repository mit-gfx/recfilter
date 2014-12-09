/**
 * \file inverfc_acklam.h
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

#ifndef _INVERFC_ACKLAM_H_
#define _INVERFC_ACKLAM_H_

#include <math.h>
#include "erfc_cody.h"

#ifndef M_SQRT2
/** \brief The constant \f$ \sqrt{2} \f$ */
#define M_SQRT2     1.41421356237309504880168872420969808
#endif
#ifndef M_SQRT2PI
/** \brief The constant \f$ \sqrt{2 \pi} \f$ */
#define M_SQRT2PI   2.50662827463100050241576528481104525
#endif

double inverfc_acklam(double x);

/** \brief Short alias of inverfc_acklam() */
#define inverfc(x)  inverfc_acklam(x)

#endif /* _INVERFC_ACKLAM_H_ */
