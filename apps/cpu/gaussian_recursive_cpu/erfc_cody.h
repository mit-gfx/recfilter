/**
 * \file erfc_cody.h
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

#ifndef _ERFC_CODY_H_
#define _ERFC_CODY_H_

#include <math.h>

#ifndef M_1_SQRTPI
/** \brief The constant \f$ 1/\pi \f$ */
#define M_1_SQRTPI  0.564189583547756286948
#endif

double erfc_cody(double x);

/** \brief Short alias of erfc_cody() */
#define erfc(x)     erfc_cody(x)

#endif /* _ERFC_CODY_H_ */
