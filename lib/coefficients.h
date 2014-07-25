#ifndef _COEFFICIENTS_H_
#define _COEFFICIENTS_H_

#include "recfilter.h"

/** @brief Weight coefficients (tail_size x tile_width) for
 * applying scans corresponding to split indices split_id1 to
 * split_id2 in the SplitInfo struct (defined in coefficients.cpp).
 * It is meaningful to apply subsequent scans on the tail of any scan
 * as it undergoes other scans only if they happen after the first
 * scan. The SpliInfo object stores the scans in reverse order, hence indices
 * into the SplitInfo object split_id1 and split_id2 must be decreasing
 */
Halide::Image<float> tail_weights(SplitInfo s, int s1, bool clamp_border=false);


/** @brief Weight coefficients (tail_size x tile_width) for
 * applying scan's corresponding to split indices split_id1
 * (defined in coefficients.cpp)
 */
Halide::Image<float> tail_weights(SplitInfo s, int s1, int s2, bool clamp_border=false);


#endif // _COEFFICIENTS_H_
