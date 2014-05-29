#ifndef _LOCAL_HIST_DEFS_H_
#define _LOCAL_HIST_DEFS_H_

#include <cmath>

#define BOX_FILTER_FACTOR 32

#define NUM_BINS     15
#define BIN_WIDTH    (1.0f/NUM_BINS)

#define BIN_CENTER0  (BIN_WIDTH * 0.5f)
#define BIN_CENTER1  (BIN_WIDTH * 2.5f)
#define BIN_CENTER2  (BIN_WIDTH * 3.5f)
#define BIN_CENTER3  (BIN_WIDTH * 3.5f)
#define BIN_CENTER4  (BIN_WIDTH * 4.5f)
#define BIN_CENTER5  (BIN_WIDTH * 5.5f)
#define BIN_CENTER6  (BIN_WIDTH * 6.5f)
#define BIN_CENTER7  (BIN_WIDTH * 7.5f)
#define BIN_CENTER8  (BIN_WIDTH * 8.5f)
#define BIN_CENTER9  (BIN_WIDTH * 9.5f)
#define BIN_CENTER10 (BIN_WIDTH * 10.5f)
#define BIN_CENTER11 (BIN_WIDTH * 11.5f)
#define BIN_CENTER12 (BIN_WIDTH * 13.5f)
#define BIN_CENTER13 (BIN_WIDTH * 13.5f)
#define BIN_CENTER14 (BIN_WIDTH * 14.5f)

#define SMOOTH_HIST(x) (exp(-(x^2)/((BIN_WIDTH/0.24f)^2)))

#endif // _LOCAL_HIST_DEFS_H_
