/**
 * \file box_filter.h
 *
 * Iterated box filters using IIR formulation
 * - single iteration is computed as summed are table followed by finite differencing
 * - 2-4 iterations are computed as n-order integral image followed by finite differencing
 * - higher iterations are computed as cascades of 1 to 4 interations of box filters
 *
 * n-order integral image can be computed in a single pass as n-order IIR filter.
 */

#include <iostream>
#include <Halide.h>

#include "recfilter.h"
#include "iir_coeff.h"

using namespace Halide;

using std::vector;
using std::string;
using std::stringstream;
using std::cerr;
using std::endl;

/** Compute an order n integral image */
template<typename T>
RecFilter integral_image(
        int order,          ///< order of integral image
        int width,          ///< width of image
        int height,         ///< height of image
        int tile_width,     ///< tile width for tiling the intergal image
        Image<T> I          ///< input image expressed as a Func
        );

/** Compute an order n integral image */
RecFilter integral_image(
        int order,          ///< order of integral image
        int width,          ///< width of image
        int height,         ///< height of image
        int tile_width,     ///< tile width for tiling the intergal image
        Func I              ///< input image expressed as a Func
        );

/** Compute a n-times box filter from an n-order intergal image */
RecFilter derivative_image(
        int order,          ///< order of box filter
        int filter_radius,  ///< box filter radius
        int width,          ///< width of image
        int height,         ///< height of image
        int tile_width,     ///< tile width for tiling
        Func I              ///< n-order integral image
        );

// -----------------------------------------------------------------------------

template<typename T>
RecFilter integral_image(int order, int width, int height, int tile_width, Image<T> I) {
    int n = I.dimensions();
    vector<Var> a;
    for (int i=0; i<n; i++) {
        Var x;
        a.push_back(x);
    }
    Func R;
    R(a) = I(a);
    return integral_image(order, width, height, tile_width, R);
}

RecFilter integral_image(int order, int width, int height, int tile_width, Func I) {
    vector<float> coeff = integral_image_coeff(order);

    RecFilterDim x("x", width);
    RecFilterDim y("y", height);

    stringstream s;
    s << "IntImg_" << order;

    RecFilter F(s.str());

    F(x,y) = I(x,y);                // k order intergral image
    F.add_filter(x, coeff);         // k = num of box filter iterations
    F.add_filter(y, coeff);

    F.split(x, tile_width, y, tile_width);

    // -------------------------------------------------------------------------

    if (F.target().has_gpu_feature()) {
        int order    = 1;
        int n_scans  = 2;
        int ws       = 32;
        int unroll_w = ws/4;
        int intra_tiles_per_warp = ws / (order*n_scans);
        int inter_tiles_per_warp = 4;

        F.intra_schedule(1).compute_locally()
            .reorder_storage(F.inner(), F.outer())
            .unroll         (F.inner_scan())
            .split          (F.inner(1), unroll_w)
            .unroll         (F.inner(1).split_var())
            .reorder        (F.inner_scan(), F.inner(1).split_var(), F.inner(), F.outer())
            .gpu_threads    (F.inner(0), F.inner(1))
            .gpu_blocks     (F.outer(0), F.outer(1));

        F.intra_schedule(2).compute_locally()
            .unroll         (F.inner_scan())
            .split          (F.outer(0), intra_tiles_per_warp)
            .reorder        (F.inner(),  F.inner_scan(), F.tail(), F.outer(0).split_var(), F.outer())
            .fuse           (F.tail(), F.inner(0))
            .gpu_threads    (F.tail(), F.outer(0).split_var())
            .gpu_blocks     (F.outer(0), F.outer(1));

        F.inter_schedule().compute_globally()
            .reorder_storage(F.inner(), F.tail(), F.outer())
            .unroll         (F.outer_scan())
            .split          (F.outer(0), inter_tiles_per_warp)
            .reorder        (F.outer_scan(), F.tail(), F.outer(0).split_var(), F.inner(), F.outer())
            .gpu_threads    (F.inner(0), F.outer(0).split_var())
            .gpu_blocks     (F.outer(0));
    }

    return F;
}

RecFilter derivative_image(int order, int filter_radius, int width, int height, int tile_width, Func I) {
    int B = filter_radius;
    int z = 2*B+1;

    Var xo("xo"), xi("xi"), xii("xii");
    Var yo("yo"), yi("yi"), yii("yii");

    RecFilterDim x("x", width);
    RecFilterDim y("y", height);

    Var u = x.var();
    Var v = y.var();

    RecFilter F("Box");

    if (order==1) {
        // s = [-1 (2B-1 zeros) 1] * (1/z)
        // s2 = conv2(s, s')
        F(x,y) = (
                 1.0f * I(clamp(x+B+0, 0, width-1), clamp(y+B+0, 0, height-1)) +
                -1.0f * I(clamp(x+B+0, 0, width-1), clamp(y-B-1, 0, height-1)) +
                 1.0f * I(clamp(x-B-1, 0, width-1), clamp(y-B-1, 0, height-1)) +
                -1.0f * I(clamp(x-B-1, 0, width-1), clamp(y+B+0, 0, height-1))) / std::pow(z,2);

    } else {
        cerr << "Cannot compute the higher than second order derivative. Instead, "
             << "compute lower order derivative and cascade them to the same effect." << endl;
        assert(false);
    }

    F.compute_globally()
        .split  (u, xo, xi, tile_width)
        .split  (v, yo, yi, tile_width)
        .split  (yi,yi, yii,8)
        .unroll (yii)
        .reorder(yii,xi,yi,xo,yo)
        .gpu    (xo,yo,xi,yi);

    return F;
}
