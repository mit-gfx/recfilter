/**
 * \file box_filter.h
 *
 * Iterated box filters using IIR formulation
 * - single iteration is computed as summed are table followed by finite differencing
 * - 2 iterations are computed as 2-order integral image cascaded in x and y
 * - higher iterations can be computed as cascades of the above two
 *
 * \todo the assumption is that the image is padded by k pixels where
 * k = box_filter_radius * num_applications_of_filter + 1
 */

#include <iostream>
#include <Halide.h>

#include "recfilter.h"
#include "iir_coeff.h"

using namespace Halide;

RecFilter box_filter_order_1(Image<float> I, int width, int height, int B, int tile_width, bool autoschedule) {
    RecFilterDim x("x", width);
    RecFilterDim y("y", height);

    RecFilter F("Box1_Sat");
    RecFilter D("Box1_Diff");

    // integral_image
    F(x,y) = I(x,y);
    F.add_filter(x, {1.0f, 1.0f});
    F.add_filter(y, {1.0f, 1.0f});
    F.split(x, tile_width, y, tile_width);

    // derivative_image, set border as input image
    Func f = F.as_func();
    D(x,y) =(1.0f * f(clamp(x+B+0, 0, width-1), clamp(y+B+0, 0, height-1)) +
            -1.0f * f(clamp(x+B+0, 0, width-1), clamp(y-B-1, 0, height-1)) +
             1.0f * f(clamp(x-B-1, 0, width-1), clamp(y-B-1, 0, height-1)) +
            -1.0f * f(clamp(x-B-1, 0, width-1), clamp(y+B+0, 0, height-1))) / std::pow(2*B+1,2);


    // -------------------------------------------------------------------------
    // x-y overlapped schedule copied from summed area table

    Var xo("xo"), xi("xi"), xii("xii");
    Var yo("yo"), yi("yi"), yii("yii");

    Var u = x.var();
    Var v = y.var();

    // -------------------------------------------------------------------------
    // schedule for the differencing operator, this is a pointwise operation
    // easiest to just use Halide scheduling API for these
    D.as_func()
        .compute_root()
        .split  (u, xo, xi, tile_width)
        .split  (v, yo, yi, tile_width)
        .split  (yi,yi, yii,8)
        .unroll (yii)
        .reorder(yii,xi,yi,xo,yo)
        .gpu    (xo,yo,xi,yi);

    // -------------------------------------------------------------------------
    // schedule for the recursive filters

    if (autoschedule) {
        F.gpu_auto_schedule();
    }
    else {
        int n_scans  = 2;
        int ws       = 32;
        int unroll_w = ws/4;
        int intra_tiles_per_warp = ws / n_scans;
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

    return D;
}

RecFilter box_filter_order_2(Func I, int width, int height, int B, int tile_width, bool autoschedule) {
    RecFilter sat_x ("Box2_Satx");     // 2nd order prefix sum along x
    RecFilter diff_x("Box2_Diffx");    // 2nd order finite differencing along x
    RecFilter sat_y ("Box2_Saty");     // 2nd order prefix sum along y
    RecFilter diff_y("Box2_Diffy");    // 2nd order finite differencing along y

    RecFilterDim x("x", width);
    RecFilterDim y("y", height);

    Var xi, yi, xo, yo, yii;
    Var u = x.var();
    Var v = y.var();

    std::vector<float> coeff = integral_image_coeff(2);

    // -------------------------------------------------------------------------

    // x filter: 2nd order intergal image along x followed by finite differencing

    sat_x(x,y) = I(x,y);
    sat_x.add_filter(+x, coeff);

    // difference operator
    Func fA; fA(u,v) = (sat_x(min(u+B,width-1), v)-sat_x(max(u-B-1,0), v)) / float(2*B+1);
    diff_x(x,y) = (fA(min(x+B,width-1), v)-fA(max(x-B-1,0), v)) / float(2*B+1);


    // y filter: 2nd order intergal image along y followed by finite differencing

    sat_y(x,y) = diff_x.as_func()(x,y);
    sat_y.add_filter(+y, coeff);

    // difference operator
    Func gA; gA(u,v) = (sat_y(u, min(v+B,height-1))-sat_y(u, max(v-B-1,0))) / float(2*B+1);
    diff_y(x,y) = (gA(x, min(y+B,height-1))-gA(x, max(y-B-1,0))) / float(2*B+1);

    // -------------------------------------------------------------------------
    // tile the IIR filters

    sat_x.split_all_dimensions(tile_width);
    sat_y.split_all_dimensions(tile_width);

    // -------------------------------------------------------------------------
    // schedule for the differencing operators in both dimensions
    // these are pointwise operations and not recursive filters, easiest to just
    // use Halide scheduling API for these

    diff_x.as_func()
        .compute_root()
        .split  (u, xo, xi, tile_width)
        .split  (v, yo, yi, tile_width)
        .split  (yi,yi, yii, 8)
        .unroll (yii)
        .reorder(yii,xi,yi,xo,yo)
        .gpu    (xo,yo,xi,yi);

    diff_y.as_func()
        .compute_root()
        .split  (u, xo, xi, tile_width)
        .split  (v, yo, yi, tile_width)
        .split  (yi,yi, yii, 8)
        .unroll (yii)
        .reorder(yii,xi,yi,xo,yo)
        .gpu    (xo,yo,xi,yi);

    // -------------------------------------------------------------------------
    // schedule for the recursive filters

    if (autoschedule) {
        sat_x.gpu_auto_schedule();
        sat_y.gpu_auto_schedule();
    }
    else {
        int ws       = 32;
        int unroll_w = 8;
        int tiles_per_warp = 4;

        // schedule for x IIR filter

        sat_x.intra_schedule().compute_locally()
            .split          (sat_x.full(0), ws, sat_x.inner())
            .split          (sat_x.inner(1), unroll_w)
            .unroll         (sat_x.inner(1).split_var())
            .unroll         (sat_x.inner_scan())
            .reorder        (sat_x.inner_scan(), sat_x.inner(1).split_var(), sat_x.tail(), sat_x.inner(), sat_x.outer(), sat_x.full())
            .gpu_threads    (sat_x.inner(0), sat_x.inner(1))
            .gpu_blocks     (sat_x.outer(0), sat_x.full(0));

        sat_x.inter_schedule().compute_globally()
            .reorder_storage(sat_x.full(0), sat_x.tail(), sat_x.outer(0))
            .split          (sat_x.full(0), ws, sat_x.inner())
            .unroll         (sat_x.outer_scan())
            .split          (sat_x.full(0), tiles_per_warp)
            .reorder        (sat_x.outer_scan(), sat_x.tail(), sat_x.full(0).split_var(), sat_x.inner(), sat_x.full(0))
            .gpu_threads    (sat_x.inner(0), sat_x.full(0).split_var())
            .gpu_blocks     (sat_x.full(0));

        // slightly different schedule for y IIR filter

        sat_y.intra_schedule().compute_locally()
            .reorder_storage(sat_y.full(0), sat_y.inner(), sat_y.outer(0))
            .split      (sat_y.full(0), ws)
            .split      (sat_y.inner(0), unroll_w)
            .unroll     (sat_y.inner(0).split_var())
            .unroll     (sat_y.inner_scan())
            .reorder    (sat_y.inner_scan(), sat_y.inner(0).split_var(), sat_y.inner(0), sat_y.full(0).split_var(), sat_y.full(0), sat_y.outer(0))
            .gpu_threads(sat_y.full(0).split_var(), sat_y.inner(0))
            .gpu_blocks (sat_y.full(0), sat_y.outer(0));

        sat_y.inter_schedule().compute_globally()
            .reorder_storage(sat_y.full(0), sat_y.tail(), sat_y.outer(0))
            .split          (sat_y.full(0), ws, sat_y.inner())
            .unroll         (sat_y.outer_scan())
            .split          (sat_y.full(0), tiles_per_warp)
            .reorder        (sat_y.outer_scan(), sat_y.tail(), sat_y.full(0).split_var(), sat_y.inner(), sat_y.full(0))
            .gpu_threads    (sat_y.inner(0), sat_y.full(0).split_var())
            .gpu_blocks     (sat_y.full(0));
    }

    return diff_y;
}
