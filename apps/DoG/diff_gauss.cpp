/**
 * \file diff_gauss.cpp
 *
 * Difference of Gaussians approximated using box filters
 * - convert RGB image to V or HSV
 * - compute first box filter
 *   - compute x-y overlapped summed area table
 *   - compute two different box filters using different box radii
 * - apply 2nd order x box filter on each output
 * - apply 2nd order y box filter on each output
 *   - compute the difference between the two in the finite difference operator
 *
 * \todo imporve numerical stability by dividing the image by 2r before computing
 *   and integral image instead of computing the integral image and then dividing
 *   during finite differencing
 */

#include <iostream>
#include <Halide.h>

#include <recfilter.h>
#include <iir_coeff.h>

using namespace Halide;

/// Difference operator to compute 1D x double box filter from 2nd order intergal image
Expr diff_op_x(Func F, Expr x, Expr y, int w, int h, int B, int channel);

/// Difference operator to compute 1D y double box filter from 2nd order intergal image
Expr diff_op_y(Func F, Expr x, Expr y, int w, int h, int B, int channel);

/// Difference operator to compute 2D xy single box filter from summed area table
Expr diff_op_xy(Func F, Expr x, Expr y, int w, int h, int B);

/// Manual schedule for difference of Gaussians, this is not used in the benchmark
/// and only illustrates what the automatic scheduler does under the hood, these can
/// used instead of the auto schedules
void manual_schedules(void);

int main(int argc, char** argv) {
    Arguments args(argc, argv);

    int iter       = args.iterations;
    int tile_width = args.block;
    int width      = args.width;
    int height     = args.width;

    // box filter radii for approximating Gaussians of given sigma with 3 iterations
    int sigma1 = 1.0f;
    int sigma2 = 2.0f;
    int B1 = gaussian_box_filter(sigma1, 3);
    int B2 = gaussian_box_filter(sigma2, 3);

    Image<int16_t> image = generate_random_image<int16_t>(width,height);

    // pad the image with zeros at borders
    int pad = std::max(3*B1+3, 3*B2+3);
    for (int k=0; k<image.channels(); k++) {
        for (int i=0; i<image.width(); i++) {
            for (int j=0; j<image.height(); j++) {
                if (i<pad || i>width-pad || j<pad || j>height-pad) {
                    image(i,j,k) = 0.0f;
                }
            }
        }
    }

    RecFilterDim x("x", width);
    RecFilterDim y("y", height);

    // Compute V of HSV from RGB input
    RecFilter V("V");
    // V(x,y) = Cast::make(type_of<float>(), max(image(x,y,0), max(image(x,y,1),image(x,y,2))));
    V(x,y) = Internal::Cast::make(type_of<float>(), image(x,y));

    // summed table from V channel
    RecFilter SAT("SAT");
    SAT(x,y) = V.as_func()(x,y);
    SAT.add_filter(+x, {1.0f, 1.0f});
    SAT.add_filter(+y, {1.0f, 1.0f});

    // compute 1 iteration of box filters from this result
    RecFilter Box1("Box1");
    Box1(x,y) = Tuple(diff_op_xy(SAT.as_func(), x, y, width, height, B1), diff_op_xy(SAT.as_func(), x, y, width, height, B2));

    // compute 2nd order x box filter, calculate the 2nd order intergal image first on both outputs
    RecFilter SAT2x("SAT2x");
    SAT2x(x,y) = Tuple(Box1.as_func()(x,y)[0], Box1.as_func()(x,y)[1]);
    SAT2x.add_filter(+x, {1.0f, 2.0f, -1.0f});

    // complete the 2nd order x box filter by difference operator
    RecFilter Box2x("Box2x");
    Box2x(x,y) = Tuple(diff_op_x(SAT2x.as_func(), x, y, width, height, B1, 0), diff_op_x(SAT2x.as_func(), x, y, width, height, B2, 1));

    // compute 2nd order y box filter, calculate the 2nd order intergal image first on both outputs
    RecFilter SAT2y("SAT2y");
    SAT2y(x,y) = Tuple(Box2x.as_func()(x,y)[0], Box2x.as_func()(x,y)[1]);
    SAT2y.add_filter(+y, {1.0f, 2.0f, -1.0f});

    // complete 2nd order y box filter, subtract the result and take square as final result
    RecFilter DoG("DoG");
    DoG(x,y) = diff_op_y(SAT2y.as_func(), x, y, width, height, B1, 0) - diff_op_y(SAT2y.as_func(), x, y, width, height, B2, 1);


    // -------------------------------------------------------------------------
    // tile all the intgral image functions

    SAT  .split_all_dimensions(tile_width);
    SAT2x.split_all_dimensions(tile_width);
    SAT2y.split_all_dimensions(tile_width);

    // -------------------------------------------------------------------------
    // schedules

    SAT  .gpu_auto_schedule(128);
    SAT2x.gpu_auto_schedule(128);
    SAT2y.gpu_auto_schedule(128);

    Box1 .gpu_auto_schedule(128, tile_width);
    Box2x.gpu_auto_schedule(128, tile_width);
    DoG  .gpu_auto_schedule(128, tile_width);

    // -------------------------------------------------------------------------
    // compile and profile
    DoG.profile(iter);

    return EXIT_SUCCESS;
}

Expr diff_op_x(Func F, Expr x, Expr y, int w, int h, int B, int c) {
    Expr e = (F(min(x+B,w-1), y)[c] - 2.0f*F(max(x-1,0), y)[c] + F(max(x-2*B-2,0), y)[c]) / float(2*B+1);
    return e;
}

Expr diff_op_y(Func F, Expr x, Expr y, int w, int h, int B, int c) {
    Expr e = (F(x, min(y+B,h-1))[c] - 2.0f*F(x, max(y-1,0))[c] + F(x, max(y-2*B-2,0))[c]) / float(2*B+1);
    return e;
}


// Difference operator to compute 2D xy single box filter from summed area table
Expr diff_op_xy(Func F, Expr x, Expr y, int w, int h, int B) {
    Expr e =(1.0f * F(clamp(x+B+0, 0, w-1), clamp(y+B+0, 0, h-1)) +
            -1.0f * F(clamp(x+B+0, 0, w-1), clamp(y-B-1, 0, h-1)) +
            1.0f * F(clamp(x-B-1, 0, w-1), clamp(y-B-1, 0, h-1)) +
            -1.0f * F(clamp(x-B-1, 0, w-1), clamp(y+B+0, 0, h-1))) / std::pow(2*B+1,2);
    return e;
}


void manual_schedules(void) {
    int ws       = 32;
    int unroll_w = ws/4;
    int intra_tiles_per_warp = ws/2;
    int inter_tiles_per_warp = 4;

    Var xo("xo"), xi("xi"), xii("xii"), u(x.var());
    Var yo("yo"), yi("yi"), yii("yii"), v(y.var());

    // first box filter
    {
        RecFilter F = SAT;

        F.intra_schedule(1).compute_locally()
            .reorder_storage(F.inner(), F.outer())
            .unroll         (F.inner_scan())
            .split          (F.inner(1), unroll_w)
            .unroll         (F.inner(1).split_var())
            .reorder        (F.inner_scan(), F.inner(1).split_var(), F.inner(), F.outer())
            .gpu_threads    (F.inner(0), F.inner(1))
            .gpu_blocks     (F.outer(0), F.outer(1));

        F.intra_schedule(2).compute_locally()
            .unroll     (F.inner_scan())
            .split      (F.outer(0), intra_tiles_per_warp)
            .reorder    (F.inner(),  F.inner_scan(), F.tail(), F.outer(0).split_var(), F.outer())
            .fuse       (F.tail(), F.inner(0))
            .gpu_threads(F.tail(), F.outer(0).split_var())
            .gpu_blocks (F.outer(0), F.outer(1));

        F.inter_schedule().compute_globally()
            .reorder_storage(F.inner(), F.tail(), F.outer())
            .unroll         (F.outer_scan())
            .split          (F.outer(0), inter_tiles_per_warp)
            .reorder        (F.outer_scan(), F.tail(), F.outer(0).split_var(), F.inner(), F.outer())
            .gpu_threads    (F.inner(0), F.outer(0).split_var())
            .gpu_blocks     (F.outer(0));
    }


    // second order 2 box filter along x
    {
        RecFilter sat_x  = SAT2x;

        sat_x.intra_schedule().compute_locally()
            .split      (sat_x.full(0), ws, sat_x.inner())
            .split      (sat_x.inner(1), unroll_w)
            .unroll     (sat_x.inner(1).split_var())
            .unroll     (sat_x.inner_scan())
            .reorder    (sat_x.inner_scan(), sat_x.inner(1).split_var(), sat_x.tail(), sat_x.inner(), sat_x.outer(), sat_x.full())
            .gpu_threads(sat_x.inner(0), sat_x.inner(1))
            .gpu_blocks (sat_x.outer(0), sat_x.full(0));

        sat_x.inter_schedule().compute_globally()
            .reorder_storage(sat_x.full(0), sat_x.tail(), sat_x.outer(0))
            .split          (sat_x.full(0), ws, sat_x.inner())
            .unroll         (sat_x.outer_scan())
            .split          (sat_x.full(0), inter_tiles_per_warp)
            .reorder        (sat_x.outer_scan(), sat_x.tail(), sat_x.full(0).split_var(), sat_x.inner(), sat_x.full(0))
            .gpu_threads    (sat_x.inner(0), sat_x.full(0).split_var())
            .gpu_blocks     (sat_x.full(0));
    }

    // slightly different schedule for y IIR filter
    {
        RecFilter sat_y  = SAT2y;

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
            .split          (sat_y.full(0), inter_tiles_per_warp)
            .reorder        (sat_y.outer_scan(), sat_y.tail(), sat_y.full(0).split_var(), sat_y.inner(), sat_y.full(0))
            .gpu_threads    (sat_y.inner(0), sat_y.full(0).split_var())
            .gpu_blocks     (sat_y.full(0));
    }

    {
        Box1.as_func().compute_root()
            .split  (u, xo, xi, tile_width)
            .split  (v, yo, yi, tile_width)
            .split  (yi,yi, yii,8)
            .unroll (yii)
            .reorder(yii,xi,yi,xo,yo)
            .gpu    (xo,yo,xi,yi);

        Box2x.as_func().compute_root()
            .split  (u, xo, xi, tile_width)
            .split  (v, yo, yi, tile_width)
            .split  (yi,yi, yii, 8)
            .unroll (yii)
            .reorder(yii,xi,yi,xo,yo)
            .gpu    (xo,yo,xi,yi);

        DoG.as_func().compute_root()
            .split  (u, xo, xi, tile_width)
            .split  (v, yo, yi, tile_width)
            .split  (yi,yi, yii,8)
            .unroll (yii)
            .reorder(yii,xi,yi,xo,yo)
            .gpu    (xo,yo,xi,yi);
    }
}
