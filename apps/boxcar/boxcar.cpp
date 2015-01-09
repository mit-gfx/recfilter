/**
 * \file box_filter.cpp
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
#include "gaussian_weights.h"

using namespace Halide;

using std::vector;
using std::string;
using std::cerr;
using std::cout;
using std::endl;

template<typename T>
void check_result(RecFilter, Image<T> image);

template<typename T>
RecFilter box_filter (int iterations, int filter_radius, int width, int height, int tile_width, Image<T> I);
RecFilter box_filter (int iterations, int filter_radius, int width, int height, int tile_width, RecFilter I);
RecFilter box_filter (int iterations, int filter_radius, int width, int height, int tile_width, Func I);
Expr finite_diff_expr(int iterations, int filter_radius, int width, int height, Func I,      Expr x, Expr y);

// -----------------------------------------------------------------------------

int main(int argc, char **argv) {
    const int box_filter_radius = 10;

    Arguments args(argc, argv);
    bool nocheck   = args.nocheck;
    int iter       = args.iterations;
    int tile_width = args.block;
    int min_w      = args.min_width;
    int max_w      = args.max_width;
    int inc_w      = tile_width;
    int filter_reps= args.filter_reps;

    // Profile the filter for all image widths
    for (int in_w=min_w; in_w<=max_w; in_w+=inc_w) {
        Image<float> I = generate_random_image<float>(in_w,in_w);

        RecFilter F = box_filter(filter_reps, box_filter_radius, in_w, in_w, tile_width, I);

        float time = F.profile(iter);

        cerr << "Width " << in_w << " " << time << " ms" << endl;

        if (!nocheck) {
            check_result(F, I);
        }
    }

    return EXIT_SUCCESS;
}

RecFilter box_filter(int iterations, int filter_radius, int width, int height, int tile_width, Func I) {
    vector<float> coeff = integral_image_coeff(iterations);

    int B = filter_radius;

    RecFilterDim x("x", width);
    RecFilterDim y("y", height);

    Var xo("xo"), xi("xi"), xii("xii");
    Var yo("yo"), yi("yi"), yii("yii");

    Var u = x.var();
    Var v = y.var();

    RecFilter S("S");
    RecFilter F("Box");

    S(x,y) = I(x,y);                // k order intergral image
    S.add_filter(x, coeff);         // k = num of box filter iterations
    S.add_filter(y, coeff);

    F(x,y) = finite_diff_expr(iterations, B, width, height, S.as_func(), x, y);

    S.split(x, tile_width, y, tile_width);

    // -------------------------------------------------------------------------

    int tiles_per_warp = 2;
    int unroll_w       = 8;

    if (!F.target().has_gpu_feature()) {
        cerr << "\nOnly GPU schedule available" << endl;
        exit(EXIT_FAILURE);
    }

    F.compute_globally()
        .split(u, xo, xi, 32)
        .split(v, yo, yi, 32)
        .split(yi,yi, yii,4).unroll(yii)
        .reorder(yii,xi,yi,xo,yo)
        .gpu(xo,yo,xi,yi);

    S.intra_schedule(1).compute_locally()
        .reorder_storage(F.inner(), F.outer())
        .unroll         (F.inner_scan())
        .split          (F.inner(1), unroll_w)
        .unroll         (F.inner(1).split_var())
        .reorder        (F.inner_scan(), F.inner(1).split_var(), F.inner(), F.outer())
        .gpu_threads    (F.inner(0), F.inner(1))
        .gpu_blocks     (F.outer(0), F.outer(1));

    S.intra_schedule(2).compute_locally()
        .reorder_storage(F.tail(), F.inner(), F.outer())
        .unroll         (F.inner_scan())
        .split          (F.outer(0), tiles_per_warp)
        .reorder        (F.inner_scan(), F.tail(), F.outer(0).split_var(), F.inner(), F.outer())
        .gpu_threads    (F.inner(0), F.outer(0).split_var())
        .gpu_blocks     (F.outer(0), F.outer(1));

    S.inter_schedule().compute_globally()
        .reorder_storage(F.inner(), F.tail(), F.outer())
        .unroll         (F.outer_scan())
        .split          (F.outer(0), tiles_per_warp)
        .reorder        (F.outer_scan(), F.tail(), F.outer(0).split_var(), F.inner(), F.outer())
        .gpu_threads    (F.inner(0), F.outer(0).split_var())
        .gpu_blocks     (F.outer(0));

    return F;
}

Expr finite_diff_expr(int iterations, int filter_radius, int width, int height, Func I, Expr x, Expr y) {
    int B = filter_radius;
    int z = (2*B+1)*(2*B+1);

    Expr e = (I(min(x+B,width-1), min(y+B,height-1))
            + I(max(x-B-1,0),     max(y-B-1,0))
            - I(min(x+B,width-1), max(y-B-1,0))
            - I(max(x-B-1,0),     min(y+B,height-1))) / z;

    return e;
}

template<typename T>
RecFilter box_filter(int iterations, int filter_radius, int width, int height, int tile_width, Image<T> I) {
    Var x,y;
    Func f;
    f(x,y) = I(x,y);
    return box_filter(iterations, filter_radius, width, height, tile_width, f);
}

RecFilter box_filter(int iterations, int filter_radius, int width, int height, int tile_width, RecFilter I) {
    return box_filter(iterations, filter_radius, width, height, tile_width, I.as_func());
}

template<typename T>
void check_result(RecFilter F, Image<T> I) {
    int width  = I.width();
    int height = I.height();

    Realization out = F.realize();

    Image<T> hl_out(out);
    Image<T> ref(width, height);

    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            ref(x,y) = I(x,y);
        }
    }
    for (int y=0; y<height; y++) {
        for (int x=1; x<width; x++) {
            ref(x,y) += ref(x-1,y);
        }
    }
    for (int y=1; y<height; y++) {
        for (int x=0; x<width; x++) {
            ref(x,y) += ref(x,y-1);
        }
    }

    cout << CheckResultVerbose<T>(ref,hl_out) << endl;
}
