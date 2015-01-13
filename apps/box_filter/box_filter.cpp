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
using std::stringstream;
using std::cerr;
using std::endl;

template<typename T>
void check_result(RecFilter, Image<T> image);

RecFilter integral_image(int order, int width, int height, int tile_width, ImageParam I);
RecFilter derivative_image(int order, int filter_radius, int width, int height, int tile_width, Func I);

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

    stringstream s;
    s << "box_filter_" << filter_reps << ".perflog";

    Log log(s.str());
    log << "Width\tOurs" << endl;

    // Profile the filter for all image widths
    for (int in_w=min_w; in_w<=max_w; in_w+=inc_w) {
        Image<float> image = generate_random_image<float>(in_w,in_w);
        ImageParam I;
        I.set(image);

        RecFilter S = integral_image  (filter_reps, in_w, in_w, tile_width, I);
        RecFilter D = derivative_image(filter_reps, box_filter_radius, in_w, in_w, tile_width, S.as_func());

        float time = D.profile(iter);

        cerr << in_w << " " << time << " ms" << endl;
        log  << in_w << " " << RecFilter::throughput(time,in_w*in_w) << endl;

        if (!nocheck) {
            check_result(D, image);
        }
    }

    return EXIT_SUCCESS;
}

RecFilter integral_image(int order, int width, int height, int tile_width, ImageParam I) {
    vector<float> coeff = integral_image_coeff(order);

    RecFilterDim x("x", width);
    RecFilterDim y("y", height);

    RecFilter S("S");

    S(x,y) = I(x,y);                // k order intergral image
    S.add_filter(x, coeff);         // k = num of box filter iterations
    S.add_filter(y, coeff);

    S.split(x, tile_width, y, tile_width);

    // -------------------------------------------------------------------------

    int tiles_per_warp = 2;
    int unroll_w       = 8;

    S.intra_schedule(1).compute_locally()
        .reorder_storage(S.inner(), S.outer())
        .unroll         (S.inner_scan())
        .split          (S.inner(1), unroll_w)
        .unroll         (S.inner(1).split_var())
        .reorder        (S.inner_scan(), S.inner(1).split_var(), S.inner(), S.outer())
        .gpu_threads    (S.inner(0), S.inner(1))
        .gpu_blocks     (S.outer(0), S.outer(1));

    S.intra_schedule(2).compute_locally()
        .reorder_storage(S.tail(), S.inner(), S.outer())
        .unroll         (S.inner_scan())
        .split          (S.outer(0), tiles_per_warp)
        .reorder        (S.inner_scan(), S.tail(), S.outer(0).split_var(), S.inner(), S.outer())
        .gpu_threads    (S.inner(0), S.outer(0).split_var())
        .gpu_blocks     (S.outer(0), S.outer(1));

    S.inter_schedule().compute_globally()
        .reorder_storage(S.inner(), S.tail(), S.outer())
        .unroll         (S.outer_scan())
        .split          (S.outer(0), tiles_per_warp)
        .reorder        (S.outer_scan(), S.tail(), S.outer(0).split_var(), S.inner(), S.outer())
        .gpu_threads    (S.inner(0), S.outer(0).split_var())
        .gpu_blocks     (S.outer(0));

    return S;
}

RecFilter derivative_image(int order, int filter_radius, int width, int height, int tile_width, Func I) {
    if (order>4) {
        RecFilter F = derivative_image(3, 1, width, height, tile_width, I);
        return derivative_image(order-3, 1, width, height, tile_width, F.as_func());
    }
    else {

        int B = filter_radius;
        int z = (2*B+1)*(2*B+1);

        Var xo("xo"), xi("xi"), xii("xii");
        Var yo("yo"), yi("yi"), yii("yii");

        RecFilterDim x("x", width);
        RecFilterDim y("y", height);

        Var u = x.var();
        Var v = y.var();

        RecFilter F("Box");

        if (order==1) {
            F(x,y) = (I(min(x+B,width-1), min(y+B,height-1))
                    + I(max(x-B-1,0),     max(y-B-1,0))
                    - I(min(x+B,width-1), max(y-B-1,0))
                    - I(max(x-B-1,0),     min(y+B,height-1))) / z;
        } else if (order==2) {

        } else if (order==3) {

        } else {
            cerr << "Invalid order" << endl;
            assert(false);
        }

        F.compute_globally()
            .split(u, xo, xi, tile_width)
            .split(v, yo, yi, tile_width)
            .split(yi,yi, yii,4).unroll(yii)
            .reorder(yii,xi,yi,xo,yo)
            .gpu(xo,yo,xi,yi);

        return F;
    }
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

    cerr << CheckResultVerbose<T>(ref,hl_out) << endl;

}
