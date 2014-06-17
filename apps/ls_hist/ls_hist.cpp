#include <iostream>
#include <Halide.h>

#include "../../lib/split.h"

#include "../gaussian_weights.h"

#define MAX_THREAD 192

using namespace Halide;

using std::vector;
using std::string;
using std::cerr;
using std::endl;

#define NUM_BINS       (15)
#define BIN_WIDTH      (1.0f/float(NUM_BINS))
#define BIN_CENTER(i)  (BIN_WIDTH*(float(i)+0.5f))
#define GAUSS_SIGMA    (BIN_WIDTH)

std::pair<float, Image<float> > gaussian_weights(float sigma, int order) {
    Image<float> W(4,3);
    float b0, a1, a2;
    if (order == 1) {
        weights1(sigma, b0, a1);
        W(0,0) = a1;
        W(1,0) = a1;
        W(2,0) = a1;
        W(3,0) = a1;
    } else {
        weights2(sigma, b0, a1, a2);
        W(0,0) = a1; W(0,1) = a2;
        W(1,0) = a1; W(1,1) = a2;
        W(2,0) = a1; W(2,1) = a2;
        W(3,0) = a1; W(3,1) = a2;
    }
    return std::pair(b0, W);
}


int main(int argc, char **argv) {
    Arguments args("lshist", argc, argv);

    bool  verbose = args.verbose;
    bool  nocheck = args.nocheck;
    int   width   = args.width;
    int   height  = args.height;
    int   tile_width = args.block;
    int   iterations = args.iterations;

    Image<float> random_image = generate_random_image<float>(width,height);

    // pass the image through the lookup tables
    vector<ImageParam> hist_image
    for (int i=0; i<NUM_BINS; i++) {
        float mu = BIN_CENTER(i);
        float sigma = GAUSS_SIGMA;
        Image image_bin_i(image.width(), image.height());
        for (int x=0; x<image.width(); x++) {
            for (int y=0; y<image.height(); y++) {
                float c = random_image(x,y);
                image_bin_i(x,y) = gaussIntegral(c, mu, sigma);
            }
        }

        ImageParam h_image(type_of<float>(),2);
        h_image.set(image_bin_i);
        hist_image.push_back(h_image);
    }

    ImageParam image(type_of<float>(), 2);
    image.set(random_image);

    // ----------------------------------------------------------------------------------------------

    Var x("x"), y("y");

    // recursive filter wieghts for Gaussian convolution
    int order = 2;

    float b0       = gaussian_weights(sigma, order).first;
    Image<float> W = gaussian_weights(sigma, order).second;

    RDom rx(0, image.width(),"rx");
    RDom ry(0, image.width(),"ry");
    RDom rz(0, image.height(),"rz");
    RDom rw(0, image.height(),"rw");

    Var xi("xi"), yi("yi");
    Var xo("xo"), yo("yo");

    RDom rxi(0, tile, "rxi");
    RDom ryi(0, tile, "ryi");
    RDom rzi(0, tile, "rzi");
    RDom rwi(0, tile, "rwi");

    // bounds
    Expr iw = image.width()-1;
    Expr ih = image.height()-1;

    vector<Func> Gaussians;

    for (int i=0; i<NUM_BINS; i++) {
        string name = Internal::int_to_string(i);
        Func G("G" + name);
        Func L("L" + name);

        // pass image through historgam lookup table
        L(x,y) = select((x<0 || y<0 || x>iw || y>ih), 0.0f,
                hist_image[i](clamp(x,0,iw),clamp(y,0,ih)));

        // Gaussian convolution - 2D causal-anticausal filter
        // feed forward coeff multiplied with input to Gaussian filter
        G(x, y) = b0 * I(x,y);
        G(rx,y) = G(rx,y)
            + select(rx>0, W(0,0)*G(max(0,rx-1),y), 0.0f)
            + select(rx>1, W(0,1)*G(max(0,rx-2),y), 0.0f);
        G(iw-ry,y) = G(iw-ry,y)
            + select(ry>0, W(1,0)*G(min(iw,iw-ry+1),y), 0.0f)
            + select(ry>1, W(1,1)*G(min(iw,iw-ry+2),y), 0.0f);
        G(x,rz) = G(x,rz)
            + select(rz>0, W(2,0)*G(x,max(0,rz-1)), 0.0f)
            + select(rz>1, W(2,1)*G(x,max(0,rz-2)), 0.0f);
        G(x,ih-rw) = G(x,ih-rw)
            + select(rw>0, W(3,0)*G(x,min(ih,ih-rw+1)), 0.0f)
            + select(rw>1, W(3,1)*G(x,min(ih,ih-rw+2)), 0.0f);

        split(G,W,
                Internal::vec(  0,  0,  1,  1),
                Internal::vec(  x,  x,  y,  y),
                Internal::vec( xi, xi, yi, yi),
                Internal::vec( xo, xo, yo, yo),
                Internal::vec( rx, ry, rz, rw),
                Internal::vec(rxi,ryi,rzi,rwi),
                Internal::vec( fx, fx, fy, fy));

        vector<Func> func_list;
        extract_func_calls(G, func_list);

        map<string,Func> functions;
        for (size_t i=0; i<func_list.size(); i++) {
            cerr << func_list[i] << endl;
            func_list[i].compute_root();
            functions[func_list[i].name()] = func_list[i];
        }

        // add to the overall list
        Gaussians.push_back(G);

        // realize it now just to check
        Target target = get_jit_target_from_environment();
        if (target.has_gpu_feature() || (target.features & Target::GPUDebug)) {
        }

        cerr << "\nJIT compilation ... " << endl;
        G.compile_jit();

        Buffer hl_out_buff(type_of<int>(), width,height);
        {
            Timer t("Running ... ");
            G.realize(hl_out_buff);
            hl_out_buff.free_dev_buffer();
        }
        hl_out_buff.copy_to_host();
        hl_out_buff.free_dev_buffer();
    }


    // ----------------------------------------------------------------------------------------------

    if (!nocheck) {
        cerr << "\nChecking difference ... " << endl;
        Image<float> hl_out(hl_out_buff);
    }

    return 0;
}
