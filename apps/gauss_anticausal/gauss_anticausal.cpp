#include <iostream>
#include <cstdio>
#include <cmath>

#include <Halide.h>

#include "../gaussian_weights.h"
#include "../../lib/split.h"

#define MAX_THREAD 192

using namespace Halide;

using std::vector;
using std::string;
using std::cerr;
using std::endl;

Image<float> reference_recursive_filter(Image<float> in, int order, float sigma);
Image<float> reference_gaussian(Image<float> in, float sigma);

std::pair<float, Image<float> > gaussian_weights(float sigma, int order) {
    Image<float> W(4,3);
    float b0, a1, a2;
    if (order == 1) {
        weights1(sigma, b0, a1);
        W(0,0) = -a1;
        W(1,0) = -a1;
        W(2,0) = -a1;
        W(3,0) = -a1;
    } else {
        weights2(sigma, b0, a1, a2);
        W(0,0) = -a1; W(0,1) = -a2;
        W(1,0) = -a1; W(1,1) = -a2;
        W(2,0) = -a1; W(2,1) = -a2;
        W(3,0) = -a1; W(3,1) = -a2;
    }
    return std::pair<float, Image<float> >(b0, W);
}


int main(int argc, char **argv) {
    Arguments args("gauss_anticausal", argc, argv);

    bool  verbose = args.verbose;
    bool  nocheck = args.nocheck;
    int   width   = args.width;
    int   height  = args.height;
    int   tile    = args.block;
    int   iterations = args.iterations;

    float sigma = 5.0f;

    Image<float> random_image = generate_random_image<float>(width,height);

    ImageParam image(type_of<float>(), 2);
    image.set(random_image);

    // ----------------------------------------------------------------------------------------------

    // recursive filter weights for Gaussian convolution
    int order = 2;
    float b0       = gaussian_weights(sigma, order).first;
    Image<float> W = gaussian_weights(sigma, order).second;

    RDom rx(0, image.width(), "rx");
    RDom ry(0, image.width(), "ry");
    RDom rz(0, image.height(),"rz");
    RDom rw(0, image.height(),"rw");

    Var x("x"), y("y");
    Var xi("xi"), yi("yi");
    Var xo("xo"), yo("yo");

    RDom rxi(0, tile, "rxi");
    RDom ryi(0, tile, "ryi");
    RDom rzi(0, tile, "rzi");
    RDom rwi(0, tile, "rwi");

    // bounds
    Expr iw = image.width()-1;
    Expr ih = image.height()-1;

    Func G("Gaussian");
    Func S("Result");

    // Gaussian convolution - 2D causal-anticausal filter
    // feed forward coeff multiplied with input to Gaussian filter
    G(x, y) = b0 * select((x<0 || y<0 || x>iw || y>ih), 0.0f, image(clamp(x,0,iw),clamp(y,0,ih)));
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

    S(x,y) = G(x,y);

    split(G,W,
            Internal::vec(  0,  0,  1,  1),
            Internal::vec(  x,  x,  y,  y),
            Internal::vec( xi, xi, yi, yi),
            Internal::vec( xo, xo, yo, yo),
            Internal::vec( rx, ry, rz, rw),
            Internal::vec(rxi,ryi,rzi,rwi),
            Internal::vec(order,order,order,order));

    inline_function(S, "Gaussian");

    vector<Func> func_list;
    extract_func_calls(S, func_list);

    map<string,Func> functions;
    for (size_t i=0; i<func_list.size(); i++) {
        cerr << func_list[i] << endl;
        func_list[i].compute_root();
        functions[func_list[i].name()] = func_list[i];
    }

    // realize it now just to check
    Target target = get_jit_target_from_environment();
    if (target.has_gpu_feature() || (target.features & Target::GPUDebug)) {
    }

    cerr << "\nJIT compilation ... " << endl;
    S.compile_jit();

    Buffer hl_out_buff(type_of<float>(), width,height);
    {
        Timer t("Running ... ");
        S.realize(hl_out_buff);
        hl_out_buff.free_dev_buffer();
    }
    hl_out_buff.copy_to_host();
    hl_out_buff.free_dev_buffer();

    // ----------------------------------------------------------------------------------------------

    if (!nocheck) {
        cerr << "\nChecking difference ...\n" << endl;
        Image<float> hl_out(hl_out_buff);
        Image<float> ref1 = reference_recursive_filter(random_image, order, sigma);
        Image<float> ref2 = reference_gaussian(random_image, sigma);
        cerr << "Difference with ref recursive filter \n" << CheckResult(ref1,hl_out) << endl;
        cerr << "Difference with true Gaussian \n" << CheckResultVerbose(ref2,hl_out) << endl;
    }

    return 0;
}

Image<float> reference_recursive_filter(Image<float> in, int order, float sigma) {
    int width = in.width();
    int height= in.height();

    float b0       = gaussian_weights(sigma, order).first;
    Image<float> W = gaussian_weights(sigma, order).second;

    int order_x = order;
    int order_y = order;

    Image<float> ref(width,height);

    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            ref(x,y) = b0 * in(x,y);
        }
    }

    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            for (int k=1; k<=order_x; k++) {
                ref(x,y) += (x>=k ? W(0,k-1)*ref(x-k,y) : 0.0f);
            }
        }
    }

    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            for (int k=1; k<=order_y; k++) {
                ref(x,y) += (y>=k ? W(1,k-1)*ref(x,y-k) : 0.0f);
            }
        }
    }

    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            for (int k=1; k<=order_x; k++) {
                ref(width-1-x,y) += (x>=k ? W(2,k-1)*ref(width-1-(x-k),y) : 0.0f);
            }
        }
    }

    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            for (int k=1; k<=order_y; k++) {
                ref(x,height-1-y) += (y>=k ? W(3,k-1)*ref(x,height-1-(y-k)) : 0.0f);
            }
        }
    }

    return ref;
}

Image<float> reference_gaussian(Image<float> in, float sigma) {
    int width = in.width();
    int height= in.height();
    Image<float> ref(width,height);
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            float a = 0.0f;
            float w = 0.0f;
            for (int j=0; j<height; j++) {
                for (int i=0; i<width; i++) {
                    float d = (x-i)*(x-i) + (y-j)*(y-j);
                    float g = gaussian(std::sqrt(d), 0.0f, sigma);
                    a += g * in(i,j);
                    w += g;
                }
            }
            ref(x,y) = a/w;
        }
    }
    return ref;
}
