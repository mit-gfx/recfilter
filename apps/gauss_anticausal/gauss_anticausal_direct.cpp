#include <iostream>
#include <cstdio>
#include <cmath>

#include <Halide.h>

#include "../../lib/gaussian_weights.h"
#include "../../lib/split.h"

#define MAX_THREAD 192

using namespace Halide;

using std::vector;
using std::string;
using std::cerr;
using std::endl;

Image<float> reference_gaussian(Image<float> in, float sigma);

int main(int argc, char **argv) {
    Arguments args("gauss_anticausal_cascaded", argc, argv);

    bool nocheck = args.nocheck;
    int  width   = args.width;
    int  height  = args.width;
    int  tile    = args.block;

    float sigma = 16.0f;

    Image<float> random_image = generate_random_image<float>(width,height);

    ImageParam image(type_of<float>(), 2);
    image.set(random_image);

    // ----------------------------------------------------------------------------------------------

    // recursive filter weights for Gaussian convolution
    RDom rx(0, image.width(), "rx");
    RDom ry(0, image.width(), "ry");
    RDom rz(0, image.height(),"rz");
    RDom rw(0, image.height(),"rw");

    Var x ("x") , y ("y");
    Var xi("xi"), yi("yi");
    Var xo("xo"), yo("yo");

    RDom rxi(0, tile, "rxi");
    RDom ryi(0, tile, "ryi");
    RDom rzi(0, tile, "rzi");
    RDom rwi(0, tile, "rwi");

    // bounds
    Expr iw = image.width()-1;
    Expr ih = image.height()-1;

    Func G("G");
    Func S("Result");

    // Third order filter
    {
        int order       = 3;
        int num_scans   = 4;
        Image<float> B  = gaussian_weights(sigma, order, num_scans).first;
        Image<float> W  = gaussian_weights(sigma, order, num_scans).second;

        G(x, y) = image(clamp(x,0,iw), clamp(y,0,ih));

        G(rx,y) = B(0)*G(rx,y)
           + W(0,0)*G(max(0,rx-1),y)
           + W(0,1)*G(max(0,rx-2),y)
           + W(0,2)*G(max(0,rx-3),y);

        G(iw-ry,y) = B(1)*G(iw-ry,y)
           + W(1,0)*G(min(iw,iw-ry+1),y)
           + W(1,1)*G(min(iw,iw-ry+2),y)
           + W(1,2)*G(min(iw,iw-ry+3),y);

        G(x,rz) = B(2)*G(x,rz)
           + W(2,0)*G(x,max(0,rz-1))
           + W(2,1)*G(x,max(0,rz-2))
           + W(2,2)*G(x,max(0,rz-3));

        G(x,ih-rw) = B(3)*G(x,ih-rw)
           + W(3,0)*G(x,min(ih,ih-rw+1))
           + W(3,1)*G(x,min(ih,ih-rw+2))
           + W(3,2)*G(x,min(ih,ih-rw+3));

        S(x,y) = G(x,y);

        split(G,B,W,
                Internal::vec(  0,  0,  1,  1),
                Internal::vec(  x,  x,  y,  y),
                Internal::vec( xi, xi, yi, yi),
                Internal::vec( xo, xo, yo, yo),
                Internal::vec( rx, ry, rz, rw),
                Internal::vec(rxi,ryi,rzi,rwi),
                Internal::vec(order,order,order,order));

        inline_function(S, "G");
    }


    // ----------------------------------------------------------------------------------------------

    map<string,Func> functions = extract_func_calls(S);
    map<string,Func>::iterator f    = functions.begin();
    map<string,Func>::iterator fend = functions.end();
    for (; f!=fend; f++) {
        cerr << f->second << endl;
        f->second.compute_root();
    }

    // ----------------------------------------------------------------------------------------------

    Target target = get_jit_target_from_environment();
    if (target.has_gpu_feature() || (target.features & Target::GPUDebug)) {
    }

    // ----------------------------------------------------------------------------------------------

    cerr << "\nJIT compilation ... " << endl;
    S.compile_jit();

    Buffer hl_out_buff(type_of<float>(), width,height);
    {
        Timer t("Running ... ");
        S.realize(hl_out_buff);
    }
    hl_out_buff.copy_to_host();
    hl_out_buff.free_dev_buffer();

    // ----------------------------------------------------------------------------------------------

    if (!nocheck) {
        cerr << "\nChecking difference ...\n" << endl;
        Image<float> hl_out(hl_out_buff);
        Image<float> ref = reference_gaussian(random_image, sigma);
        cerr << "Difference with true Gaussian \n" << CheckResult(ref,hl_out) << endl;
    }

    return 0;
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
