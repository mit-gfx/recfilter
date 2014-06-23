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
    int  height  = args.height;
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

    Func G1("G1");
    Func G2("G2");
    Func G1_result("G1_result");
    Func G2_result("G2_result");
    Func S("Result");

    // First order filter
    {
        int order       = 1;
        int num_scans   = 4;
        Image<float> B  = gaussian_weights(sigma, order, num_scans).first;
        Image<float> W  = gaussian_weights(sigma, order, num_scans).second;

        //B(0)=2;W(0,0)=1;
        //B(1)=1;W(1,0)=0;
        //B(2)=2;W(2,0)=1;
        //B(3)=1;W(3,0)=0;

        G1(x, y) = image(clamp(x,0,iw), clamp(y,0,ih));
        G1(rx,y) = B(0)*G1(rx,y) + W(0,0)*G1(max(0,rx-1),y);
        G1(ry,y) = B(1)*G1(ry,y) + W(1,0)*G1(max(0,ry-1),y);
        G1(x,rz) = B(2)*G1(x,rz) + W(2,0)*G1(x,max(0,rz-1));
        G1(x,rw) = B(3)*G1(x,rw) + W(3,0)*G1(x,max(0,rw-1));
        //G1(rx,y)    = B(0)*G1(rx,y)    + W(0,0)*G1(max(0,rx-1),y);
        //G1(iw-ry,y) = B(1)*G1(iw-ry,y) + W(1,0)*G1(min(iw,iw-ry+1),y);
        //G1(x,rz)    = B(2)*G1(x,rz)    + W(2,0)*G1(x,max(0,rz-1));
        //G1(x,ih-rw) = B(3)*G1(x,ih-rw) + W(3,0)*G1(x,min(ih,ih-rw+1));

        G1_result(x,y) = G1(x,y);

        split(G1,B,W,
                Internal::vec(  0,  0,  1,  1),
                Internal::vec(  x,  x,  y,  y),
                Internal::vec( xi, xi, yi, yi),
                Internal::vec( xo, xo, yo, yo),
                Internal::vec( rx, ry, rz, rw),
                Internal::vec(rxi,ryi,rzi,rwi),
                Internal::vec(order,order,order,order));

//        inline_function(G1_result, "G1");
    }

    // Second order filter
    {
        int order       = 2;
        int num_scans   = 4;
        Image<float> B  = gaussian_weights(sigma, order, num_scans).first;
        Image<float> W  = gaussian_weights(sigma, order, num_scans).second;

        G2(x, y) = G1_result(x,y);

        G2(rx,y) = B(0)*G2(rx,y)
            + W(0,0)*G2(max(0,rx-1),y)
            + W(0,1)*G2(max(0,rx-2),y);

        G2(iw-ry,y) = B(1)*G2(iw-ry,y)
            + W(1,0)*G2(min(iw,iw-ry+1),y)
            + W(1,1)*G2(min(iw,iw-ry+2),y);

        G2(x,rz) = B(2)*G2(x,rz)
            + W(2,0)*G2(x,max(0,rz-1))
            + W(2,1)*G2(x,max(0,rz-2));

        G2(x,ih-rw) = B(3)*G2(x,ih-rw)
            + W(3,0)*G2(x,min(ih,ih-rw+1))
            + W(3,1)*G2(x,min(ih,ih-rw+2));

        G2_result(x,y) = G2(x,y);

//        split(G2,B,W,
//                Internal::vec(  0,  0,  1,  1),
//                Internal::vec(  x,  x,  y,  y),
//                Internal::vec( xi, xi, yi, yi),
//                Internal::vec( xo, xo, yo, yo),
//                Internal::vec( rx, ry, rz, rw),
//                Internal::vec(rxi,ryi,rzi,rwi),
//                Internal::vec(order,order,order,order));

//        inline_function(G2_result, "G2");
    }

    S(x,y) = G1_result(x,y);

    // ----------------------------------------------------------------------------------------------

//    inline_function(S, "G1_result");
//    inline_function(S, "G2_result");

    vector<Func> func_list;
    extract_func_calls(S, func_list);

    map<string,Func> functions;
    for (size_t i=0; i<func_list.size(); i++) {
        cerr << func_list[i] << endl;
        func_list[i].compute_root();
        functions[func_list[i].name()] = func_list[i];

        if (func_list[i].name() == "G1-Intra") {
            Func f;
            f(x,y,xo) = func_list[i](x%4, x/4, y%4, y/4, xo);
            Image<float> a = f.realize(width, height, 4);
            cerr << a << endl;
        }
        if (func_list[i].name().find("Tail_x") != string::npos) {
            Func f;
            f(x,y) = func_list[i](0, x, y%4, y/4);
            Image<float> a = f.realize(width/tile, height);
            cerr << a << endl;
        }
        if (func_list[i].name().find("Tail_y") != string::npos) {
            Func f;
            f(x,y) = func_list[i](x%4, x/4, 0, y);
            Image<float> a = f.realize(width, height/tile);
            cerr << a << endl;
        }
        if (func_list[i].name().find("Deps") != string::npos) {
            Func f;
            f(x,y) = func_list[i](x%4, x/4, y%4, y/4);
            Image<float> a = f.realize(width, height);
            cerr << a << endl;
        }
        if (func_list[i].name() == "G1-Final-Sub") {
            Func f;
            f(x,y) = func_list[i](x%4, x/4, y%4, y/4);
            Image<float> a = f.realize(width, height);
            cerr << a << endl;
        }
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
        hl_out_buff.free_dev_buffer();
    }
    hl_out_buff.copy_to_host();
    hl_out_buff.free_dev_buffer();

    // ----------------------------------------------------------------------------------------------

    if (!nocheck) {
        cerr << "\nChecking difference ...\n" << endl;
        Image<float> hl_out(hl_out_buff);
        Image<float> ref = reference_gaussian(random_image, sigma);
        cerr << "Difference with true Gaussian \n" << CheckResultVerbose(ref,hl_out) << endl;
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
