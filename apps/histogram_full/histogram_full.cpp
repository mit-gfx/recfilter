#include <iostream>
#include <cstdio>
#include <stdexcept>
#include <Halide.h>
#include <common_inc/timer.h>
#include <common_inc/utils.h>

using namespace Halide;

class Args {                        ///< Command line arguments
public:
    int  width;                     ///< width exponent, actual image width = 2^width
    int  block;                     ///< block exponent, actual block size = 2^block
    Args(int argc, char** argv);
};

int main(int argc, char **argv) {
    Args args(argc, argv);

    int BLOCK = (1<<args.block);
    int WIDTH = (1<<args.width);
    int BINS  = 10;

    // ----------------------------------------------------------------------------------------------

    std::cerr << "\nCreating random image ... " << std::endl;

    Image<int> image(WIDTH, WIDTH);
    Image<int> ref_hist(BINS);
    Image<int> hl_out(BINS);

    srand(0);
    for (int y=0; y<WIDTH; y++) {
        for (int x=0; x<WIDTH; x++) {
            int r = std::min(std::max(0,rand()%BINS), BINS-1);
            image(x,y) = r;
            ref_hist(r) += 1;
        }
    }

    // ----------------------------------------------------------------------------------------------

    int numThreadsX = std::max(1, std::max(32, WIDTH/BLOCK));
    int numThreadsY = std::max(1, std::max(8, WIDTH/BLOCK));

    // ----------------------------------------------------------------------------------------------

    ///
    /// Func Hist("Hist");
    /// Var x("x");
    /// Var p("p"), q("q");
    ///
    /// RDom r(0, WIDTH, 0, WIDTH);
    ///
    /// Hist(x) = 0;
    /// Hist(clamp(image(r.x,r.y), 0, BINS-1)) += 1;
    /// Hist.cuda_tile(x, BINS);                                // applied on base case of both hInner and hOuter
    ///
    /// Hist.update().split(r, rOuter, rInner, p, q);
    /// Hist.update()
    ///     .reorder(Var(rInner.x.name()),Var(rInner.y.name()), p, q) // applied on update case of hInner
    ///     .reorder(Var(rOuter.x.name()),Var(rOuter.y.name()), x)    // applied on update case of hOuter
    ///     .cuda_tile(p, q, numThreads, numThreads);                 // applied on update case of hOuter
    ///     .cuda_tile(x, BINS);                                      // applied on update case of hInner
    ///

    Func hInner("hist_inner");
    Func hOuter("hist_outer");

    Var x("x"), p("p"), q("q");

    RDom rInner(0, BLOCK, 0, BLOCK);
    RDom rOuter(0, WIDTH/BLOCK, 0, WIDTH/BLOCK);

    hInner(p,q,x) = 0;
    hInner(p,q,clamp(image(p*BLOCK+rInner.x, q*BLOCK+rInner.y), 0, BINS-1)) += 1;
    hInner.compute_root();
    hInner.update().reorder(Var(rInner.x.name()), Var(rInner.y.name()), p, q);
    hInner.cuda_tile(x, BINS);
    hInner.update().cuda_tile(p, q, numThreadsX, numThreadsY);

    hOuter(x) = 0;
    hOuter(x) += hInner(rOuter.x, rOuter.y, x);
    hOuter.compute_root();
    hInner.cuda_tile(x, BINS);
    hOuter.update().reorder(Var(rOuter.x.name()), Var(rOuter.y.name()), x);
    hInner.update().cuda_tile(x, BINS);
    hOuter.bound(x, 0, BINS);

    // ----------------------------------------------------------------------------------------------

    ///
    /// JIT compilation
    ///

    std::cerr << "\nJIT compilation ... " << std::endl;
    hOuter.compile_jit();

    std::cerr << "Running ... " << std::endl;
    hl_out = hOuter.realize(BINS);

    // ----------------------------------------------------------------------------------------------

    std::cerr << "Reference histogram with 10 bins" << std::endl << ref_hist << std::endl;
    std::cerr << "Halide histogram with 10 bins" << std::endl << hl_out << std::endl;

    return EXIT_SUCCESS;
}

Args::Args(int argc, char** argv) : width(10), block(5) {
    std::string desc(
            "\nUsage\n ./histogram_full [-width|-w] [-block|-b] [-help]\n\n"
            "\twidth\t  exponent to decide image width = 2^w [default = 2^10] \n"
            "\tblock\t  exponent to decide block size  = 2^b [default = 2^5] \n"
            "\thelp\t   show help message\n"
            );


    try {
        for (int i=1; i<argc; i++) {
            std::string option = argv[i];

            if (!option.compare("-help") || !option.compare("--help")) {
                throw std::runtime_error("Showing help message");
            }

            else if (!option.compare("-b") || !option.compare("--b") || !option.compare("-block") || !option.compare("--block")) {
                if ((i+1) < argc)
                    block = atoi(argv[++i]);
                else
                    throw std::runtime_error("-block requires a int value");
            }

            else if (!option.compare("-w") || !option.compare("--w") || !option.compare("-width") || !option.compare("--width")) {
                if ((i+1) < argc)
                    width = atoi(argv[++i]);
                else
                    throw std::runtime_error("-width requires a int value");
            }

            else {
                throw std::runtime_error("Bad command line option " + option);
            }
        }

        if (block > width) throw std::runtime_error("block cannot be larger than image width");

    } catch (std::runtime_error& e) {
        std::cerr << std::endl << e.what() << std::endl << desc << std::endl;
        exit(EXIT_FAILURE);
    }
}
