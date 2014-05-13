#include <iostream>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <Halide.h>
#include <common_inc/utils.h>

using namespace Halide;

class Args {                        ///< Command line arguments
public:
    bool nocheck;                   ///< skip checking Halide output with reference solution
    bool verbose;                   ///< display input, reference output, halide output, difference
    int  width;                     ///< width exponent, actual image width = 2^width
    int  block;                     ///< block exponent, actual block width = 2^block
    Args(int argc, char** argv);
};



int main(int argc, char **argv) {
    Args args(argc, argv);

    int BLOCK = (1<<args.block);
    int WIDTH = (1<<args.width);

    bool nocheck = args.nocheck ;
    bool verbose = args.verbose;

    // ----------------------------------------------------------------------------------------------

    std::cerr << "\nCreating random array ... " << std::endl;

    Image<int> image(WIDTH);
    Image<int> ref(WIDTH);
    for (int x=0; x<WIDTH; x++)
        image(x) = 1 + int(rand() % 9);

    // ----------------------------------------------------------------------------------------------

    ///
    /// Func Scan("Scan");
    ///
    /// Var x("x"), p("p");
    /// RDom r(0, WIDTH);
    /// RDOM rInner, rOuter;
    ///
    /// Scan(x) = 0;
    /// Scan(r) = Scan(r-1) + select(x<0, 0, image(clamp(x,0,WIDTH-1)));
    ///
    /// Scan.compute_root().cuda_tile(x, BLOCK);
    /// Scan.update().split(r, rInner, rOuter, p, BLOCK).cuda_tile(p, BLOCK);
    ///


    Var x("x"), p("p");

    Func ScanInner ("ScanInner");
    Func ScanOuter ("ScanOuter");
    Func Scan      ("Scan");

    RDom rInner(0, BLOCK);
    RDom rOuter(0, WIDTH/BLOCK);

    ScanInner(p,x)      = 0;
    ScanInner(p,rInner) = select(BLOCK*p+rInner<0, 0, image(clamp(BLOCK*p+rInner, 0, WIDTH-1)))
                        + ScanInner(p, rInner-1);

    ScanOuter(p,x)      = 0;
    ScanOuter(rOuter,x) = select(rOuter<0, 0, ScanInner(rOuter, x))
                        + ScanOuter(rOuter-1, x);

    Scan(x) = ScanInner(x/BLOCK, x%BLOCK) + ScanOuter(x/BLOCK-1, BLOCK-1);

    ScanInner.compute_root();
    ScanOuter.compute_root();
    Scan     .compute_root();

    ScanInner.reorder(x, p);
    ScanInner.cuda_tile(p, std::min(WIDTH/BLOCK, BLOCK));
    ScanInner.update().reorder(Var(rInner.x.name()), p);
    ScanInner.update().cuda_tile(p, std::min(WIDTH/BLOCK, BLOCK));

    Scan.cuda_tile(x, std::min(WIDTH/BLOCK, BLOCK));


    // ----------------------------------------------------------------------------------------------

    ///
    /// JIT compilation
    ///

    std::cerr << "\nJIT compilation ... " << std::endl;
    Scan.compile_jit();

    std::cerr << "Running ... " << std::endl;
    Image<int> hl_out = Scan.realize(WIDTH);

    // ----------------------------------------------------------------------------------------------

    ///
    /// Check the difference
    ///

    if (!nocheck || verbose) {
        for (int x=0; x<WIDTH; x++)
            ref(x) = image(x);
        for (int x=0; x<WIDTH; x++)
            ref(x) = ref(x) + (x>0 ? ref(x-1) : 0);

        Image<int> diff(WIDTH);
        int diff_sum = 0;
        for (int x=0; x<WIDTH; x++) {
            diff(x) = ref(x) - hl_out(x);
            diff_sum += diff(x);
        }

        if (verbose) {
            std::cerr << "Input" << std::endl << image << std::endl;
            std::cerr << "Reference" << std::endl << ref << std::endl;
            std::cerr << "Halide output" << std::endl << hl_out << std::endl;
            std::cerr << "Difference" << std::endl << diff << std::endl;
        } else {
            std::cerr << "\nDifference = " << diff_sum << std::endl;
            std::cerr << std::endl;
        }
    }

    return EXIT_SUCCESS;
}


Args::Args(int argc, char** argv) : nocheck(false), verbose(false), width(12), block(5) {
    std::string desc(
            "\nUsage\n ./scan [-width|-w] [-nocheck] [-verbose|-v] [-help]\n\n"
            "\twidth\t  exponent to decide width of image, width = 2^w [default 2^12] \n"
            "\tblock\t  exponent to decide block size  = 2^b [default = 2^5] \n"
            "\tnocheck\t  skip checking Halide result with reference solution [default = false]\n"
            "\tverbose\t  print input image, refernce output, halide output and difference [default = false]\n"
            "\thelp\t  show help message\n"
            );


    try {
        for (int i=1; i<argc; i++) {
            std::string option = argv[i];

            if (!option.compare("-help") || !option.compare("--help")) {
                throw std::runtime_error("Showing help message");
            }

            else if (!option.compare("--nocheck") || !option.compare("-nocheck")) {
                nocheck = true;
            }

            else if (!option.compare("-v") || !option.compare("--v") || !option.compare("--verbose") || !option.compare("-verbose")) {
                verbose = true;
            }

            else if (!option.compare("-w") || !option.compare("--w") || !option.compare("-width") || !option.compare("--width")) {
                if ((i+1) < argc)
                    width = atoi(argv[++i]);
                else
                    throw std::runtime_error("-width requires a int value");
            }

            else if (!option.compare("-b") || !option.compare("--b") || !option.compare("-block") || !option.compare("--block")) {
                if ((i+1) < argc)
                    block = atoi(argv[++i]);
                else
                    throw std::runtime_error("-block requires a int value");
            }

            else {
                throw std::runtime_error("Bad command line option " + option);
            }
        }

    } catch (std::runtime_error& e) {
        std::cerr << std::endl << e.what() << std::endl << desc << std::endl;
        exit(EXIT_FAILURE);
    }
}
