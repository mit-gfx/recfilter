#include <iostream>
#include <cstdio>
#include <stdexcept>
#include <Halide.h>
#include <common_inc/utils.h>

using namespace Halide;

class Args {                        ///< Command line arguments
public:
    int  width;                     ///< width exponent, actual image width = 2^width
    int  filter;                    ///< box filter width exponent, actual filter width = 2^filter
    int  block;                     ///< block exponent, actual block size = 2^block
    bool nocheck;                   ///< skip checking Halide output with reference solution
    bool verbose;                   ///< display input, reference output, halide output, difference
    Args(int argc, char** argv);
};



int main(int argc, char **argv) {
    Args args(argc, argv);

    int BLOCK  = (1<<args.block);
    int FILTER = (1<<args.filter);
    int WIDTH  = (1<<args.width);

    bool  nocheck = args.nocheck ;
    bool  verbose = args.verbose;

    // ----------------------------------------------------------------------------------------------

    std::cerr << "\nCreating random array ... " << std::endl;

    int ref_sum = 0;

    Image<int> image(WIDTH);

    srand(0);
    for (int x=0; x<WIDTH; x++) {
        image(x) = 1 + (rand() % 9);
    }


    // ----------------------------------------------------------------------------------------------

    int numThreads = std::max(1, std::min(256, FILTER/BLOCK));

    // ----------------------------------------------------------------------------------------------

    Var x("x"), p("p");

    Func Input("Input");
    Input(x) = select(x>=0 && x<WIDTH, image(clamp(x,0,WIDTH-1)), 0);

    // ----------------------------------------------------------------------------------------------


    Func ScanInner ("ScanInner");
    Func ScanOuter ("ScanOuter");
    Func Scan      ("Scan");

    RDom rInner(0, BLOCK);
    RDom rOuter(0, WIDTH/BLOCK);
    RDom rSum  (0, FILTER);

    ScanInner(p,x)      = select(p==0 && x==0, sum(image(clamp(rSum,0,WIDTH-1))), 0);
    ScanInner(p,rInner) = select(BLOCK*p+rInner+FILTER<0   || BLOCK*p+rInner+FILTER>WIDTH-1,  0, image(clamp(BLOCK*p+rInner+FILTER,  0,WIDTH-1)))
                        - select(BLOCK*p+rInner-FILTER-1<0 || BLOCK*p+rInner-FILTER-1>WIDTH-1,0, image(clamp(BLOCK*p+rInner-FILTER-1,0,WIDTH-1)))
                        + ScanInner(p, rInner)
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
        Image<int> ref(WIDTH);
        Image<int> diff(WIDTH);

        for (int y=0; y<WIDTH; y++) {
            for (int x=0; x<2*FILTER+1; x++) {
                ref(y) += (x-FILTER+y>=0 && x-FILTER+y<WIDTH ? image(x-FILTER+y) : 0);
            }
        }

        int diff_sum = 0;
        for (int x=0; x<WIDTH; x++) {
            diff(x) = abs(ref(x) - hl_out(x));
            diff_sum += diff(x);
        }

        if (verbose) {
            std::cerr << "Input" << std::endl << image << std::endl;
            std::cerr << "Reference" << std::endl << ref << std::endl;
            std::cerr << "Halide output" << std::endl << hl_out << std::endl;
        } else {
            std::cerr << "\nDifference = " << diff_sum << std::endl;
            std::cerr << std::endl;
        }
    }

    return EXIT_SUCCESS;
}


Args::Args(int argc, char** argv) :
    width(20),
    filter(10),
    block(5),
    nocheck(false),
    verbose(false)
{
    std::string desc(
            "\nUsage\n ./box1D [-width|-w] [-filter|-f f] [-block|-b] [-nocheck] [-verbose|-v] [-help]\n\n"
            "\twidth\t  exponent to decide image width = 2^w [default = 2^20] \n"
            "\tfilter\t  exponent to decide summation width = 2^f [default = 2^10] \n"
            "\tblock\t  exponent to decide block size  = 2^b [default = 2^7] \n"
            "\tnocheck\t  skip checking Halide result with reference solution [default = false]\n"
            "\tverbose\t  print input image, refernce output, halide output and difference [default = false]\n"
            "\thelp\t   show help message\n"
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

            else if (!option.compare("-b") || !option.compare("--b") || !option.compare("-block") || !option.compare("--block")) {
                if ((i+1) < argc)
                    block = atoi(argv[++i]);
                else
                    throw std::runtime_error("-block requires a int value");
            }

            else if (!option.compare("-f") || !option.compare("--f") || !option.compare("-filter") || !option.compare("--filter")) {
                if ((i+1) < argc)
                    filter = atoi(argv[++i]);
                else
                    throw std::runtime_error("-filter requires a int value");
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

        if (filter> width)  throw std::runtime_error("Filter radius cannot be larger than width of image");

    } catch (std::runtime_error& e) {
        std::cerr << std::endl << e.what() << std::endl << desc << std::endl;
        exit(EXIT_FAILURE);
    }
}
