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
    int WIDTH  = (1<<args.width);
    int FILTER = (1<<args.filter);

    bool  nocheck = args.nocheck ;
    bool  verbose = args.verbose;

    // ----------------------------------------------------------------------------------------------

    std::cerr << "\nCreating random array ... " << std::endl;

    Image<int> image(WIDTH, WIDTH);

    srand(0);
    for (int y=0; y<WIDTH; y++) {
        for (int x=0; x<WIDTH; x++) {
            int r = 1 + (rand() % 9);
            image(x,y) = r;
        }
    }

    // ----------------------------------------------------------------------------------------------

    int numThreadsX = std::max(1, std::max(32, WIDTH/BLOCK));
    int numThreadsY = std::max(1, std::max(8, WIDTH/BLOCK));

    // ----------------------------------------------------------------------------------------------

    Var x("x"), y("y");

    Func Input("Input");
    Input(x,y) = select(x>=0 && x<WIDTH && y>=0 && y<WIDTH, image(clamp(x,0,WIDTH-1), clamp(y, 0, WIDTH-1)), 0);

    // ----------------------------------------------------------------------------------------------


    ///
    /// Func Sum("Sum");
    /// RDom r(0, WIDTH, 0, WIDTH);
    ///
    /// Var p("pure_x"), q("pure_y");
    /// RDom rInner(0, BLOCK, 0, BLOCK);
    /// RDom rOuter;
    ///
    /// Sum(x,y)  = 0;
    /// Sum(x,y) += Input(x+r.x, y+r.y);
    ///
    /// Sum.compute_root();
    /// Sum.cuda_tile(x, y, numThreadsX, numThreadsY);                  // applied on base case of both SumInner SumOuter
    ///
    /// Sum.update().split(r, rOuter, rInner, p, q);
    /// Sum.update()
    ///     .reorder(Var(rInner.x.name()), p, x)                        // applied on update of SumInner
    ///     .reorder(Var(rInner.y.name()), q, y)                        // applied on update of SumInner
    ///     .reorder(Var(rOuter.x.name()), Var(rOuter.y.name()), x, y)  // applied on update of SumOuter
    ///     .cuda_tile(x, y, numThreadsX, numThreadsY)                  // applied on update of SumInner SumOuter
    ///

    Var p("p"), q("q");
    RDom rInner(0, BLOCK, 0, BLOCK);
    RDom rOuter(0, FILTER/BLOCK, 0, FILTER/BLOCK);
    Func SumInner("SumInner"), SumOuter("SumOuter");

    SumInner(p,q,x,y)  = 0;
    SumInner(p,q,x,y) += Input(x + p*BLOCK + rInner.x, y + q*BLOCK + rInner.y);
    SumInner.compute_root();
    SumInner.cuda_tile(x, y, numThreadsX, numThreadsY);
    SumInner.update().reorder(Var(rInner.x.name()), p, x);
    SumInner.update().reorder(Var(rInner.y.name()), q, y);
    SumInner.update().cuda_tile(x, y, numThreadsX, numThreadsY);

    SumOuter(x,y) = 0;
    SumOuter(x,y) += SumInner(rOuter.x, rOuter.y, x, y);
    SumOuter.compute_root();
    SumOuter.cuda_tile(x, y, numThreadsX, numThreadsY);
    SumOuter.update().reorder(Var(rOuter.x.name()), Var(rOuter.y.name()), x, y);
    SumOuter.update().cuda_tile(x, y, numThreadsX, numThreadsY);

    // ----------------------------------------------------------------------------------------------

    ///
    /// JIT compilation
    ///

    std::cerr << "\nJIT compilation ... " << std::endl;
    SumOuter.compile_jit();

    std::cerr << "Running ... " << std::endl;
    Image<int> hl_out = SumOuter.realize(WIDTH, WIDTH);

    // ----------------------------------------------------------------------------------------------


    ///
    /// Check the difference
    ///

    if (!nocheck || verbose) {
        Image<int> ref(WIDTH, WIDTH);
        Image<int> diff(WIDTH, WIDTH);

        for (int y=0; y<WIDTH; y++) {
            for (int x=0; x<WIDTH; x++) {
                for (int q=0; q<FILTER; q++) {
                    for (int p=0; p<FILTER; p++) {
                        ref(x,y) += (x+p<WIDTH && y+q<WIDTH ? image(x+p,y+q) : 0);
                    }
                }
            }
        }

        int diff_sum = 0;
        for (int y=0; y<WIDTH; y++) {
            for (int x=0; x<WIDTH; x++) {
                diff(x,y) = abs(ref(x,y) - hl_out(x,y));
                diff_sum += diff(x,y);
            }
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

Args::Args(int argc, char** argv) :
    width(10),
    filter(7),
    block(5),
    nocheck(false),
    verbose(false)
{
    std::string desc(
            "\nUsage\n ./box2D [-width|-w] [-filter|-f f] [-block|-b] [-nocheck] [-verbose|-v] [-help]\n\n"
            "\twidth\t  exponent to decide image width = 2^w [default = 2^10, min = 2^5] \n"
            "\tfilter\t  exponent to decide summation width = 2^f [default = 2^7, min = 2^5] \n"
            "\tblock\t  exponent to decide block size  = 2^b [default = 2^5] \n"
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
                    filter = std::max(5, atoi(argv[++i]));
                else
                    throw std::runtime_error("-filter requires a int value");
            }

            else if (!option.compare("-w") || !option.compare("--w") || !option.compare("-width") || !option.compare("--width")) {
                if ((i+1) < argc)
                    width = std::max(5, atoi(argv[++i]));
                else
                    throw std::runtime_error("-width requires a int value");
            }

            else {
                throw std::runtime_error("Bad command line option " + option);
            }
        }

        if (block > filter) throw std::runtime_error("Summation block cannot be larger than filter radius");
        if (filter> width)  throw std::runtime_error("Filter radius cannot be large than width of image");


    } catch (std::runtime_error& e) {
        std::cerr << std::endl << e.what() << std::endl << desc << std::endl;
        exit(EXIT_FAILURE);
    }
}
