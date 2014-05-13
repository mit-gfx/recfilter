#include "split.h"

using std::string;

Arguments::Arguments(string app_name, int argc, char** argv) :
    width  (256),
    height (256),
    block  (32),
    debug  (false),
    verbose(false),
    nocheck(false),
    weight (1.0f),
    iterations(1)
{
    string desc = "\nUsage\n ./"+ app_name;
    desc.append(string(
                "[-width|-w w] [-block|-b b] [-weight W] [-iter i] [-nocheck] [-verbose|-v] [-debug|-d] [-help]\n\n"
                "\twidth\t  width of input image [default = 1024]\n"
                "\tblock\t  block size for splitting up image [default = 8]\n"
                "\tweight\t  first order filter weight [default = 0.5]\n"
                "\tverbose\t  print input image, refernce output, halide output and difference [default = false]\n"
                "\tnocheck\t do not check against reference solution [default = false]\n"
                "\tdebug\t  compute and print intermediate stages of recursive filter [default = false]\n"
                "\titer\t  number of profiling iterations [default = 1]\n"
                "\thelp\t  show help message\n"
                )
            );

    try {
        for (int i=1; i<argc; i++) {
            string option = argv[i];

            if (!option.compare("-help") || !option.compare("--help")) {
                throw std::runtime_error("Showing help message");
            }

            else if (!option.compare("-nocheck") || !option.compare("--nocheck")) {
                nocheck = true;
            }

            else if (!option.compare("-v") || !option.compare("--v") || !option.compare("--verbose") || !option.compare("-verbose")) {
                verbose = true;
            }

            else if (!option.compare("-d") || !option.compare("--d") || !option.compare("--debug") || !option.compare("-debug")) {
                debug = true;
            }

            else if (!option.compare("-iter") || !option.compare("--iter")) {
                if ((i+1) < argc)
                    iterations = atoi(argv[++i]);
                else
                    throw std::runtime_error("-iterations requires a int value");
            }

            else if (!option.compare("-weight") || !option.compare("--weight")) {
                if ((i+1) < argc)
                    weight = atof(argv[++i]);
                else
                    throw std::runtime_error("-weight requires a float value");
            }

            else if (!option.compare("-w") || !option.compare("--w") || !option.compare("-width") || !option.compare("--width")) {
                if ((i+1) < argc) {
                    width = atoi(argv[++i]);
                    height = width;
                }
                else
                    throw std::runtime_error("-width requires an integer value");
            }

            else if (!option.compare("-b") || !option.compare("--b") || !option.compare("-block") || !option.compare("--block")) {
                if ((i+1) < argc)
                    block = atoi(argv[++i]);
                else
                    throw std::runtime_error("-block requires an integer value");
            }

            else {
                throw std::runtime_error("Bad command line option " + option);
            }
        }

        if (width%block)
            throw std::runtime_error("Width should be a multiple of block size");

    } catch (std::runtime_error& e) {
        std::cerr << std::endl << e.what() << std::endl << desc << std::endl;
        exit(EXIT_FAILURE);
    }
}
