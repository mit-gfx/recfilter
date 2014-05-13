#include <iostream>
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <Halide.h>
#include <common_inc/timer.h>

using namespace Halide;

typedef struct {
    int width;     // image width
    int height;    // image height
    int block;     // block size
    bool debug;    // display intermediate stages
    bool verbose;  // display input, reference output, halide output, difference
    float weight;  // First order filter weight
    int  iterations; // profiling iterations
} Args;

void printImage(Image<float> image, const char* name);
void parseCommandLine(Args& args, int argc, char** argv);

int main(int argc, char **argv) {

    Args args;
    parseCommandLine(args, argc, argv);

    bool  verbose = args.verbose;
    bool  debug   = args.debug;
    int   width   = args.width;
    int   height  = args.height;
    int   block   = args.block;
    float W       = args.weight;
    int   iterations = args.iterations;

    // ----------------------------------------------------------------------------------------------

    std::cerr << "\nCreating random image and reference solution ... " << std::flush;

    Image<float> image(width,height);
    Image<float> ref(width,height);
    for (int y=0; y<height; y++)
        for (int x=0; x<width; x++)
            image(x,y) = 1.0f + float(rand() % 9);

    for (int y=0; y<height; y++)           // init the solution
        for (int x=0; x<width; x++)
            ref(x,y) = image(x,y);
    for (int y=0; y<height; y++)           // causal x filtering
        for (int x=0; x<width; x++)
            ref(x,y) = ref(x,y) + (x>0 ? W*ref(x-1,y) : 0);
    for (int y=0; y<height; y++)           // causal y filtering
        for (int x=0; x<width; x++)
            ref(x,y) = ref(x,y) + (y>0 ? W*ref(x,y-1) : 0);
    for (int y=height-1; y>=0; y--)        // anti-causal y filtering
        for (int x=width-1; x>=0; x--)
            ref(x,y) = ref(x,y) + (x<width-1 ? W*ref(x+1,y) : 0);
    for (int y=height-1; y>=0; y--)        // anti-causal y filtering
        for (int x=width-1; x>=0; x--)
            ref(x,y) = ref(x,y) + (y<height-1 ? W*ref(x,y+1) : 0);

    std::cerr << "done" << std::endl;

    // ----------------------------------------------------------------------------------------------

#define PAD           (1)                         // Padding = filter order
#define REVERSE_X(x)  (x_nblocks-1-(x))           // reverse along x for anticausal filtering
#define REVERSE_Y(y)  (y_nblocks-1-(y))           // reverse along y for anticausal filtering

    //
    // Halide solution
    //
    Var i("i"), j("j"), ox("ox"), oy("oy"), ix("ix"), iy("iy");

    //
    // Because of PAD, the inner dimensions range from 0 to block_size+2*PAD-1
    // Outer dimensions are the number of blocks in x and y
    //
    Expr x_extent = block+2*PAD-1;
    Expr y_extent = block+2*PAD-1;
    Expr x_nblocks = width/block+std::min(1,width%block);
    Expr y_nblocks = height/block+std::min(1,height%block);

    // Reductions do not need to be performed on the padded zone of inner dimensions
    RDom rdom(PAD, block, PAD, block);

    //
    // Reduction variables for transferring prologues/epilogues from one block to
    // next. The range of these is determined by outer block dimensions
    // i.e. number of blocks in x and y
    //
    RDom prolx(0, x_nblocks);
    RDom proly(0, y_nblocks);
    RDom epilx(0, x_nblocks);
    RDom epily(0, y_nblocks);

    //
    // Powers
    // pow_series_0(i) = 0                         , for i<0
    //                 = w^i                       , otherwise
    // pow_series_1(i) = 0                         , for i<0
    //                 w + w^3 + w^5 ... w^(2i+1)  , otherwise
    // pow_series_2(i) = 0                         , for i<0
    //                 w^5, w^4 + w^6, w^3 + w^5 + w^7, w^2 + w^4 + w^6 + w^8, w^1 + w^3 + w^5 + w^7 + w^9 ... otherwise
    //
    RDom rp1(1, block);
    Func pow_series_0("Pow_Series_0");
    pow_series_0(i)   = select(i>=0, Halide::pow(W,i), 0.0f);

    Func pow_series_1("Pow_Series_1");
    pow_series_1(i)   = select(i>=0, Halide::pow(W,2*i+1), 0.0f);
    pow_series_1(rp1) = pow_series_1(rp1) + pow_series_1(rp1-1);

    Func pow_series_2("Pow_Series_2");
    pow_series_2(i)   = select(i>=0, Halide::pow(W, block-i), 0.0f);
    pow_series_2(rp1) = pow_series_2(rp1) + W*pow_series_2(rp1-1);


    //
    // Func definitions for algorithm
    //
    Func Input            ("Input");              // input 4D image
    Func Causal_no_pred   ("Causal_no_pred");     // causal filter with 0 prologue
    Func Filtered_no_pred ("Filtered_no_pred");   // intra block filtering
    Func Pr_comp_x        ("Pr_comp_x");          // complete prologues along x
    Func Pr_comp_y        ("Pr_comp_y");          // complete prologues along y
    Func Ep_comp_x        ("Ep_comp_x");          // complete epilogues along x
    Func Ep_comp_y        ("Ep_comp_y");          // complete epilogues along y
    Func F                ("Out_image");          // output 2D image

    //
    // Stage 0: Convert 2D image into 4D. Outer dimensions: ox,oy; inner dimensions: ix,iy
    // ox = 0 to image_width / x_block_size
    // oy = 0 to image_height/ y_block_size
    // ix = 0 to x_block_size + 2*PAD
    // iy = 0 to y_block_size + 2*PAD
    // Both ix and iy should normally range from 0 to x_block_size. But we pad it on either
    // side with zeros to accommodate reductions. Since we have causal and anti-causal
    // filters of order "PAD", we need "PAD" zeros on either side.
    //
    Input(ox, oy, ix, iy) = select(ix>=PAD && ix<=block+PAD-1 && iy>=PAD && iy<=block+PAD-1,
            image(clamp(ox*block+(ix-PAD),0,width-1), clamp(oy*block+(iy-PAD),0,height-1)), 0.0f);

    //
    // Stage 1: compute the causal and anti-causal filtering within each block
    // with no contribution from neighboring blocks - inter-block parallelism to be exploited
    //
    Causal_no_pred(ox, oy, ix, iy) = Input(ox, oy, ix, iy);
    Causal_no_pred(ox, oy, rdom.x, rdom.y) = Causal_no_pred(ox,oy,rdom.x,rdom.y)
        + W*  Causal_no_pred(ox, oy, rdom.x-1, rdom.y)
        + W*  Causal_no_pred(ox, oy, rdom.x,   rdom.y-1)
        - W*W*Causal_no_pred(ox, oy, rdom.x-1, rdom.y-1);
    Filtered_no_pred(ox, oy, ix, iy) = Causal_no_pred(ox, oy, ix, iy);
    Filtered_no_pred(ox, oy, x_extent-rdom.x, y_extent-rdom.y) = Filtered_no_pred(ox, oy, x_extent-rdom.x, y_extent-rdom.y)
        + W*  Filtered_no_pred(ox, oy, x_extent-rdom.x+1, y_extent-rdom.y)
        + W*  Filtered_no_pred(ox, oy, x_extent-rdom.x,   y_extent-rdom.y+1)
        - W*W*Filtered_no_pred(ox, oy, x_extent-rdom.x+1, y_extent-rdom.y+1);

    //
    // Stage 2(a): Complete the y prologues by accumulating prologues of previous block.
    //
    Pr_comp_y(ox, oy, ix) = select(ox>=0 && oy>=0 && ox<x_nblocks && oy<y_nblocks, Filtered_no_pred(ox, oy, ix, PAD+block-1), 0.0f);
    Pr_comp_y(ox, proly, ix) = Pr_comp_y(ox,proly,ix) + pow_series_0(block) * Pr_comp_y(ox,proly-1,ix);

    //
    // Stage 2(b): Complete the twice incomplete y epilogues by accumulating
    // - next block's y epilogue
    // - prevoius block's y prologue multiplied by (w + w^3 + w^5 ...w^(2b-1)) [b=block size]
    //
    Ep_comp_y(ox, oy, ix) = select(ox>=0 && oy>=0 && ox<x_nblocks && oy<y_nblocks, Filtered_no_pred(ox, oy, ix, PAD), 0.0f);
    Ep_comp_y(ox, REVERSE_Y(epily), ix) = Ep_comp_y(ox, REVERSE_Y(epily), ix)
        + Ep_comp_y(ox, REVERSE_Y(epily-1), ix) * pow_series_0(block)
        + Pr_comp_y(ox, REVERSE_Y(epily+1), ix) * pow_series_1(block-1)
    ;


    //
    // Stage 3(a): Complete the thrice incomplete x prologues by accumulating
    // - x prologue of previous block along x
    // - y epilogue of next block along y
    // - y prologue of previous block along y
    //
    Pr_comp_x(ox, oy, iy) = select(ox>=0 && oy>=0 && ox<x_nblocks && oy<y_nblocks, Filtered_no_pred(ox, oy, PAD+block-1, iy), 0.0f);
    Pr_comp_x(prolx, oy, iy) = Pr_comp_x(prolx, oy, iy)
        + Pr_comp_x(prolx-1,oy,  iy)          * pow_series_0(block)
        + Ep_comp_y(prolx, oy+1, block+PAD-1) * pow_series_0( y_extent-iy-PAD+1)
        + Pr_comp_y(prolx, oy-1, block+PAD-1) * pow_series_2(y_extent-iy-PAD)
    ;


    //
    // Stage 3(b): Complete the four times incomplete x epilogues by accumulating
    // - x epilogue of next block along x
    // - x prologue of previous block along x
    // - y epilogue of next block along y
    // - y prologue of previous block along y
    //
    Ep_comp_x(ox, oy, iy) = select(ox>=0 && oy>=0 && ox<x_nblocks && oy<y_nblocks, Filtered_no_pred(ox, oy, PAD, iy), 0.0f);
    Ep_comp_x(REVERSE_X(epilx), oy, iy) = Ep_comp_x(REVERSE_X(epilx), oy, iy)
        + Ep_comp_x(REVERSE_X(epilx-1),oy,   iy)  * pow_series_0(block)
        + Pr_comp_x(REVERSE_X(epilx+1),oy,   iy)  * pow_series_1(block-1)
        + Ep_comp_y(REVERSE_X(epilx),  oy+1, PAD) * pow_series_0(y_extent-iy-PAD+1)
        + Pr_comp_y(REVERSE_X(epilx),  oy-1, PAD) * pow_series_2(y_extent-iy-PAD)
    ;

    //
    // Stage 4: Add the completed prologues and epilogues to the initial filtering
    // result in Stage 1
    //
    F(i,j) = Filtered_no_pred(i/block, j/block, i%block+PAD, j%block+PAD)
        + Pr_comp_y(i/block, j/block-1, i%block+PAD) * pow_series_2(y_extent-j%block-PAD-PAD)
        + Ep_comp_y(i/block, j/block+1, i%block+PAD) * pow_series_0(y_extent-j%block-PAD-PAD+1)
        + Pr_comp_x(i/block-1, j/block, j%block+PAD) * pow_series_2(x_extent-i%block-PAD-PAD)
        + Ep_comp_x(i/block+1, j/block, j%block+PAD) * pow_series_0(x_extent-i%block-PAD-PAD+1)
    ;

    // ----------------------------------------------------------------------------------------------

    //
    // Scheduling
    //

    std::string target = getenv("HL_TARGET");

    if (target.compare("ptx") == 0) {
        pow_series_0.compute_root();
        pow_series_1.compute_root();
        pow_series_2.compute_root();

        // Algorithm 5.1
        Causal_no_pred.compute_root().cuda_tile(ox, oy, iy, 8, 4, 4);
        Causal_no_pred.update().reorder(Var(rdom.y.name()), Var(rdom.x.name()), oy, ox).cuda_tile(ox, oy, 16, 8);

        Filtered_no_pred.compute_root().cuda_tile(ox, oy, iy, 8, 4, 4);
        Filtered_no_pred.update().reorder(Var(rdom.y.name()), Var(rdom.x.name()), oy, ox).cuda_tile(ox, oy, 16, 8);

        // Algorithm 5.2
        Pr_comp_y.compute_root().cuda_tile(ox, oy, ix, 8, 4, 4);
        Pr_comp_y.update().reorder(Var(proly.x.name()), ox, ix).cuda_tile(ox, ix, 16, 8);

        // Algorithm 5.3
        Ep_comp_y.compute_root().cuda_tile(ox, oy, ix, 8, 4, 4);
        Ep_comp_y.update().reorder(Var(epily.x.name()), ox, ix).cuda_tile(ox, ix, 16, 8);

        // Algorithm 5.4
        Pr_comp_x.compute_root().cuda_tile(ox, oy, iy, 8, 4, 4);
        Pr_comp_x.update().reorder(Var(prolx.x.name()), oy, iy).cuda_tile(oy, iy, 8, 8);

        // Algorithm 5.5
        Ep_comp_x.compute_root().cuda_tile(ox, oy, iy, 8, 4, 4);
        Ep_comp_x.update().reorder(Var(epilx.x.name()), oy, iy).cuda_tile(oy, iy, 8, 8);

        // Accumulate final result
        F.compute_root().cuda_tile(i, j, 16, 8);
    }
    else {
        std::cerr << "Warning: No CPU scheduling" << std::endl;
    }

    // ----------------------------------------------------------------------------------------------

    //
    // JIT compilation
    //

    std::cerr << "\nJIT compilation ... " << std::endl;
    F.compile_jit();

    Buffer hl_out_buff(type_of<float>(), width,height);
    {
        Timer t("Running ... ");
        for (int k=0; k<iterations; k++) {
            F.realize(hl_out_buff);
            if (k == iterations-1) {
                hl_out_buff.copy_to_host();
            }
            hl_out_buff.free_dev_buffer();
        }
    }

    // ----------------------------------------------------------------------------------------------

    std::cerr << "\nChecking difference ... " << std::flush;
    Image<float> hl_out(hl_out_buff);
    Image<float> diff(width,height);
    float diff_sum = 0;
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            diff(x,y) = ref(x,y) - hl_out(x,y);
            if (std::abs(1.0f - hl_out(x,y)/ref(x,y)) <= 1e-4)
                diff(x,y) = 0;
            diff_sum += diff(x,y);
        }
    }
    std::cerr << "done" << std::endl;

    if (verbose) {
        printImage(image, "Input");
        printImage(ref,   "Reference");
        printImage(hl_out,"Halide output");
        printImage(diff,  "Difference (less than 0.01\% difference rounded to 0)");
    } else {
        std::cerr << "\nDifference = " << diff_sum << std::endl;
        std::cerr << "Computed as sum of pixel differences where less than 0.01\% pixel difference is rounded to 0" << std::endl;
        std::cerr << std::endl;
    }

    if (debug) {
        int xblocks = width/block+std::min(1,width%block);
        int yblocks = height/block+std::min(1,height%block);
        Func F0, F1, F2, F3, F4, F5, F6, F7, F8;
        F4(i,j) = Pr_comp_x(i,j/block,j%block+PAD);
        F5(i,j) = Pr_comp_y(i/block,j,i%block+PAD);
        F6(i,j) = Ep_comp_x(i,j/block,j%block+PAD);
        F7(i,j) = Ep_comp_y(i/block,j,i%block+PAD);
        F8(i,j)  = Filtered_no_pred(i/block, j/block, i%block+PAD, j%block+PAD);
        Image<float> if4 = F4.realize(xblocks,height);
        Image<float> if5 = F5.realize(width,yblocks);
        Image<float> if6 = F6.realize(xblocks,height);
        Image<float> if7 = F7.realize(width,yblocks);
        Image<float> if8 = F8.realize(width,height);
        printImage(if4, "Prx_comp");
        printImage(if5, "Pry_comp");
        printImage(if6, "Epx_comp");
        printImage(if7, "Epy_comp");
        printImage(if8, "Incomplete");
    }

    return EXIT_SUCCESS;
}


void printImage(Image<float> image, const char* name) {
    std::cerr << std::endl << name << std::endl;
    for (int y=0; y<image.height(); y++) {
        for (int x=0; x<image.width(); x++)
            std::cerr << image(x,y) << " ";
        std::cerr << std::endl;
    }
}

void parseCommandLine(Args& args, int argc, char** argv) {
    std::string desc(
            "\nUsage\n ./recursive_filter [-width|-w w] [-block|-b b] [-weight W] [-iter i] [-verbose|-v] [-debug|-d] [-help]\n\n"
            "\twidth\t  width of input image [default = 1024]\n"
            "\tblock\t  block size for splitting up image [default = 8]\n"
            "\tweight\t  first order filter weight [default = 0.5]\n"
            "\tverbose\t  print input image, refernce output, halide output and difference [default = false]\n"
            "\tdebuge\t  compute and print intermediate stages of recursive filter [default = false]\n"
            "\titer\t  number of profiling iterations [default = 1]\n"
            "\thelp\t  show help message\n"
            );

    args.width   = 1024;     // image width
    args.height  = 1024;     // image height
    args.block   = 8;        // block size
    args.debug   = false;    // display intermediate stages
    args.verbose = false;    // display input, referenve output, halide output, difference
    args.weight  = 0.5f;     // first order filter weight
    args.iterations = 1;     // profiling iterations


    try {
        for (int i=1; i<argc; i++) {
            std::string option = argv[i];

            if (!option.compare("-help") || !option.compare("--help")) {
                throw std::runtime_error("Showing help message");
            }

            else if (!option.compare("-v") || !option.compare("--v") || !option.compare("--verbose") || !option.compare("-verbose")) {
                args.verbose = true;
            }

            else if (!option.compare("-d") || !option.compare("--d") || !option.compare("--debug") || !option.compare("-debug")) {
                args.debug = true;
            }

            else if (!option.compare("-iter") || !option.compare("--iter")) {
                if ((i+1) < argc)
                    args.iterations = atoi(argv[++i]);
                else
                    throw std::runtime_error("-iterations requires a int value");
            }

            else if (!option.compare("-weight") || !option.compare("--weight")) {
                if ((i+1) < argc)
                    args.weight = atof(argv[++i]);
                else
                    throw std::runtime_error("-weight requires a float value");
            }

            else if (!option.compare("-w") || !option.compare("--w") || !option.compare("-width") || !option.compare("--width")) {
                if ((i+1) < argc) {
                    args.width = atoi(argv[++i]);
                    args.height = args.width;
                }
                else
                    throw std::runtime_error("-width requires an integer value");
            }

            else if (!option.compare("-b") || !option.compare("--b") || !option.compare("-block") || !option.compare("--block")) {
                if ((i+1) < argc)
                    args.block = atoi(argv[++i]);
                else
                    throw std::runtime_error("-block requires an integer value");
            }

            else {
                throw std::runtime_error("Bad command line option " + option);
            }
        }

        if (args.width%args.block)
            throw std::runtime_error("Width should be a multiple of block size");

    } catch (std::runtime_error& e) {
        std::cerr << std::endl << e.what() << std::endl << desc << std::endl;
        exit(EXIT_FAILURE);
    }
}
