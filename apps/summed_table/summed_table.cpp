#include <iostream>
#include <cstdio>
#include <string>
#include <cmath>
#include <Halide.h>

#include <common_inc/timer.h>
#include <common_inc/filter.h>
#include <common_inc/utils.h>

using namespace Halide;

int main(int argc, char **argv) {
    Arguments args("summed_table", argc, argv);

    int PAD   =  1;
    int BLOCK = (1<<args.block);
    int WIDTH = (1<<args.width);

    // ----------------------------------------------------------------------------------------------

    Image<float> image = generate_random_image<float>(WIDTH,WIDTH);

    // ----------------------------------------------------------------------------------------------

    ///
    /// Halide solution
    ///
    Var x("x"), y("y"), ix("ix"), iy("iy"), ox("ox"), oy("oy"), tx("tx"), ty("ty");

    /// Number of blocks in image
    Expr numBlocks = WIDTH/BLOCK;

    /// Reductions do not need to be performed on the padded zone of inner dimensions
    RDom rx(PAD, BLOCK);
    RDom ry(PAD, BLOCK);

    ///
    /// Reduction variables for transferring prologues from one block to
    /// next, range is determined by number of blocks in x and y
    ///
    RDom px(0, numBlocks);
    RDom py(0, numBlocks);

    ///
    /// Func definitions for algorithm
    ///
    Func Input             ("Input");               ///< Input 4D image
    Func Prologue_incomp_x ("Prologue_incomp_x");   ///< x causal filter to compute incomplete prologues
    Func Prologue_incomp_y ("Prologue_incomp_y");   ///< y causal filter to compute incomplete prologues
    Func Prologue_incomp   ("Prologue_incomp");     ///< Incomplete prologues
    Func Prologue_comp_x   ("Prologue_comp_x");     ///< Complete prologues along x
    Func Prologue_comp_y   ("Prologue_comp_y");     ///< Complete prologues along y
    Func Causal_x          ("Causal_x");            ///< x causal filter to compute final result
    Func Causal_y          ("Causal_y");            ///< y causal filter to compute final result
    Func Final             ("Final");               ///< Output 2D image

    ///
    /// Stage 0: Convert 2D image into 4D.
    /// Outer dimensions: ox,oy; inner dimensions: ix,iy
    /// ox = 0 to image_width / block_size
    /// oy = 0 to image_width/ block_size
    /// ix = 0 to block_size + 2*PAD
    /// iy = 0 to block_size + 2*PAD
    /// Both ix and iy should normally range from 0 to block_size-1.
    /// But we pad it on either side with zeros to accommodate reductions.
    ///
    /// Stage 1: compute the causal filtering within each block
    /// with no contribution from neighboring blocks - inter-block parallelism to be exploited
    ///
    Prologue_incomp_y(ix, iy, ox, oy)  = select(ix>=PAD && ix<=BLOCK+PAD-1 && iy>=PAD && iy<=BLOCK+PAD-1, image(clamp(ox*BLOCK+(ix-PAD),0,-1), clamp(oy*BLOCK+(iy-PAD),0,WIDTH-1)), 0.0f);
    Prologue_incomp_y(ix, ry, ox, oy) += Prologue_incomp_y(ix,   ry-1, ox, oy);
    Prologue_incomp_x(ix, iy, ox, oy)  = Prologue_incomp_y(ix,   iy  , ox, oy);
    Prologue_incomp_x(rx, iy, ox, oy) += Prologue_incomp_x(rx-1, iy  , ox, oy);
    Prologue_incomp  (ix, x, ox, oy)   = select(x==0, Prologue_incomp_x(ix, PAD+BLOCK-1, ox, oy), Prologue_incomp_x(PAD+BLOCK-1, ix, ox, oy));

    Prologue_incomp_y.compute_at(Prologue_incomp, Var("blockidx"));
    Prologue_incomp_x.compute_at(Prologue_incomp, Var("blockidx"));

    Prologue_incomp_y.reorder(ix, iy, ox, oy).cuda_threads(ix);
    Prologue_incomp_x.reorder(iy, ix, ox, oy).cuda_threads(iy);

    Prologue_incomp_y.update().reorder(Var(ry.x.name()), ix, ox, oy).cuda_threads(ix);
    Prologue_incomp_x.update().reorder(Var(rx.x.name()), iy, ox, oy).cuda_threads(iy);

    Prologue_incomp.compute_root().cuda_blocks(ox,oy).cuda_threads(ix);

    ///
    /// Stage 2: Compute the incomplete y prologues as last column of each inner block.
    /// Complete the y prologues oy accumulating prologues of previous block.
    ///
    Prologue_comp_y(x, oy)  = 0.0f;
    Prologue_comp_y(x, py) += Prologue_incomp(x%BLOCK+PAD, 0, x/BLOCK, py) + Prologue_comp_y(x, py-1);

    Prologue_comp_y.compute_root().reorder(oy, x).cuda_tile(x, BLOCK*6);
    Prologue_comp_y.update().reorder(Var(py.x.name()), x).cuda_tile(x, BLOCK*6);


    ///
    /// Stage 3: Causal the the twice incomplete x prologues as last row of each.
    /// Complete the twice incomplete x prologues oy accumulating
    /// - x prologue of previous block along x
    /// - y prologue of previous block along y
    ///
    Prologue_comp_x(y, ox)  = 0.0f;
    Prologue_comp_x(y, px) += Prologue_incomp(y%BLOCK+PAD, 1, px, y/BLOCK) + Prologue_comp_x(y, px-1) + Prologue_comp_y(px*BLOCK+BLOCK-1, y/BLOCK-1);

    Prologue_comp_x.compute_root().reorder(ox, y).cuda_tile(y, BLOCK*6);
    Prologue_comp_x.update().reorder(Var(px.x.name()), y).cuda_tile(y, BLOCK*6);

    ///
    /// Stage 4: Add the completed prologues and epilogues to the initial filtering
    /// result in Stage 1
    ///
    Causal_y(ix, iy, ox, oy)  = select(ix>=PAD && ix<=BLOCK+PAD-1 && iy>=PAD && iy<=BLOCK+PAD-1, image(clamp(ox*BLOCK+(ix-PAD),0,-1), clamp(oy*BLOCK+(iy-PAD),0,WIDTH-1)), 0.0f);
    Causal_y(ix, ry, ox, oy) += Causal_y(ix,   ry-1, ox, oy);
    Causal_x(ix, iy, ox, oy)  = Causal_y(ix,   iy  , ox, oy);
    Causal_x(rx, iy, ox, oy) += Causal_x(rx-1, iy  , ox, oy);

    Final(x,y) = Causal_x(x%BLOCK+PAD, y%BLOCK+PAD, x/BLOCK, y/BLOCK) + Prologue_comp_y(x, y/BLOCK-1) + Prologue_comp_x(y, x/BLOCK-1);

#define PRE_FETCH 1
#if PRE_FETCH
    // split iy by 7, because we can only launch 32*6 threads in a block
    Causal_y.compute_at(Final, Var("blockidx")).split(iy, iy, y, 7).reorder(y, iy, ix, ox, oy).cuda_threads(ix, iy);
#else
    Causal_y.compute_at(Final, Var("blockidx")).reorder(iy, ix, ox, oy).cuda_threads(ix);
#endif
    Causal_y.update().reorder(Var(ry.x.name()), ix, ox, oy).cuda_threads(ix);

    Causal_x.compute_at(Final, Var("blockidx")).reorder(ix, iy, ox, oy).cuda_threads(iy);
    Causal_x.update().reorder(Var(rx.x.name()), iy, ox, oy).cuda_threads(iy);

    Final.compute_root().split(y, oy, iy, BLOCK).split(x, ox, ix, BLOCK).reorder(iy, ix, ox, oy).cuda_blocks(ox,oy).cuda_threads(ix);
    Final.bound(x, 0, WIDTH).bound(y, 0, WIDTH);

    // ----------------------------------------------------------------------------------------------

    ///
    /// JIT compilation
    ///

    std::cerr << "\nJIT compilation ... " << std::endl;
    Final.compile_jit();

    Buffer hl_out_buff(type_of<float>(), WIDTH, WIDTH);
    {
        Timer t("Running ... ");
        for (int k=0; k<args.iterations; k++) {
            Final.realize(hl_out_buff);
            if (k == args.iterations-1)
                hl_out_buff.copy_to_host();
            hl_out_buff.free_dev_buffer();
        }
    }

    // ----------------------------------------------------------------------------------------------


    ///
    /// Check the difference
    ///

    if (!args.nocheck || args.verbose) {
        Timer t("Naive CPU serial computation");

        Image<float> ref = reference_recursive_filter<float>(image);

        Image<float> hl_out(hl_out_buff);
        Image<float> diff(WIDTH,WIDTH);
        float diff_sum = 0;
        for (int y=0; y<WIDTH; y++) {
            for (int x=0; x<WIDTH; x++) {
                diff(x,y) = std::abs(ref(x,y) - hl_out(x,y));
                diff_sum += diff(x,y);
            }
        }

        if (args.verbose) {
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
