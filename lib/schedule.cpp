#include "recfilter.h"
#include "recfilter_internals.h"

using namespace Halide;

RecFilter& RecFilter::compute_root(FuncTag f) { }
RecFilter& RecFilter::compute_at  (FuncTag f) { }
RecFilter& RecFilter::split       (FuncTag f, VarTag old, VarTag outer, VarTag inner, Expr factor) { }
RecFilter& RecFilter::fuse        (FuncTag f, VarTag inner, VarTag outer, VarTag fused) { }
RecFilter& RecFilter::serial      (FuncTag f, VarTag var) { }
RecFilter& RecFilter::parallel    (FuncTag f, VarTag var) { }
RecFilter& RecFilter::parallel    (FuncTag f, VarTag var, Expr task_size) { }
RecFilter& RecFilter::vectorize   (FuncTag f, VarTag var) { }
RecFilter& RecFilter::unroll      (FuncTag f, VarTag var) { }
RecFilter& RecFilter::vectorize   (FuncTag f, VarTag var, int factor) { }
RecFilter& RecFilter::unroll      (FuncTag f, VarTag var, int factor) { }
RecFilter& RecFilter::bound       (FuncTag f, VarTag var, Expr min, Expr extent) { }
RecFilter& RecFilter::tile        (FuncTag f, VarTag x, VarTag y, VarTag xo, VarTag yo, VarTag xi, VarTag yi, Expr xfactor, Expr yfactor) { }
RecFilter& RecFilter::tile        (FuncTag f, VarTag x, VarTag y, VarTag xi, VarTag yi, Expr xfactor, Expr yfactor) { }
RecFilter& RecFilter::reorder     (FuncTag f, VarTag x, VarTag y) { }
RecFilter& RecFilter::reorder     (FuncTag f, VarTag x, VarTag y, VarTag z) { }
RecFilter& RecFilter::reorder     (FuncTag f, VarTag x, VarTag y, VarTag z, VarTag w) { }
RecFilter& RecFilter::reorder     (FuncTag f, VarTag x, VarTag y, VarTag z, VarTag w, VarTag t) { }
RecFilter& RecFilter::reorder     (FuncTag f, VarTag x, VarTag y, VarTag z, VarTag w, VarTag t1, VarTag t2) { }
RecFilter& RecFilter::reorder     (FuncTag f, VarTag x, VarTag y, VarTag z, VarTag w, VarTag t1, VarTag t2, VarTag t3) { }
RecFilter& RecFilter::reorder     (FuncTag f, VarTag x, VarTag y, VarTag z, VarTag w, VarTag t1, VarTag t2, VarTag t3, VarTag t4) { }
RecFilter& RecFilter::reorder     (FuncTag f, VarTag x, VarTag y, VarTag z, VarTag w, VarTag t1, VarTag t2, VarTag t3, VarTag t4, VarTag t5) { }
RecFilter& RecFilter::reorder     (FuncTag f, VarTag x, VarTag y, VarTag z, VarTag w, VarTag t1, VarTag t2, VarTag t3, VarTag t4, VarTag t5, VarTag t6) { }
RecFilter& gpu_threads            (FuncTag f, VarTag thread_x) { }
RecFilter& gpu_threads            (FuncTag f, VarTag thread_x, VarTag thread_y) { }
RecFilter& gpu_threads            (FuncTag f, VarTag thread_x, VarTag thread_y, VarTag thread_z) { }
RecFilter& gpu_blocks             (FuncTag f, VarTag block_x) { }
RecFilter& gpu_blocks             (FuncTag f, VarTag block_x, VarTag block_y) { }
RecFilter& gpu_blocks             (FuncTag f, VarTag block_x, VarTag block_y, VarTag block_z) { }
RecFilter& gpu                    (FuncTag f, VarTag block_x, VarTag thread_x) { }
RecFilter& gpu                    (FuncTag f, VarTag block_x, VarTag block_y, VarTag thread_x, VarTag thread_y) { }
RecFilter& gpu                    (FuncTag f, VarTag block_x, VarTag block_y, VarTag block_z, VarTag thread_x, VarTag thread_y, VarTag thread_z) { }
RecFilter& gpu_tile               (FuncTag f, VarTag x, int x_size) { }
RecFilter& gpu_tile               (FuncTag f, VarTag x, VarTag y, int x_size, int y_size) { }
RecFilter& gpu_tile               (FuncTag f, VarTag x, VarTag y, VarTag z, int x_size, int y_size, int z_size) { }
RecFilter& reorder_storage        (FuncTag f, VarTag x, VarTag y) { }
RecFilter& reorder_storage        (FuncTag f, VarTag x, VarTag y, VarTag z) { }
RecFilter& reorder_storage        (FuncTag f, VarTag x, VarTag y, VarTag z, VarTag w) { }
RecFilter& reorder_storage        (FuncTag f, VarTag x, VarTag y, VarTag z, VarTag w, VarTag t) { }
