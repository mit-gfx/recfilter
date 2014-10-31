#include "recfilter.h"
#include "recfilter_internals.h"

// RecFilter& RecFilter::store_root  (FuncTag f, ) { }
// RecFilter& RecFilter::compute_root(FuncTag f, ) { }
// RecFilter& RecFilter::compute_at  (FuncTag f, Func f, Var var) { }
// RecFilter& RecFilter::compute_at  (FuncTag f, Func f, RVar var) { }
// RecFilter& RecFilter::store_at    (FuncTag f, Func f, Var var) { }
// RecFilter& RecFilter::store_at    (FuncTag f, Func f, RVar var) { }
// RecFilter& RecFilter::split       (FuncTag f, VarTag old, VarTag outer, VarTag inner, Expr factor) { }
// RecFilter& RecFilter::fuse        (FuncTag f, VarTag inner, VarTag outer, VarTag fused) { }
// RecFilter& RecFilter::serial      (FuncTag f, VarTag var) { }
// RecFilter& RecFilter::parallel    (FuncTag f, VarTag var) { }
// RecFilter& RecFilter::parallel    (FuncTag f, VarTag var, Expr task_size) { }
// RecFilter& RecFilter::vectorize   (FuncTag f, VarTag var) { }
// RecFilter& RecFilter::unroll      (FuncTag f, VarTag var) { }
// RecFilter& RecFilter::vectorize   (FuncTag f, VarTag var, int factor) { }
// RecFilter& RecFilter::unroll      (FuncTag f, VarTag var, int factor) { }
// RecFilter& RecFilter::bound       (FuncTag f, Var var, Expr min, Expr extent) { }
// RecFilter& RecFilter::tile        (FuncTag f, VarTag x, VarTag y, VarTag xo, VarTag yo, VarTag xi, VarTag yi, Expr xfactor, Expr yfactor) { }
// RecFilter& RecFilter::tile        (FuncTag f, VarTag x, VarTag y, VarTag xi, VarTag yi, Expr xfactor, Expr yfactor) { }
// RecFilter& RecFilter::reorder     (FuncTag f, VarTag x, VarTag y) { }
// RecFilter& RecFilter::reorder     (FuncTag f, VarTag x, VarTag y, VarTag z) { }
// RecFilter& RecFilter::reorder     (FuncTag f, VarTag x, VarTag y, VarTag z, VarTag w) { }
// RecFilter& RecFilter::reorder     (FuncTag f, VarTag x, VarTag y, VarTag z, VarTag w, VarTag t) { }
// RecFilter& RecFilter::reorder     (FuncTag f, VarTag x, VarTag y, VarTag z, VarTag w, VarTag t1, VarTag t2) { }
// RecFilter& RecFilter::reorder     (FuncTag f, VarTag x, VarTag y, VarTag z, VarTag w, VarTag t1, VarTag t2, VarTag t3) { }
// RecFilter& RecFilter::reorder     (FuncTag f, VarTag x, VarTag y, VarTag z, VarTag w, VarTag t1, VarTag t2, VarTag t3, VarTag t4) { }
// RecFilter& RecFilter::reorder     (FuncTag f, VarTag x, VarTag y, VarTag z, VarTag w, VarTag t1, VarTag t2, VarTag t3, VarTag t4, VarTag t5) { }
// RecFilter& RecFilter::reorder     (FuncTag f, VarTag x, VarTag y, VarTag z, VarTag w, VarTag t1, VarTag t2, VarTag t3, VarTag t4, VarTag t5, VarTag t6) { }
