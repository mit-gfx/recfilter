#ifndef _RECURSIVE_FILTER_H_
#define _RECURSIVE_FILTER_H_

#include <iomanip>
#include <vector>
#include <queue>
#include <string>
#include <stdexcept>
#include <cstdio>

#include <Halide.h>

struct SplitInfo {
    bool clamp_border;

    int filter_order;
    int filter_dim;
    int num_splits;

    Halide::Expr tile_width;
    Halide::Expr image_width;
    Halide::Expr num_tiles;

    Halide::Var  var;
    Halide::Var  inner_var;
    Halide::Var  outer_var;

    vector<bool> scan_causal;
    vector<int>  scan_id;

    vector<Halide::RDom> rdom;
    vector<Halide::RDom> outer_rdom;
    vector<Halide::RDom> inner_rdom;
    vector<Halide::RDom> tail_rdom;

    Halide::Image<float> feedfwd_coeff;
    Halide::Image<float> feedback_coeff;
};

// ----------------------------------------------------------------------------

struct RecFilterContents {
    std::string name;
    std::vector<SplitInfo> split_info;
};

// ----------------------------------------------------------------------------

class RecFilter {
public:
    struct CheckResult {
        Halide::Image<float> ref, out;
        CheckResult(
                Halide::Image<float> r,
                Halide::Image<float> o) :
            ref(r), out(o) {}
    };

    struct CheckResultVerbose {
        Halide::Image<float> ref, out;
        CheckResultVerbose(
                Halide::Image<float> r,
                Halide::Image<float> o) :
            ref(r), out(o) {}
    };

private:
    Halide::Internal::IntrusivePtr<RecFilterContents> contents;

public:

    /** Construct an empty named recursive filter */
    RecFilter(std::string n="");

    /** Set the dimensions of the output of the recursive filter */
    void setArgs(
            std::vector<Var> args,      ///< dimensions of the output domain
            std::vector<Expr> width     ///< size of the dimension
            );

    /** Add a pure definition to the recursive filter, can be Tuple.
     * All Vars in the pure definition should be args of the filter
     * */
    void define(Expr pure_def);

    /** Add a scan to the recursive filter */
    void addScan(
            bool causal                 ///< causal or anticausal scan
            std::vector<float> feedback,///< feedback coeff
            );

    /** Return a Halide function that is required to compute the
     * recursive filter, raise error if no function by the given
     * name is required to compute the filter */
    Halide::Func func(std::string func_name);

    /** Split a list of dimensions by a splitting factor
     * (defined in \file split.cpp) */
    void split(
            std::vector<Var> dims,  ///< list of dimensions to split
            std::vector<Expr> tile  ///< splitting factor in each dimension
            );

    /** Cascade scans in different dimensions of a function,
     * (defined in \file reorder.cpp)
     *
     *
     * */
    std::vector<RecFilter> cascade_scans(std::vector<std::vector<int> > scan);

    /** Inline all calls to a pure function
     * (defined in \file reorder.cpp) */
    void inline_func(
            std::string func_name   ///< name of function to be inlined
            );

    /** Swap two dimensions of a function, reorders the memory layout
     * (defined in \file reorder.cpp) */
    void swap_variables(
            std::string func,   ///< name of function whose dimensions must be swapped
            Halide::Var a,      ///< pure arg of first dimension to swap
            Halide::Var b       ///< pure arg of second dimension to swap
            );

    /** Merge multiple functions into a single function with mutiple outputs
     * (defined in \file reorder.cpp) */
    void merge_func(
            std::string func_a, ///< name of first function to merge
            std::string func_b, ///< name of second function to merge
            std::string merged  ///< name of merged function
            );

    /** Merge multiple functions into a single function with mutiple outputs
     * (defined in \file reorder.cpp) */
    void merge_func(
            std::string func_a, ///< name of first function to merge
            std::string func_b, ///< name of second function to merge
            std::string func_c, ///< name of third function to merge
            std::string merged  ///< name of merged function
            );

    /** Merge multiple functions into a single function with mutiple outputs
     * (defined in \file reorder.cpp) */
    void merge_func(
            std::string func_a, ///< name of first function to merge
            std::string func_b, ///< name of second function to merge
            std::string func_c, ///< name of third function to merge
            std::string func_d, ///< name of fourth function to merge
            std::string merged  ///< name of merged function
            );

    /** Merge multiple functions into a single function with mutiple outputs
     * (defined in \file reorder.cpp) */
    void merge_func(
            std::vector<std::string> funcs, ///< list of names of functions to merge
            std::string merged              ///< name of merged function
            );
};

#endif // _RECURSIVE_FILTER_H_
