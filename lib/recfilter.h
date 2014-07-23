#ifndef _RECURSIVE_FILTER_H_
#define _RECURSIVE_FILTER_H_

#include <iomanip>
#include <vector>
#include <queue>
#include <string>
#include <stdexcept>
#include <cstdio>

#include <Halide.h>

/** Info required to split a particular dimension of the recursive filter */
struct SplitInfo {
    bool clamp_border;      ///< should image border be clamped to non-zero entries

    int filter_order;       ///< order of recursive filter in a given dimension
    int filter_dim;         ///< dimension id
    int num_splits;         ///< number of scans in the dimension that must be tiled

    Halide::Expr tile_width;    ///< width of each tile after splitting the dimension
    Halide::Expr image_width;   ///< width of the image in this dimension
    Halide::Expr num_tiles;     ///< number of tile in this dimension

    Halide::Var  var;           ///< variable that represents this dimension
    Halide::Var  inner_var;     ///< inner variable after splitting
    Halide::Var  outer_var;     ///< outer variable or tile index after splitting

    vector<bool> scan_causal;   ///< causal or anticausal flag for each scan
    vector<int>  scan_id;       ///< scan or reduction definition id of each scan

    vector<Halide::RDom> rdom;       ///< reduction domain of each scan
    vector<Halide::RDom> outer_rdom; ///< outer reduction domain of each scan
    vector<Halide::RDom> inner_rdom; ///< outer reduction domain of each scan
    vector<Halide::RDom> tail_rdom;  ///< reduction domain to extract the tail of each scan

    Halide::Image<float> feedfwd_coeff;  ///< copy of feedfwd coeff for the whole filter
    Halide::Image<float> feedback_coeff; ///< copy of feedback coeff for the whole filter
};

// ----------------------------------------------------------------------------

/** Data members of the recursive filter */
struct RecFilterContents {
    /** Smart pointer */
    mutable Halide::Internal::RefCount ref_count;

    /** Function that contains the definition of the filter  */
    Halide::Func recfilter;

    /** List of functions along with their names that the filter depends upon */
    std::map<std::string, Halide::Internal::Function> func_map;

    /** List of functions that the filter depends upon */
    std::vector<Halide::Internal::Function> func_list;

    /** Splitting info for each dimension of the filter */
    std::vector<SplitInfo> split_info;

    /** Matrix of feed forward coefficients, only one for each reduction definition  */
    Halide::Image<float> feedfwd_coeff;

    /** Matrix of feed back coefficients, number of scans x max order
     * order j-th feedback coefficient of i-th reduction defintion is
     * feedback_ceff(i,j) */
    Halide::Image<float> feedback_coeff;
};

// ----------------------------------------------------------------------------

class RecFilter {
private:

    /** Data members of the recursive filter */
    Halide::Internal::IntrusivePtr<RecFilterContents> contents;

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

public:
    /** Construct an empty named recursive filter */
    RecFilter(std::string name = "RecFilter");

    /** Reconstruct a recursive filter from its contents */
    RecFilter(const Halide::Internal::IntrusivePtr<RecFilterContents> &c) : contents(c) {}

    /** Set the dimensions of the output of the recursive filter */
    void setArgs(
            std::vector<Halide::Var> args,      ///< dimensions of the output domain
            std::vector<Halide::Expr> width     ///< size of the dimension
            );

    /** Add a pure definition to the recursive filter, can be Tuple.
     * All Vars in the pure definition should be args of the filter
     */
    // {@
    void define(Halide::Expr pure_def);
    void define(Halide::Tuple pure_def);
    // @}

    /** Add a scan to the recursive filter */
    void addScan(
            bool causal,                ///< causal or anticausal scan
            Halide::Var x,              ///< dimension to a reduction
            Halide::RDom rx,            ///< domain of the scan
            float feedfwd,              ///< single feedforward coeff
            std::vector<float> feedback ///< n feedback coeffs, where n is filter order
            );

    /** Return a Halide function that is required to compute the
     * recursive filter, raise error if no function by the given
     * name is required to compute the filter */
    Halide::Func func(std::string func_name);

    /** Split a list of dimensions by a splitting factor
     * (defined in \file split.cpp) */
    void split(
            std::vector<Halide::Var> dims,  ///< list of dimensions to split
            std::vector<Halide::Expr> tile  ///< splitting factor in each dimension
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
