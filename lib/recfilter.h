#ifndef _RECURSIVE_FILTER_H_
#define _RECURSIVE_FILTER_H_

#include <iomanip>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdio>

#include <Halide.h>

/** Info required to split a particular dimension of the recursive filter */
struct SplitInfo {
    bool clamp_border;          ///< should image border be clamped to non-zero entries

    int filter_order;           ///< order of recursive filter in a given dimension
    int filter_dim;             ///< dimension id
    int num_splits;             ///< number of scans in the dimension that must be tiled

    Halide::Expr tile_width;    ///< width of each tile after splitting the dimension
    Halide::Expr image_width;   ///< width of the image in this dimension
    Halide::Expr num_tiles;     ///< number of tile in this dimension

    Halide::Var  var;           ///< variable that represents this dimension
    Halide::Var  inner_var;     ///< inner variable after splitting
    Halide::Var  outer_var;     ///< outer variable or tile index after splitting

    vector<bool> scan_causal;   ///< causal or anticausal flag for each scan
    vector<int>  scan_id;       ///< scan or reduction definition id of each scan

    vector<Halide::RDom> rdom;       ///< RDom reduction domain of each scan
    vector<Halide::RDom> outer_rdom; ///< outer RDom of each scan
    vector<Halide::RDom> inner_rdom; ///< inner RDom of each scan
    vector<Halide::RDom> tail_rdom;  ///< RDom to extract the tail of each scan

    Halide::Image<float> feedfwd_coeff; ///< feedforward coeffs, only one for each scan
    Halide::Image<float> feedback_coeff;///< feedback coeffs (num_scans x max_order) order j-th coeff of i-th scan is (i+1,j) */
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
};

}

// ----------------------------------------------------------------------------

/** Compare ref and Halide solutions and print the mean square error */
struct CheckResult {
    Halide::Image<float> ref;   ///< reference solution
    Halide::Image<float> out;   ///< Halide solution
    CheckResult(
            Halide::Image<float> r,
            Halide::Image<float> o) :
        ref(r), out(o) {}
};

/** Compare ref and Halide solutions and print the verbose difference */
struct CheckResultVerbose {
    Halide::Image<float> ref;   ///< reference solution
    Halide::Image<float> out;   ///< Halide solution
    CheckResultVerbose(
            Halide::Image<float> r,
            Halide::Image<float> o) :
        ref(r), out(o) {}
};

// ----------------------------------------------------------------------------

class RecFilter {
private:

    /** Data members of the recursive filter */
    Halide::Internal::IntrusivePtr<Internal::RecFilterContents> contents;

public:

    /** Macros to indicate causal or anticausal scan */
    typedef enum {CAUSAL, ANTICAUSAL} Causality;

    /** Construct an empty named recursive filter */
    RecFilter(std::string name = "RecFilter");

    /** Reconstruct a recursive filter from its contents */
    RecFilter(const Halide::Internal::IntrusivePtr<RecFilterContents> &c) : contents(c) {}

    /**@name Recursive filter specification
     * @brief Set the dimensions of the output of the recursive filter
     */
    // {@
    void setArgs(Halide::Var x);
    void setArgs(Halide::Var x, Halide::Var y);
    void setArgs(Halide::Var x, Halide::Var y, Halide::Var z);
    void setArgs(std::vector<Halide::Var> args);
    // @}

    /** @name Recursive filter definition
     * @brief Add a pure definition to the recursive filter, can be Tuple
     * All Vars in the pure definition should be args of the filter
     */
    // {@
    void define(Halide::Expr pure_def);
    void define(Halide::Tuple pure_def);
    // @}

    /** @name Routines to add scans to a recursive filter
     *  @brief Add a scan to the recursive filter given parameters
     *  defaults filter order = 1, feedforward/feedback coefficient = 1.0,
     *  causalilty = CAUSAL
     */
    // {@
    void addScan(
            Halide::Var x,              ///< dimension to a reduction
            Halide::RDom rx,            ///< domain of the scan
            float feedfwd,              ///< single feedforward coeff
            std::vector<float> feedback,///< n feedback coeffs, where n is filter order
            Causality c = CAUSAL        ///< causal or anticausal scan
            );
    void addScan(
            Halide::Var x,              ///< dimension to a reduction
            Halide::RDom rx,            ///< domain of the scan
            Causality c = CAUSAL        ///< causal or anticausal scan
            );
    void addScan(
            Halide::Var x,              ///< dimension to a reduction
            Halide::RDom rx,            ///< domain of the scan
            std::vector<float> feedback,///< n feedback coeffs, where n is filter order
            Causality c = CAUSAL        ///< causal or anticausal scan
            );
    // @}


    /**@name Dependency graph of the recursive filter
     * @brief Return only the final recursive filter as Halide function,
     * or any function in required to compute the complete filter (searches
     * the dependency graph by function name) or all the functions in the
     * dependency graph
     */
    // {@
    Halide::Func func(void);
    Halide::Func func(std::string func_name);
    std::map<std::string,Halide::Func> funcs(void);
    // @}


    /**@name Splitting routines
     * @brief Split a list of dimensions by separate tiling factors
     * (defined in split.cpp)
     */
    // {@
    void split(Halide::Var x, Halide::Expr tx);
    void split(Halide::Var x, Halide::Expr tx, Halide::Var y, Halide::Expr ty);
    void split(std::map<std::string, Halide::Expr> dims);
    // @}


    /** @brief Cascade scans in different dimensions of a function,
     * (defined in reorder.cpp)
     */
    std::vector<RecFilter> cascade_scans(std::vector<std::vector<int> > scan);


    /** @brief Inline all calls to a pure function
     * (defined in reorder.cpp)
     */
    void inline_func(
            std::string func_name   ///< name of function to be inlined
            );


    /** @brief Reorder memory layout by swapping two dimensions of a function
     * (defined in reorder.cpp) */
    void swap_variables(
            std::string func,   ///< name of function whose dimensions must be swapped
            Halide::Var a,      ///< pure arg of first dimension to swap
            Halide::Var b       ///< pure arg of second dimension to swap
            );


    /**@name Merging routines
     * @brief Merge multiple functions into a single function with mutiple outputs
     * The functions to be merged are searched in the dependency graph of functions
     * required to compute the recursive filter (defined in reorder.cpp)
     */
    // {@
    void merge_func(
            std::string func_a, ///< name of first function to merge
            std::string func_b, ///< name of second function to merge
            std::string merged  ///< name of merged function
            );
    void merge_func(
            std::string func_a, ///< name of first function to merge
            std::string func_b, ///< name of second function to merge
            std::string func_c, ///< name of third function to merge
            std::string merged  ///< name of merged function
            );
    void merge_func(
            std::string func_a, ///< name of first function to merge
            std::string func_b, ///< name of second function to merge
            std::string func_c, ///< name of third function to merge
            std::string func_d, ///< name of fourth function to merge
            std::string merged  ///< name of merged function
            );
    void merge_func(
            std::vector<std::string> func, ///< list of names of functions to merge
            std::string merged             ///< name of merged function
            );
    // @}
};

#endif // _RECURSIVE_FILTER_H_
