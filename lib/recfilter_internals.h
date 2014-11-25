#ifndef _RECURSIVE_FILTER_INTERNALS_H_
#define _RECURSIVE_FILTER_INTERNALS_H_

#include <vector>
#include <string>
#include <Halide.h>

/** Info about scans in a particular dimension */
struct FilterInfo {
    int                  filter_order;  ///< order of recursive filter in a given dimension
    int                  filter_dim;    ///< dimension id
    int                  num_scans;     ///< number of scans in the dimension that must be tiled
    Halide::Expr         image_width;   ///< image width in this dimension
    Halide::Var          var;           ///< variable that represents this dimension
    Halide::RDom         rdom;          ///< RDom update domain of each scan
    std::vector<bool>    scan_causal;   ///< causal or anticausal flag for each scan
    std::vector<int>     scan_id;       ///< scan or update definition id of each scan
};

// ----------------------------------------------------------------------------

/** Recursive filter function with scheduling interface */
class RecFilterFunc {
public:
    /** Halide function */
    Halide::Internal::Function func;

    /** Category tag for the function */
    FuncTag func_category;

    /** Category tags for all the pure def vars  */
    std::map<std::string, VarTag> pure_var_category;

    /** Category tags for all the vars in all the update defs */
    std::vector<std::map<std::string,VarTag> >  update_var_category;

    /** Name of a function that calls this function; set if this function
     * has the REINDEX_FOR_READ tag set */
    std::string caller_func;

    /** Name of a function that this function calls; set if this function
     * has the REINDEX_FOR_WRITE tag set */
    std::string callee_func;

    /** Function schedule as valid Halide code; first element is pure
     * def schedule, subsequent entries are update def schedule */
    std::map<int, std::vector<std::string> >  schedule;
};

// ----------------------------------------------------------------------------

/** Data members of the recursive filter */
struct RecFilterContents {
    /** Smart pointer */
    mutable Halide::Internal::RefCount ref_count;

    /** Flag to indicate if the filter has been tiled  */
    bool tiled;

    /** Flag to indicate if the filter has been finalized, which performs platform specific optimizations */
    bool finalized;

    /** Name of recursive filter as well as function that contains the
     * definition of the filter  */
    std::string name;

    /** Info about all the scans in the recursive filter */
    std::vector<FilterInfo> filter_info;

    /** List of functions along with their names and their schedules */
    std::map<std::string, RecFilterFunc> func;

    /** Image border expression */
    Halide::Expr border_expr;

    /** Feed forward coeffs, only one for each scan */
    Halide::Image<float> feedfwd_coeff;

    /** Feedback coeffs (num_scans x max_order) order j-th coeff of i-th scan is (i+1,j) */
    Halide::Image<float> feedback_coeff;
};

#endif // _RECURSIVE_FILTER_INTERNALS_H_
