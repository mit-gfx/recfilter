#ifndef _RECURSIVE_FILTER_INTERNALS_H_
#define _RECURSIVE_FILTER_INTERNALS_H_

#include <vector>
#include <string>
#include <Halide.h>

/** Info required to split a particular dimension of the recursive filter */
struct SplitInfo {
    int filter_order;                   ///< order of recursive filter in a given dimension
    int filter_dim;                     ///< dimension id
    int num_splits;                     ///< number of scans in the dimension that must be tiled

    Halide::Expr tile_width;            ///< width of each tile after splitting the dimension
    Halide::Expr image_width;           ///< width of the image in this dimension
    Halide::Expr num_tiles;             ///< number of tile in this dimension

    Halide::Var  var;                   ///< variable that represents this dimension
    Halide::Var  inner_var;             ///< inner variable after splitting
    Halide::Var  outer_var;             ///< outer variable or tile index after splitting

    Halide::RDom rdom;                  ///< RDom update domain of each scan
    Halide::RDom inner_rdom;            ///< inner RDom of each scan
    Halide::RDom truncated_inner_rdom;  ///< inner RDom width a truncated
    Halide::RDom outer_rdom;            ///< outer RDom of each scan
    Halide::RDom tail_rdom;             ///< RDom to extract the tail of each scan

    vector<bool> scan_causal;           ///< causal or anticausal flag for each scan
    vector<int>  scan_id;               ///< scan or update definition id of each scan

    vector<Halide::Expr> border_expr;   ///< image border value (can't contain the var or rdom)

    Halide::Image<float> feedfwd_coeff; ///< feedforward coeffs, only one for each scan
    Halide::Image<float> feedback_coeff;///< feedback coeffs (num_scans x max_order) order j-th coeff of i-th scan is (i+1,j) */
};

// ----------------------------------------------------------------------------

/** Recursive filter function with scheduling interface */
class RecFilterFunc {
public:
    /** Function categories */
    typedef enum {
        INLINE             = 0x0000,
        FULL_RESULT_SCAN   = 0x0001,
        FULL_RESULT_PURE   = 0x0002,
        INTRA_TILE_SCAN    = 0x0004,
        INTER_TILE_SCAN    = 0x0008,
        REINDEX_FOR_WRITE  = 0x0010,
        REINDEX_FOR_READ   = 0x0020,
    } FuncCategory;

    /** Var categories */
    typedef enum {
        INNER_PURE_VAR = 0x0100,
        INNER_SCAN_VAR = 0x0200,
        OUTER_PURE_VAR = 0x0400,
        OUTER_SCAN_VAR = 0x0800,
        TAIL_DIMENSION = 0x1000,
        PURE_DIMENSION = 0x2000,
        SCAN_DIMENSION = 0x4000,
    } VarCategory;

    /** Halide function */
    Halide::Internal::Function func;

    /** Category tag for the function */
    FuncCategory func_category;

    /** Category tags for all the pure def vars  */
    map<std::string, VarCategory> pure_var_category;

    /** Category tags for all the vars in all the update defs */
    vector<map<std::string,VarCategory> >  update_var_category;

    /** Name of a function that calls this function; set if this function
     * has the REINDEX_FOR_READ tag set */
    string caller_func;

    /** Name of a function that this function calls; set if this function
     * has the REINDEX_FOR_WRITE tag set */
    string callee_func;

    /** Function schedule as valid Halide code */
    vector<std::string> as_string;
};

// ----------------------------------------------------------------------------

/** Data members of the recursive filter */
struct RecFilterContents {
    /** Smart pointer */
    mutable Halide::Internal::RefCount ref_count;

    /** Function that contains the definition of the filter  */
    Halide::Func recfilter;

    /** Splitting info for each dimension of the filter */
    std::vector<SplitInfo> split_info;

    /** List of functions along with their names and their schedules */
    std::map<std::string, RecFilterFunc> func;
};

#endif // _RECURSIVE_FILTER_INTERNALS_H_
