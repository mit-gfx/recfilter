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
    int                  image_width;   ///< image width in this dimension
    int                  tile_width;    ///< tile width in this dimension
    Halide::Var          var;           ///< variable that represents this dimension
    Halide::RDom         rdom;          ///< RDom update domain of each scan
    std::vector<bool>    scan_causal;   ///< causal or anticausal flag for each scan
    std::vector<int>     scan_id;       ///< scan or update definition id of each scan
};

// ----------------------------------------------------------------------------

enum FunctionTag : int {
    INLINE  = 0x000, ///< function to be removed by inlining
    INTER   = 0x010, ///< filter over tail elements across tiles (single 1D scan)
    INTRA_N = 0x020, ///< filter within tile (multiple scans in multiple dimensions)
    INTRA_1 = 0x040, ///< filter within tile (single scan in one dimension)
    REINDEX = 0x100, ///< function that reindexes a subset of another function to write to global mem
};

enum VariableTag: int {
    INVALID = 0x0000, ///< invalid var
    FULL    = 0x0010, ///< full dimension before tiling
    INNER   = 0x0020, ///< inner dimension after tiling
    OUTER   = 0x0040, ///< outer dimension after tiling
    TAIL    = 0x0080, ///< if dimension is at lower granularity (only for inner dimensions)
    SCAN    = 0x0100, ///< if dimension is a scan
    __1     = 0x0001, ///< first variable with one of the above tags
    __2     = 0x0002, ///< second variable with one of the above tags
    __3     = 0x0004, ///< third variable with one of the above tags
    __4     = 0x0008, ///< fourth variable with one of the above tags
    SPLIT   = 0x1000, ///< any variable generated by split scheduling operations
};


/** @name Logical operations for scheduling tags */
// {@
VarTag      operator |(const VarTag &a, const VarTag &b);
VarTag      operator &(const VarTag &a, const VarTag &b);
VariableTag operator |(const VariableTag &a, const VariableTag &b);
VariableTag operator &(const VariableTag &a, const VariableTag &b);

bool operator==(const FuncTag &a, const FuncTag &b);
bool operator==(const VarTag  &a, const VarTag &b);
bool operator!=(const FuncTag &a, const FuncTag &b);
bool operator!=(const VarTag  &a, const VarTag &b);
bool operator==(const FuncTag &a, const FunctionTag &b);
bool operator==(const VarTag  &a, const VariableTag &b);
// @}


/** @name Utils to print scheduling tags */
// {@
std::ostream &operator<<(std::ostream &s, const FunctionTag &f);
std::ostream &operator<<(std::ostream &s, const VariableTag &v);
std::ostream &operator<<(std::ostream &s, const FuncTag &f);
std::ostream &operator<<(std::ostream &s, const VarTag &v);
// @}


/** Scheduling tags for Functions */
class FuncTag {
public:
    FuncTag(void)                 : tag(INLINE){}
    FuncTag(const FuncTag     &t) : tag(t.tag) {}
    FuncTag(const FunctionTag &t) : tag(t)     {}
    FuncTag& operator=(const FuncTag     &t) { tag=t.tag; return *this; }
    FuncTag& operator=(const FunctionTag &t) { tag=t;     return *this; }
    int as_integer(void) const { return static_cast<int>(tag); }

private:
    FunctionTag tag;
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

    /** Schedule for pure def of the function as valid Halide code */
    std::vector<std::string> pure_schedule;

    /** Schedule for update defs of the function as valid Halide code */
    std::map<int, std::vector<std::string> > update_schedule;
};

// ----------------------------------------------------------------------------

/** Data members of the recursive filter */
struct RecFilterContents {
    /** Smart pointer */
    mutable Halide::Internal::RefCount ref_count;

    /** Flag to indicate if the filter has been tiled  */
    bool tiled;

    /** Flag to indicate if the filter has been JIT compiled, required before execution */
    bool compiled;

    /** Flag to indicate if the filter has been finalized, required before compilation */
    bool finalized;

    /** Image border expression */
    bool clamped_border;

    /** Name of recursive filter as well as function that contains the
     * definition of the filter  */
    std::string name;

    /** Filter output type */
    Halide::Type type;

    /** Info about all the scans in the recursive filter */
    std::vector<FilterInfo> filter_info;

    /** List of functions along with their names and their schedules */
    std::map<std::string, RecFilterFunc> func;

    /** Feed forward coeffs, only one for each scan */
    Halide::Image<double> feedfwd_coeff;

    /** Feedback coeffs (num_scans x max_order) order j-th coeff of i-th scan is (i+1,j) */
    Halide::Image<double> feedback_coeff;

    /** Compilation and execution target */
    Halide::Target target;
};

#endif // _RECURSIVE_FILTER_INTERNALS_H_
