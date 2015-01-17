#ifndef _RECURSIVE_FILTER_H_
#define _RECURSIVE_FILTER_H_

#include <iomanip>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdio>
#include <algorithm>

#include <Halide.h>

// Forward declarations of internal structures

struct FilterInfo;
struct RecFilterContents;
class RecFilterFunc;
class RecFilterSchedule;
class RecFilter;
class RecFilterDim;
class RecFilterDimAndCausality;
class RecFilterRefVar;
/// class RecFilterRefExpr;

class FuncTag;
class VarTag;

enum VariableTag : int;
enum FunctionTag : int;

// ----------------------------------------------------------------------------


/** Filter dimension with variable name and width of image in the dimension */
class RecFilterDim {
private:
    Halide::Var v;  ///< variable for the dimension
    int         e;  ///< size of input/output buffer in the dimension

public:
    RecFilterDim(void) {}
    RecFilterDim(std::string var_name, int var_extent):
        v(var_name), e(var_extent) {}

    Halide::Var  var   (void) const { return v; }
    int          extent(void) const { return e; }

    /** Express as Halide::Expr so that it can be used to index other Halide
     * functions and buffers */
    operator Halide::Expr(void) {
        return Halide::Internal::Variable::make(Halide::Int(32), v.name());
    }
};

/** Filter dimension augmented with causality */
class RecFilterDimAndCausality {
private:
    RecFilterDim r; ///< variable for the filter dimension
    bool         c; ///< causality

public:
    RecFilterDimAndCausality(void) {}
    RecFilterDimAndCausality(RecFilterDim rec_var, bool causal):
        r(rec_var), c(causal) {}

    Halide::Var  var   (void) const { return r.var();    }
    int          extent(void) const { return r.extent(); }
    bool         causal(void) const { return c;          }

    /** Express as Halide::Expr so that it can be used to index other Halide
     * functions and buffers */
    operator Halide::Expr(void) {
        return Halide::Internal::Variable::make(Halide::Int(32), r.var().name());
    }
};

/** Operators to indicate causal and anticausal scans in a particular filter dimension */
// {@
/** Operator to create causal scan indication, +x indicates causal scan where
 * x is a RecFilterDim object */
RecFilterDimAndCausality operator+(RecFilterDim x);

/** Operator to create anticausal scan indication, -x indicates causal scan where
 * x is a RecFilterDim object */
RecFilterDimAndCausality operator-(RecFilterDim x);
// @}


// ----------------------------------------------------------------------------

/** Recursive filter class */
class RecFilter {
private:

    /** Data members of the recursive filter */
    Halide::Internal::IntrusivePtr<RecFilterContents> contents;

    /** Get the recursive filter function by name */
    RecFilterFunc& internal_function(std::string func_name);

    /** Get all recursive filter funcs that have the given tag */
    std::vector<std::string> internal_functions(FuncTag ftag);

    /** Get all the vars of a given recursive filter function with the given tag */
    std::map< int,std::vector<Halide::VarOrRVar> > internal_func_vars(RecFilterFunc f, VarTag vtag);

    /** Get one of the vars of a given recursive filter function with the given tag
     * indicated by the */
    std::map<int,Halide::VarOrRVar> internal_func_vars(RecFilterFunc f, VarTag vtag, uint vidx);

    /** Inline all calls to a given function
     *
     * Preconditions:
     * - function must not have any update definitions
     *
     * Side effects: function is completely removed from the filters depedency graph
     *
     * \param[in] func_name name of function to inline
     */
    void inline_func(std::string func_name);

    /** Finalize the filter; triggers automatic function transformations and cleanup */
    void finalize(void);

    /** Perform chores before realizing: compile the filter if not already done, upload
     * buffers to device and allocate buffers for realization
     *
     * \returns realization object that contains allocated buffers
     */
    Halide::Realization create_realization(void);

public:

    /** Empty constructor */
    RecFilter(std::string name="");

    /** Standard assignment operator */
    RecFilter& operator=(const RecFilter &r);

    /** Name of the filter */
    std::string name(void) const;

    /**@name Recursive filter specification */
    // {@
    RecFilterRefVar  operator()(RecFilterDim x);
    RecFilterRefVar  operator()(RecFilterDim x, RecFilterDim y);
    RecFilterRefVar  operator()(RecFilterDim x, RecFilterDim y, RecFilterDim z);
    RecFilterRefVar  operator()(std::vector<RecFilterDim> x);
    /// RecFilterRefExpr operator()(Halide::Expr x);
    /// RecFilterRefExpr operator()(Halide::Expr x, Halide::Expr y);
    /// RecFilterRefExpr operator()(Halide::Expr x, Halide::Expr y, Halide::Expr z);
    /// RecFilterRefExpr operator()(std::vector<Halide::Expr> x);
    // @}

    /** Add a pure definition to the recursive filter
     * \param pure_args list of pure args
     * \param pure_def list of expressions to initialize the filter
     */
    void define(std::vector<RecFilterDim> pure_args, std::vector<Halide::Expr> pure_def);

    /**@name Compile and run */
    // {@
    /** Trigger JIT compilation for specified hardware-platform target; dumps the generated
     * codegen in human readable HTML format if filename is specified */
    void compile_jit(std::string filename="");

    /** Compute the filter
     * \returns Realization object that contains all the buffers
     */
    Halide::Realization realize(void);

    /** Profile the filter
     * \param iterations number of profiling iterations
     * \returns computation time in milliseconds
     */
    float profile(int iterations);
    // @}


    /** @name Routines to add filters
     *
     *  @brief Add a causal or anticausal scan to the recursive filter with given
     *  feedback and feed forward coefficients
     *
     * \param x filter dimension
     * \param coeff 1 feedforward and n feedback coeffs (n = filter order)
     *
     * Preconditions:
     * - first argument must of of the form +x, -x or x where x is a RecFilterDim object
     */
    // {@
    void add_filter(RecFilterDim x, std::vector<float> coeff);
    void add_filter(RecFilterDimAndCausality x, std::vector<float> coeff);
    // @}

    /** @name Image boundary conditions
     * Clamp image border to the last pixel in all boundaries, default border is 0
     */
    // {@
    void set_clamped_image_border(void);
    // @}

    /** Cast the recfilter as a Halide::Func; this returns the function that holds
     * the final result of this filter; useful for extracting the result of this
     * function to use as input to other Halide Func
     */
    Halide::Func as_func(void);

    /**
     * Extract the constituent function by name, useful for debugging:
     * - realize only a particular stage for correctness or profiling
     * - debugging/testing schedules by using Halide's scheduling primitives directly
     *   on the function instead of the high level collective scheduling
     */
    Halide::Func func(std::string func_name);


    /**@name Tiling routines
     * @brief Tile a list of dimensions into their respective tile widths specified as
     * variable-tile width pairs.
     *
     * Preconditions:
     * - dimension with specified variable name must exist
     * - tile width must be a multiple of image width for each dimension
     */
    // {@
    void split_all_dimensions(int tx);
    void split(RecFilterDim x, int tx);
    void split(RecFilterDim x, int tx, RecFilterDim y, int ty);
    void split(RecFilterDim x, int tx, RecFilterDim y, int ty, RecFilterDim z, int tz);
    void split(std::map<std::string, int> dims);
    // @}


    /** @name Cascading API
     *
     *  Cascade the filter to produce multiple filters using list of list of scans and
     *  producing a list of recursive filters each ccomputes the corresponding list
     *  of scans in an overlapped fashion
     *
     *  Preconditions:
     *  - filter must not be tiled
     *  - list of list of scans spans all the scans of the original filter
     *  - no scan is repeated in the list of list of scans
     *  - the relative order of scans with respect to causality remains preserved
     *
     *  \param a list of scans for first filter
     *  \param b list of scans for second filter
     *
     *  \return two cascaded filters
     */
    // {@
    std::vector<RecFilter> cascade(std::vector<int> a, std::vector<int> b);

    /**
     *  Cascade the filter to produce multiple filters using list of list of scans and
     *  producing a list of recursive filters each ccomputes the corresponding list
     *  of scans in an overlapped fashion
     *
     *  Preconditions:
     *  - filter must not be tiled
     *  - list of list of scans spans all the scans of the original filter
     *  - no scan is repeated in the list of list of scans
     *  - the relative order of scans with respect to causality remains preserved
     *
     *  \param scan list of list of scans, each inner list becomes a separate filter
     *
     *  \return list of cascaded filters
     */
    std::vector<RecFilter> cascade(std::vector<std::vector<int> > scan);

    /** Computing all causal scans in all dimensions in an overlapped fashion
     * and all anticausal scans in an overlapped fashion and cascade the two groups
     *
     *  Preconditions:
     *  - filter must not be tiled
     *
     * \returns list of cascaded filters
     */
     std::vector<RecFilter> cascade_by_causality(void);

    /** Compute all scans in the same dimension in an overlapped fashion and cascade
     * different dimensions
     *
     *  Preconditions:
     *  - filter must not be tiled
     *
     * \returns list of cascaded filters
     */
    std::vector<RecFilter> cascade_by_dimension(void);

    /** Cascade a higher order filter into multiple lower order filters
     *
     * Preconditions:
     * - filter must not be tiled
     * - all scans in the given filter must have the same order
     * - list of lower orders must add up the to order of given filter
     *
     * Precautions:
     * - all lower order filters will have coefficients 1.0 and the last
     *   filter will have coefficients not equal to 1.0; this can lead to
     *   numerical issues
     *
     * \param orders list of low orders to be used for cascading
     * \returns lower order filters
     */
    std::vector<RecFilter> cascade_by_order(std::vector<int> orders);

    /** Cascade a higher order filter into two lower order filters
     *
     * Preconditions:
     * - filter must not be tiled
     * - all scans in the given filter must have the same order
     * - lower orders must add up the to order of given filter
     *
     * Precautions:
     * - first lower order filter will have coefficients 1.0 and the second
     *   filter will have coefficients not equal to 1.0; this can lead to
     *   numerical issues
     *
     * \param order_a order of first filter after cascading
     * \param order_b order of second filter after cascading
     *
     * \returns pair of lower order filters
     */
    std::vector<RecFilter> cascade_by_order(int order_a, int order_b);

    /** Overlap a given filter with the current filter
     *
     * Preconditions:
     * - filter must not be tiled
     * - given filter must have same number of dimensions in the same order
     * - each scan of each dimension of given filter must have same causality
     *
     * \param fA filter to be overlapped with current filter
     * \param name name of the overlapped filter (optional)
     *
     * \returns overlapped computation of all scans in both filters
     */
    RecFilter overlap_to_higher_order_filter(RecFilter fA, std::string name="O");
    // @}


    /**@name Print Halide code for the recursive filter */
    // {@
    std::string print_functions(void) const;
    std::string print_synopsis (void) const;
    std::string print_schedule (void) const;
    std::string print_hl_code  (void) const;
    // @}

    /**@name Scheduling handles for non-tiled filters */
    // {@

    /** Extract a handle to schedule the filter (if not tiled)
     *
     * \param[in] id pure/update def id, negative for pure def, else id-th update def
     * \return scheduling handle of the pure/update def to directly use Halide API
     */
    Halide::Stage schedule(int id=-1);

    /** Specify the filter (if not tiled) to be computed in global memory
     *
     *  \return scheduling handle to directly use Halide API
     */
    Halide::Stage compute_globally(void);
    // @}

    /**@name Collective scheduling: generic handles for scheduling for tiled filter */
    // {@

    /** Extract a handle to schedule intra-tile functions if the filter is tiled
     * \param id 0 for all intra tile functions, 1 for nD intra-tile functions, otherwise 1D intra-tile functions
     */
    RecFilterSchedule intra_schedule(int id=0);

    /** Extract a handle to schedule intra-tile functions if the filter is tiled */
    RecFilterSchedule inter_schedule(void);
    // @}

    /** Get the compilation target, inferred from HL_JIT_TARGET */
    Halide::Target target(void);

    /** @name Generic handles to write scheduled for dimensions of internal functions */
    // {@
    VarTag full      (int i=-1);
    VarTag inner     (int i=-1);
    VarTag outer     (int i=-1);
    VarTag tail      (void);
    VarTag full_scan (void);
    VarTag inner_scan(void);
    VarTag outer_scan(void);
    // @}

protected:
    /** Allow scheduler access to internal functions; only needed to append the
     * scheduling commands to each RecFilterFunc::schedule */
    friend class RecFilterSchedule;
};

// -----------------------------------------------------------------------------

/** Handle to schedule internal Halide functions that constitute the
 * recursive filter */
class RecFilterSchedule {
private:
    RecFilter                recfilter;
    std::vector<std::string> func_list;

    std::map<int,std::vector<Halide::VarOrRVar> > var_list_by_tag(RecFilterFunc f, VarTag vtag);
    std::map<int,Halide::VarOrRVar> var_by_tag(RecFilterFunc f, VarTag vtag);

public:
    RecFilterSchedule(RecFilter& r, std::vector<std::string> fl);

    RecFilterSchedule& compute_globally();
    RecFilterSchedule& compute_locally();

    RecFilterSchedule& split(VarTag v, int factor);
    RecFilterSchedule& fuse (VarTag v1, VarTag v2);

    RecFilterSchedule& reorder(VarTag x, VarTag y);
    RecFilterSchedule& reorder(VarTag x, VarTag y, VarTag z);
    RecFilterSchedule& reorder(VarTag x, VarTag y, VarTag z, VarTag w);
    RecFilterSchedule& reorder(VarTag x, VarTag y, VarTag z, VarTag w, VarTag s);
    RecFilterSchedule& reorder(VarTag x, VarTag y, VarTag z, VarTag w, VarTag s, VarTag t);
    RecFilterSchedule& reorder(VarTag x, VarTag y, VarTag z, VarTag w, VarTag s, VarTag t, VarTag u);

    RecFilterSchedule& reorder_storage(VarTag x, VarTag y);
    RecFilterSchedule& reorder_storage(VarTag x, VarTag y, VarTag z);
    RecFilterSchedule& reorder_storage(VarTag x, VarTag y, VarTag z, VarTag w);
    RecFilterSchedule& reorder_storage(VarTag x, VarTag y, VarTag z, VarTag w, VarTag t);

    RecFilterSchedule& unroll     (VarTag v, int factor=0);
    RecFilterSchedule& parallel   (VarTag v, int factor=0);
    RecFilterSchedule& vectorize  (VarTag v, int factor=0);
    RecFilterSchedule& gpu_threads(VarTag v1);
    RecFilterSchedule& gpu_threads(VarTag v1, VarTag v2);
    RecFilterSchedule& gpu_threads(VarTag v1, VarTag v2, VarTag v3);

    RecFilterSchedule& gpu_blocks(VarTag v1);
    RecFilterSchedule& gpu_blocks(VarTag v1, VarTag v2);
    RecFilterSchedule& gpu_blocks(VarTag v1, VarTag v2, VarTag v3);

    RecFilterSchedule& reorder (std::vector<VarTag> x);
    RecFilterSchedule& reorder_storage(std::vector<VarTag> x);
};


// ----------------------------------------------------------------------------

class RecFilterRefVar {
private:
    RecFilter rf;
    std::vector<RecFilterDim> args;

public:
    RecFilterRefVar(RecFilter r, std::vector<RecFilterDim> a);

    /**  Use this as the left-hand-side of a definition */
    void operator=(Halide::Expr pure_def);

    /** Use this as the left-hand-side of a definition for a Func with multiple outputs */
    void operator=(const Halide::Tuple &pure_def);

    /**  Use this as the left-hand-side of a definition */
    void operator=(Halide::FuncRefVar pure_def);

    /**  Use this as the left-hand-side of a definition */
    void operator=(Halide::FuncRefExpr pure_def);

    /** Use this as the left-hand-side of a definition for a Func with multiple outputs */
    void operator=(std::vector<Halide::Expr> pure_def);

///    /** Use this RecFilterRefVar as a call to the internal recfilter output */
///    operator Halide::Expr(void);
///
///     /** Use this RecFilterRefVar as a call to the one of the output buffers of
///      * the internal recfilter */
///     Halide::Expr operator[](int);
};

/// class RecFilterRefExpr {
/// private:
///     RecFilter rf;
///     std::vector<Halide::Expr> args;
///
/// public:
///     RecFilterRefExpr(RecFilter r, std::vector<Halide::Expr> a);
///
///     /** Use this RecFilterRefVar as a call to the internal recfilter output */
///     operator Halide::Expr(void);
///
///     /** Use this RecFilterRefVar as a call to the one of the output buffers of
///      * the internal recfilter */
///     Halide::Expr operator[](int);
/// };

// -----------------------------------------------------------------------------

/** Scheduling tags for Function dimensions */
class VarTag {
public:
    VarTag(void);
    VarTag(const VarTag      &t);
    VarTag(const VariableTag &t);
    VarTag(const VarTag      &t, int i);
    VarTag(const VariableTag &t, int i);
    VarTag(int i);

    VarTag& operator=(const VarTag      &t);
    VarTag& operator=(const VariableTag &t);

    int    as_integer(void) const;
    VarTag split_var (void) const;
    int    check     (const VariableTag &t) const;
    int    count     (void) const;
    bool   same_except_count(const VarTag &t) const;

private:
    VariableTag tag;
};

// -----------------------------------------------------------------------------

/** @name Printing utils for recursive filter, Halide functions, schedules and
 * difference between computed result and reference result */
// {@
std::ostream &operator<<(std::ostream &s, const RecFilter &r);
std::ostream &operator<<(std::ostream &s, const RecFilterFunc &f);
std::ostream &operator<<(std::ostream &s, const RecFilterDim &f);
std::ostream &operator<<(std::ostream &s, const Halide::Func &f);
std::ostream &operator<<(std::ostream &s, const Halide::Internal::Function &f);
// @}

// ----------------------------------------------------------------------------

/** Command line arg parser */
class Arguments {
    public:
        int width;       ///< image width
        int max_width;   ///< max image width
        int min_width;   ///< min image width
        int block;       ///< block size
        int iterations;  ///< profiling iterations
        int filter_reps; ///< filter iterations (multiple applications of the filter)
        bool nocheck;    ///< skip check Halide result against reference solution

        /** Parse command line args from number of args and list of args */
        Arguments(int argc, char** argv);
};

// ----------------------------------------------------------------------------

// Random image generation and printing utils

/** Generate an image of a given size with random entries */
template<typename T>
Halide::Image<T> generate_random_image(size_t w, size_t h=0, size_t c=0, size_t d=0) {
    Halide::Image<T> image;

    int MIN_ELEMENT = 1;
    int MAX_ELEMENT = 1;

    if (w && h && c && d) {
        image = Halide::Image<T>(w,h,c,d);
    } else if (w && h && c) {
        image = Halide::Image<T>(w,h,c);
    } else if (w && h) {
        image = Halide::Image<T>(w,h);
    } else if (w) {
        image = Halide::Image<T>(w);
    }

    if (image.dimensions() == 1) {
        for (size_t x=0; x<w; x++) {
            image(x) = T(MIN_ELEMENT + (rand() % MAX_ELEMENT));
        }
    }
    else if (image.dimensions() == 2) {
        for (size_t y=0; y<h; y++) {
            for (size_t x=0; x<w; x++) {
                image(x,y) = T(MIN_ELEMENT + (rand() % MAX_ELEMENT));
            }
        }
    }
    else if (image.dimensions() == 3) {
        for (size_t z=0; z<c; z++) {
            for (size_t y=0; y<h; y++) {
                for (size_t x=0; x<w; x++) {
                    image(x,y,z) = T(MIN_ELEMENT + (rand() % MAX_ELEMENT));
                }
            }
        }
    }
    else if (image.dimensions() == 4) {
        for (size_t t=0; t<d; t++) {
            for (size_t z=0; z<c; z++) {
                for (size_t y=0; y<h; y++) {
                    for (size_t x=0; x<w; x++) {
                        image(x,y,z,t) = T(MIN_ELEMENT + (rand() % MAX_ELEMENT));
                    }
                }
            }
        }
    }
    return image;
}


/** Print an image */
template<typename T>
std::ostream &operator<<(std::ostream &s, Halide::Image<T> image) {
    int precision = 4;
    if (image.dimensions() == 1) {
        for (size_t x=image.min(0); x<image.min(0)+image.extent(0); x++) {
            s << std::setw(precision) << image(x) << " ";
        }
        s << "\n";
    }
    else if (image.dimensions() == 2) {
        for (size_t y=image.min(1); y<image.min(1)+image.extent(1); y++) {
            for (size_t x=image.min(0); x<image.min(0)+image.extent(0); x++) {
                s << std::setw(precision) << float(image(x,y)) << " ";
            }
            s << "\n";
        }
    }
    else if (image.dimensions() == 3) {
        for (size_t z=image.min(2); z<image.min(2)+image.extent(2); z++) {
            for (size_t y=image.min(1); y<image.min(1)+image.extent(1); y++) {
                for (size_t x=image.min(0); x<image.min(0)+image.extent(0); x++) {
                    s << std::setw(precision) << float(image(x,y,z)) << " ";
                }
                s << "\n";
            }
            s << "--\n";
        }
    }
    else if (image.dimensions() == 4) {
        for (size_t w=image.min(3); w<image.min(3)+image.extent(3); w++) {
            for (size_t z=image.min(2); z<image.min(2)+image.extent(2); z++) {
                for (size_t y=image.min(1); y<image.min(1)+image.extent(1); y++) {
                    for (size_t x=image.min(0); x<image.min(0)+image.extent(0); x++) {
                        s << std::setw(precision) << float(image(x,y,z,w)) << " ";
                    }
                    s << "\n";
                }
                s << "--\n";
            }
            s << "--\n";
        }
    }
    return s;
}

// ----------------------------------------------------------------------------

/** Compare ref and Halide solutions and print the mean square error */
template <typename T>
class CheckResult {
public:
    float max_diff;             ///< max diff percentage
    float mean_diff;            ///< mean diff percentage
    Halide::Image<T> ref;       ///< reference solution
    Halide::Image<T> out;       ///< Halide solution
    Halide::Image<float> diff;  ///< pixel wise diff

    CheckResult(Halide::Image<T> r, Halide::Image<T> o)
        : max_diff(0.0), mean_diff(0.0), ref(r), out(o)
    {
        assert(r.width()   == o.width());
        assert(r.height()  == o.height());
        assert(r.channels()== o.channels());

        int width   = r.width();
        int height  = r.height();
        int channels= r.channels();

        diff = Halide::Image<float>(width, height, channels);

        for (int z=0; z<channels; z++) {
            for (int y=0; y<height; y++) {
                for (int x=0; x<width; x++) {
                    diff(x,y,z) = r(x,y,z) - o(x,y,z);
                    float re   = 100.0 * std::abs(diff(x,y,z)) / (r(x,y,z) + 1e-9);
                    mean_diff += re;
                    max_diff   = std::max(re, max_diff);
                }
            }
        }
        mean_diff /= float(width*height*channels);
    }
};

/** Compare ref and Halide solutions and print the verbose difference */
template <typename T>
class CheckResultVerbose : public CheckResult<T> {
public:
    CheckResultVerbose(Halide::Image<T> r, Halide::Image<T> o) :
        CheckResult<T>(r,o) {}
};


/** Print the synopsis of checking error */
template<typename T>
std::ostream &operator<<(std::ostream &s, const CheckResult<T> &v) {
    s << "Max  relative error = " << v.max_diff << " % \n";
    s << "Mean relative error = " << v.mean_diff << " % \n\n";
    return s;
}

/** Print the result and synopsis of checking error */
template<typename T>
std::ostream &operator<<(std::ostream &s, const CheckResultVerbose<T> &v) {
    s << "Reference" << "\n" << v.ref << "\n";
    s << "Halide output" << "\n" << v.out << "\n";
    s << "Difference " << "\n" << v.diff << "\n";
    s << "Max  relative error = " << v.max_diff << " % \n";
    s << "Mean relative error = " << v.mean_diff << " % \n\n";
    return s;
}


#endif // _RECURSIVE_FILTER_H_
