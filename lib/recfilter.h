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
class RecFilterRefExpr;

class FuncTag;
class VarTag;

enum VariableTag : int;
enum FunctionTag : int;

// ----------------------------------------------------------------------------


/** Filter dimension with variable name and width of image in the dimension */
class RecFilterDim {
private:
    Halide::Var  v;  ///< variable for the dimension
    Halide::Expr e;  ///< size of input/output buffer in the dimension

public:
    RecFilterDim(void) {}
    RecFilterDim(std::string var_name, Halide::Expr var_extent):
        v(var_name), e(var_extent) {}

    Halide::Var  var   (void) const { return v; }
    Halide::Expr extent(void) const { return e; }

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
    Halide::Expr extent(void) const { return r.extent(); }
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

    /** @brief Inline all calls to a given function
     *
     * Preconditions:
     * - function must not have any update definitions
     *
     * Side effects: function is completely removed from the filters depedency graph
     * */
    void inline_func(std::string func_name);

    /** Finalize the filter; triggers automatic function transformations and cleanup */
    void finalize(Halide::Target target);

public:
    /**
     * Merge multiple functions into a single function with mutiple outputs
     * \param func_list list of functions to merge
     * \param merged name of merged function
     *
     * Preconditions: functions to be merged must have
     * - same pure args
     * - scans with same update args and update domains in same order
     *
     * Side effects: all calls to the functions to be merged are replaced by the merged function
     */
    void merge_func(std::vector<std::string> func_list, std::string merged);

    /** Reorder memory layout by swapping two dimensions of a function
     *  \param func_name name of function whose dimensions have to be transposed
     *  \param a variable name of the first dimension to be transposed
     *  \param b variable name of the second dimension to be transposed
     *
     * Preconditions: none
     *
     * Side effects: dimensions are also transposed in all calls to the function
     */
    void transpose_dimensions(std::string func, std::string a, std::string b);

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
    RecFilterRefExpr operator()(Halide::Expr x);
    RecFilterRefExpr operator()(Halide::Expr x, Halide::Expr y);
    RecFilterRefExpr operator()(Halide::Expr x, Halide::Expr y, Halide::Expr z);
    RecFilterRefExpr operator()(std::vector<Halide::Expr> x);
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

    /** Compute the filter for a given output buffer for specified number of iterations
     * for timing purposes; last iteration copies the result to host, returns
     * computation time in milliseconds */
    double realize(Halide::Buffer out, int iterations=1);
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
    void add_filter(RecFilterDim x, std::vector<double> coeff);
    void add_filter(RecFilterDimAndCausality x, std::vector<double> coeff);
    // @}

    /** @name Image boundary conditions
     * Clamp image border to the last pixel in all boundaries, default border is 0
     */
    // {@
    void set_clamped_image_border(void);
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
    std::vector<Halide::Func> funcs(void);
    // @}


    /**@name Tiling routines
     * @brief Tile a list of dimensions into their respective tile widths specified as
     * variable-tile width pairs.
     *
     * Preconditions:
     * - dimension with specified variable name must exist
     * - tile width must be a multiple of image width for each dimension
     */
    // {@
    void split(Halide::Expr tx);
    void split(RecFilterDim x, Halide::Expr tx);
    void split(RecFilterDim x, Halide::Expr tx, RecFilterDim y, Halide::Expr ty);
    void split(RecFilterDim x, Halide::Expr tx, RecFilterDim y, Halide::Expr ty, RecFilterDim z, Halide::Expr tz);
    void split(std::map<std::string, Halide::Expr> dims);
    // @}


    /** @name Reorder scans in the filter or cascade them to produce multiple filters
     *  @brief uses list of list of scans as argument, producing a list of
     *  recursive filters each containing the respective list of scans
     *
     *  Preconditions:
     *  - list of list of scans spans all the scans of the original filter
     *  - no scan is repeated in the list of list of scans
     *  - the relative order of reverse causality scans in same dimension remains same
     */
    // {@
    RecFilter cascade(
            std::vector<int> a      ///< reordered list of scans of the filter
            );
    std::vector<RecFilter> cascade(
            std::vector<int> a,     ///< list of scans for first filter
            std::vector<int> b      ///< list of scans for second filter
            );
    std::vector<RecFilter> cascade(
            std::vector<int> a,     ///< list of scans for first filter
            std::vector<int> b,     ///< list of scans for second filter
            std::vector<int> c      ///< list of scans for third filter
            );
    std::vector<RecFilter> cascade(
            std::vector<int> a,     ///< list of scans for first filter
            std::vector<int> b,     ///< list of scans for second filter
            std::vector<int> c,     ///< list of scans for third filter
            std::vector<int> d      ///< list of scans for fourth filter
            );
    std::vector<RecFilter> cascade(
            std::vector<std::vector<int> > scan ///< list of scans for list of filters
            );
    // @}


    /**@name Interleaving routines
     *
     * @brief Interleave two functions into a single function with output that contains
     * the buffers of both the input functions.  The functions to be interleaved are
     * searched in the dependency graph of functions required to compute the recursive filter
     *
     * Preconditions: functions to be interleaved must
     * - have same args
     * - be pure functions (no update defs)
     */
    // {@
    void interleave_func(
            std::string  func_a,    ///< name of first function to interleave
            std::string  func_b,    ///< name of second function to interleave
            std::string  merged,    ///< name of interleaved function
            std::string  var,       ///< var to the used for interleaving
            Halide::Expr stride     ///< interleaving stride
            );
    // @}


    /**@name Print Halide code for the recursive filter */
    // {@
    std::string print_functions(void) const;
    std::string print_synopsis (void) const;
    std::string print_schedule (void) const;
    std::string print_hl_code  (void) const;
    // @}

    /**@name Generic handles to write schedules for internal functions */
    // {@

    /** Extract a handle to schedule intra-tile functions
     * \param id 0 for all intra tile functions, 1 for nD intra-tile functions, otherwise 1D intra-tile functions
     */
    RecFilterSchedule intra_schedule(int id=0);

    /** Extract a handle to schedule intra-tile functions */
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

    /**
     * Millisecond-precision timer function
     * \return Clock value in milliseconds
     *
     * This routine implements a timer with millisecond precision.  In order to
     * obtain timing at high resolution, platform-specific functions are needed:
     *
     *    - On Windows systems, the GetSystemTime function is used.
     *    - On Mac and POSIX systems, the gettimeofday function is used.
     *
     * Preprocessor symbols are checked in attempt to detect whether the platform
     * is POSIX or Windows or Mac and defines millisecond_timer() accordingly.
     */
    unsigned long millisecond_timer(void);

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

    RecFilterSchedule& compute_in_global();
    RecFilterSchedule& compute_in_shared();

    RecFilterSchedule& split(VarTag v, int factor);

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

    RecFilterSchedule& unroll     (VarTag v);
    RecFilterSchedule& parallel   (VarTag v);
    RecFilterSchedule& vectorize  (VarTag v);
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

    /** Use this RecFilterRefVar as a call to the internal recfilter output */
    operator Halide::Expr(void);

    /** Use this RecFilterRefVar as a call to the one of the output buffers of
     * the internal recfilter */
    Halide::Expr operator[](int);
};

class RecFilterRefExpr {
private:
    RecFilter rf;
    std::vector<Halide::Expr> args;

public:
    RecFilterRefExpr(RecFilter r, std::vector<Halide::Expr> a);

    /** Use this RecFilterRefVar as a call to the internal recfilter output */
    operator Halide::Expr(void);

    /** Use this RecFilterRefVar as a call to the one of the output buffers of
     * the internal recfilter */
    Halide::Expr operator[](int);
};

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
    int    check     (const VarTag &t)      const;
    int    check     (const VariableTag &t) const;
    int    count     (void) const;

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
        int block;       ///< block size
        int iterations;  ///< profiling iterations
        int threads;     ///< maximum threads per GPU tile
        bool nocheck;    ///< skip check Halide result against reference solution

        /** Parse command line args from number of args and list of args */
        Arguments(int argc, char** argv);
};

// ----------------------------------------------------------------------------

// Random image generation and printing utils

/** Generate an image of a given size with random entries */
template<typename T>
Halide::Image<T> generate_random_image(size_t w, size_t h=1, size_t c=1, size_t d=1) {
    Halide::Image<T> image;

    int MIN_ELEMENT = 1;
    int MAX_ELEMENT = 1;

    if (h<=1 && c<=1 && d<=1) {
        image = Halide::Image<T>(w);
    } else if (c<=1 && d<=1) {
        image = Halide::Image<T>(w,h);
    } else if (d<=1) {
        image = Halide::Image<T>(w,h,c);
    } else {
        image = Halide::Image<T>(w,h,c,d);
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
                s << std::setw(precision) << double(image(x,y)) << " ";
            }
            s << "\n";
        }
    }
    else if (image.dimensions() == 3) {
        for (size_t z=image.min(2); z<image.min(2)+image.extent(2); z++) {
            for (size_t y=image.min(1); y<image.min(1)+image.extent(1); y++) {
                for (size_t x=image.min(0); x<image.min(0)+image.extent(0); x++) {
                    s << std::setw(precision) << double(image(x,y,z)) << " ";
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
                        s << std::setw(precision) << double(image(x,y,z,w)) << " ";
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
struct CheckResult {
    Halide::Image<T> ref;   ///< reference solution
    Halide::Image<T> out;   ///< Halide solution
    CheckResult(
            Halide::Image<T> r,
            Halide::Image<T> o) :
        ref(r), out(o) {}
};

/** Compare ref and Halide solutions and print the verbose difference */
template <typename T>
struct CheckResultVerbose {
    Halide::Image<T> ref;   ///< reference solution
    Halide::Image<T> out;   ///< Halide solution
    CheckResultVerbose(
            Halide::Image<T> r,
            Halide::Image<T> o) :
        ref(r), out(o) {}
};


/** Print the synopsis of checking error */
template<typename T>
std::ostream &operator<<(std::ostream &s, const CheckResult<T> &v) {
    assert(v.ref.width()   == v.out.width());
    assert(v.ref.height()  == v.out.height());
    assert(v.ref.channels()== v.out.channels());

    int width = v.ref.width();
    int height = v.ref.height();
    int channels = v.ref.channels();

    Halide::Image<double> diff(width, height, channels);

    double re      = 0.0;
    double max_re  = 0.0;
    double mean_re = 0.0;

    for (int z=0; z<channels; z++) {
        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {
                diff(x,y,z) = v.ref(x,y,z) - v.out(x,y,z);
                re       = std::abs(diff(x,y,z)) / v.ref(x,y,z);
                mean_re += re;
                max_re   = std::max(re, max_re);
            }
        }
    }
    mean_re /= double(width*height*channels);

    s << "Max  relative error = " << 100.0*max_re << " % \n";
    s << "Mean relative error = " << 100.0*mean_re << " % \n\n";

    return s;
}

/** Print the result and synopsis of checking error */
template<typename T>
std::ostream &operator<<(std::ostream &s, const CheckResultVerbose<T> &v) {
    assert(v.ref.width()   == v.out.width());
    assert(v.ref.height()  == v.out.height());
    assert(v.ref.channels()== v.out.channels());

    int width = v.ref.width();
    int height = v.ref.height();
    int channels = v.ref.channels();

    Halide::Image<double> diff(width, height, channels);

    double re      = 0.0;
    double max_re  = 0.0;
    double mean_re = 0.0;

    for (int z=0; z<channels; z++) {
        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {
                diff(x,y,z) = (v.ref(x,y,z) - v.out(x,y,z));
                re       = std::abs(diff(x,y,z)) / v.ref(x,y,z);
                mean_re += re;
                max_re   = std::max(re, max_re);
            }
        }
    }
    mean_re /= double(width*height*channels);

    s << "Reference" << "\n" << v.ref << "\n";
    s << "Halide output" << "\n" << v.out << "\n";
    s << "Difference " << "\n" << diff << "\n";
    s << "Max  relative error = " << 100.0*max_re << " % \n";
    s << "Mean relative error = " << 100.0*mean_re << " % \n\n";

    return s;
}


#endif // _RECURSIVE_FILTER_H_
