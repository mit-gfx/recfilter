#ifndef _RECURSIVE_FILTER_H_
#define _RECURSIVE_FILTER_H_

#include <iomanip>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdio>
#include <algorithm>

#include <Halide.h>

/** Symbolic integer zero  */
#define INT_ZERO Halide::Internal::make_zero(Halide::type_of<int>())

/** Symbolic floating point zero  */
#define FLOAT_ZERO Halide::Internal::make_zero(Halide::type_of<float>())

/** Symbolic integer one */
#define INT_ONE  Halide::Internal::make_one(Halide::type_of<int>())

/** Symbolic floating point one */
#define FLOAT_ONE  Halide::Internal::make_one(Halide::type_of<float>())

/** Symbolic undefined integer point constant */
#define INT_UNDEF Halide::undef<int>()

/** Symbolic undefined floating point constant */
#define FLOAT_UNDEF Halide::undef<float>()

// -----------------------------------------------------------------------------

// Forward declarations of internal structures

struct FilterInfo;
struct RecFilterContents;
class RecFilterFunc;
class RecFilterSchedule;

class FuncTag;
class VarTag;

enum VariableTag : int;
enum FunctionTag : int;

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
    RecFilter(void);

    /**@name Recursive filter specification */
    // {@
    /** Create a 1D recursive filter given variable and size of input/output buffer in that dimension */
    RecFilter(Halide::Var x, Halide::Expr wx, std::string name="R");

    /** Create a 2D recursive filter given variable and size of input/output buffer in that dimension */
    RecFilter(Halide::Var x, Halide::Expr wx, Halide::Var y, Halide::Expr wy, std::string name="R");

    /** Create a 3D recursive filter given variable and size of input/output buffer in that dimension */
    RecFilter(Halide::Var x, Halide::Expr wx, Halide::Var y, Halide::Expr wy, Halide::Var z, Halide::Expr wz, std::string name="R");

    /** Create a n-D recursive filter given variable and size of input/output buffer in that dimension */
    RecFilter(std::vector<Halide::Var> args, std::vector<Halide::Expr> widths, std::string name);
    // @}

    /**@name Compile and run */
    // {@
    /** Finalize the filter; triggers automatic function transformations and cleanup */
    void finalize(Halide::Target target);

    /** Trigger JIT compilation for specified hardware-platform target; dumps the generated
     * codegen in human readable HTML format if filename is specified */
    void compile_jit(Halide::Target target, std::string filename="");

    /** Compute the filter for a given output buffer for specified number of iterations
     * for timing purposes; last iteration copies the result to host */
    void realize(Halide::Buffer out, int iterations=1);
    // @}


    /** @name Recursive filter definition
     * @brief Add a pure definition to the recursive filter, can be Tuple
     *
     * Preconditions:
     * - all Halide::Vars in the pure definition must be args of the filter
     */
    // {@
    void define(Halide::Expr  pure_def);
    void define(Halide::Tuple pure_def);
    void define(std::vector<Halide::Expr> pure_def);
    // @}

    /** @name Routines to add filters
     *
     *  @brief Add a causal or anticausal scan to the recursive filter with given
     *  feedback and feed forward coefficients
     *
     * Preconditions:
     * - first argument must be an arg of the filter
     */
    // {@
    void add_filter(
            Halide::Var x,            ///< dimension to a update
            std::vector<float> coeff, ///< 1 feedforward and n feedback coeffs (n = filter order)
            bool causal               ///< causal or anticausal filter
            );

    /** Convenience routine to add causal filter */
    void add_causal_filter(
            Halide::Var x,            ///< dimension to a update
            std::vector<float> coeff  ///< 1 feedforward and n feedback coeffs (n = filter order)
            );

    /** Convenience routine to add anticausal filter */
    void add_anticausal_filter(
            Halide::Var x,            ///< dimension to a update
            std::vector<float> coeff  ///< 1 feedforward and n feedback coeffs (n = filter order)
            );
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
    void split(Halide::Var x, Halide::Expr tx);
    void split(Halide::Var x, Halide::Expr tx, Halide::Var y, Halide::Expr ty);
    void split(Halide::Var x, Halide::Expr tx, Halide::Var y, Halide::Expr ty, Halide::Var z, Halide::Expr tz);
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


    /**@name Merging and interleaving routines
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
     * \param id 0 for all intra tile functions, 1 for 1D intra-tile functions, otherwise nD intra-tile functions
     */
    RecFilterSchedule intra_schedule(int id=0);

    /** Extract a handle to schedule intra-tile functions */
    RecFilterSchedule inter_schedule(void);
    // @}


    /** @name Generic handles to write scheduled for dimensions of internal functions */
    // {@
    VarTag full_parallel (int id=0);
    VarTag inner_parallel(int id=0);
    VarTag outer_parallel(int id=0);
    VarTag full_scan     (int id=0);
    VarTag inner_scan    (void);
    VarTag inner_tail    (void);
    VarTag outer_scan    (void);
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

    RecFilterSchedule& compute_in_global();
    RecFilterSchedule& compute_in_shared();

    RecFilterSchedule& unroll(VarTag v);
    RecFilterSchedule& unroll(VarTag v, int factor);

    RecFilterSchedule& vectorize(VarTag v);
    RecFilterSchedule& vectorize(VarTag v, int factor);

    RecFilterSchedule& inner_split(VarTag v, int factor);
    RecFilterSchedule& outer_split(VarTag v, int factor);
    RecFilterSchedule& split   (VarTag v, int factor, bool do_reorder);

    RecFilterSchedule& reorder(VarTag x, VarTag y);
    RecFilterSchedule& reorder(VarTag x, VarTag y, VarTag z);
    RecFilterSchedule& reorder(VarTag x, VarTag y, VarTag z, VarTag w);
    RecFilterSchedule& reorder(VarTag x, VarTag y, VarTag z, VarTag w, VarTag t);

    RecFilterSchedule& reorder_storage(VarTag x, VarTag y);
    RecFilterSchedule& reorder_storage(VarTag x, VarTag y, VarTag z);
    RecFilterSchedule& reorder_storage(VarTag x, VarTag y, VarTag z, VarTag w);
    RecFilterSchedule& reorder_storage(VarTag x, VarTag y, VarTag z, VarTag w, VarTag t);

    RecFilterSchedule& parallel(VarTag v);
    RecFilterSchedule& parallel(VarTag v, int task_size);

    RecFilterSchedule& gpu_threads(VarTag vt1, int t1);
    RecFilterSchedule& gpu_threads(VarTag vt1, int t1, VarTag vt2, int t2);
    RecFilterSchedule& gpu_threads(VarTag vt1, int t1, VarTag vt2, int t2, VarTag vt3, int t3);

    RecFilterSchedule& gpu_blocks(VarTag vt1);
    RecFilterSchedule& gpu_blocks(VarTag vt1, VarTag vt2);
    RecFilterSchedule& gpu_blocks(VarTag vt1, VarTag vt2, VarTag vt3);

    RecFilterSchedule& reorder (std::vector<VarTag> x);
    RecFilterSchedule& reorder_storage(std::vector<VarTag> x);

///    RecFilterSchedule& gpu_tile(std::vector<std::pair<VarTag,int> > x);
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
    int    check     (const VariableTag &t) const;
    int    count     (void) const;

    void decrement_count(void);
    void increment_count(void);

private:
    VariableTag tag;
};

// -----------------------------------------------------------------------------

/** @name Printing utils for recursive filter, Halide functions, schedules and
 * difference between computed result and reference result */
// {@
std::ostream &operator<<(std::ostream &s, const RecFilter &r);
std::ostream &operator<<(std::ostream &s, const RecFilterFunc &f);
std::ostream &operator<<(std::ostream &s, const Halide::Func &f);
std::ostream &operator<<(std::ostream &s, const Halide::Internal::Function &f);
std::ostream &operator<<(std::ostream &s, const CheckResult &v);
std::ostream &operator<<(std::ostream &s, const CheckResultVerbose &v);
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

#define MIN_ELEMENT 1
#define MAX_ELEMENT 1
#define PRINT_WIDTH 3

/** Generate an image of a given size with random entries */
template<typename T>
Halide::Image<T> generate_random_image(size_t w, size_t h=1, size_t c=1, size_t d=1) {
    Halide::Image<T> image(w,h,c,d);
    for (size_t t=0; t<d; t++) {
        for (size_t z=0; z<c; z++) {
            for (size_t y=0; y<h; y++) {
                for (size_t x=0; x<w; x++) {
                    image(x,y,z,t) = T(MIN_ELEMENT + (rand() % MAX_ELEMENT));
                }
            }
        }
    }
    return image;
}


/** Print an image */
template<typename T>
std::ostream &operator<<(std::ostream &s, Halide::Image<T> image) {
    if (image.dimensions() == 1) {
        for (size_t x=image.min(0); x<image.min(0)+image.extent(0); x++) {
            s << std::setw(PRINT_WIDTH) << image(x) << " ";
        }
        s << "\n";
    }

    if (image.dimensions() == 2) {
        for (size_t y=image.min(1); y<image.min(1)+image.extent(1); y++) {
            for (size_t x=image.min(0); x<image.min(0)+image.extent(0); x++) {
                s << std::setw(PRINT_WIDTH) << image(x,y) << " ";
            }
            s << "\n";
        }
    }

    if (image.dimensions() == 3) {
        for (size_t z=image.min(2); z<image.min(2)+image.extent(2); z++) {
            for (size_t y=image.min(1); y<image.min(1)+image.extent(1); y++) {
                for (size_t x=image.min(0); x<image.min(0)+image.extent(0); x++) {
                    s << std::setw(PRINT_WIDTH) << image(x,y,z) << " ";
                }
                s << "\n";
            }
            s << "--\n";
        }
    }

    if (image.dimensions() == 4) {
        for (size_t w=image.min(3); w<image.min(3)+image.extent(3); w++) {
            for (size_t z=image.min(2); z<image.min(2)+image.extent(2); z++) {
                for (size_t y=image.min(1); y<image.min(1)+image.extent(1); y++) {
                    for (size_t x=image.min(0); x<image.min(0)+image.extent(0); x++) {
                        s << std::setw(PRINT_WIDTH) << image(x,y,z,w) << " ";
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

#endif // _RECURSIVE_FILTER_H_
