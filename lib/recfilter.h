#ifndef _RECURSIVE_FILTER_H_
#define _RECURSIVE_FILTER_H_

#include <iomanip>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdio>

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

/** Scheduling interface for functions generated for the recursive filter */
struct RecFilterFunc {
    typedef enum {
        INTRA_TILE_nD_SCAN         = 0x001,
        INTER_TILE_1D_SCAN         = 0x002,
        INTER_TILE_2D_SCAN         = 0x004,
        INTER_TILE_3D_SCAN         = 0x008,
        INTER_TILE_4D_SCAN         = 0x010,
        INTRA_TILE_nD_SCAN_WRAPPER = 0x020,
        INTER_TILE_1D_SCAN_WRAPPER = 0x040,
        INTER_TILE_2D_SCAN_WRAPPER = 0x080,
        INTER_TILE_3D_SCAN_WRAPPER = 0x100,
        INTER_TILE_4D_SCAN_WRAPPER = 0x200
    } FuncCategory;

    typedef enum {
        INNER_PURE_VAR = 0x01,
        INNER_SCAN_VAR = 0x02,
        OUTER_PURE_VAR = 0x04,
        OUTER_SCAN_VAR = 0x08
    } VarCategory;

    Halide::Internal::Function       func;
    FuncCategory                     func_category;
    map<string, VarCategory>         pure_var_category;
    vector<map<string,VarCategory> > update_var_category;

    /** Function schedule in human readable form */
    vector<std::string> schedule_description;
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
    Halide::Internal::IntrusivePtr<RecFilterContents> contents;

public:

    /** Macros to indicate causal or anticausal scan */
    typedef enum {
        CAUSAL,     ///< causal scan
        ANTICAUSAL  ///< anticausal scan
    } Causality;

    /** Macros to determine the values of pixels before the first pixel of the scan
     * for example, all pixels with negative x indices for causal x scans
     * and all pixels with indices more than image height for anticausal y scans */
    typedef enum {
        CLAMP_TO_ZERO,  ///< pixels set to 0
        CLAMP_TO_SELF,  ///< pixels clamped to filter output
        CLAMP_TO_EXPR,  ///< pixels set to a given expression
    } Border;

    /** Construct an empty named recursive filter */
    RecFilter(std::string name = "RecFilter");

    /** Reconstruct a recursive filter from its contents */
    RecFilter(const Halide::Internal::IntrusivePtr<RecFilterContents> &c) : contents(c) {}

    /**@name Compile and run
     */
    // {@

    /** Finalize the filter; triggers automatic function transformations and cleanup */
    void finalize(void);

    /** Trigger JIT compilation for specified hardware-platform target; dumps the generated
     * codegen in human readable HTML format if filename is specified */
    void compile_jit(Halide::Target target, std::string filename="");

    /** Compute the filter for a given output buffer for specified number of iterations
     * for timing purposes; last iteration copies the result to host */
    void realize(Halide::Buffer out, int iterations);
    // @}

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
    void define(Halide::Expr  pure_def);
    void define(Halide::Tuple pure_def);
    void define(std::vector<Halide::Expr> pure_def);
    // @}

    /** @name Routines to add scans to a recursive filter
     *  @brief Add a scan to the recursive filter given parameters
     *  defaults filter order = 1, feedforward/feedback coefficient = 1.0,
     *  causalilty = CAUSAL, border clamping = zero
     */
    // {@
    void addScan(
            Halide::Var x,              ///< dimension to a update
            Halide::RDom rx,            ///< domain of the scan
            float feedfwd,              ///< single feedforward coeff
            std::vector<float> feedback,///< n feedback coeffs, where n is filter order
            Causality c=CAUSAL,         ///< causal or anticausal scan
            Border b=CLAMP_TO_ZERO,     ///< value for pixels before first pixel of scan
            Halide::Expr border_expr=FLOAT_ZERO ///< user defined value if CLAMP_TO_EXPR is used (must not involve x or rx)
            );
    void addScan(
            Halide::Var x,              ///< dimension to a update
            Halide::RDom rx,            ///< domain of the scan
            Causality c=CAUSAL,         ///< causal or anticausal scan
            Border b=CLAMP_TO_ZERO,     ///< value for pixels before first pixel of scan
            Halide::Expr border_expr=FLOAT_ZERO ///< user defined value if CLAMP_TO_EXPR is used (must not involve x or rx)
            );
    void addScan(
            Halide::Var x,              ///< dimension to a update
            Halide::RDom rx,            ///< domain of the scan
            std::vector<float> feedback,///< n feedback coeffs, where n is filter order
            Causality c=CAUSAL,         ///< causal or anticausal scan
            Border b=CLAMP_TO_ZERO,     ///< value for pixels before first pixel of scan
            Halide::Expr border_expr=FLOAT_ZERO ///< user defined value if CLAMP_TO_EXPR is used (must not involve x or rx)
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
    std::vector<Halide::Func> funcs(void);
    // @}


    /**@name Splitting routines
     * @brief Split a list of dimensions into their respective tile widths specified as
     * variable-tile width pairs. If a single tile width expression is provided, then all
     * specified dimensions are split by the same factor. If no variables are specified,
     * then all the dimensions are split by the same tiling factor (defined in split.cpp).
     *
     * Preconditions:
     * - dimension with specified variable name must exist
     * - tile width must be a multiple of image width for each dimension
     */
    // {@
    void split(Halide::Expr tx);
    void split(Halide::Var x, Halide::Expr tx);
    void split(Halide::Var x, Halide::Expr tx, Halide::Var y, Halide::Expr ty);
    void split(Halide::Var x, Halide::Var y, Halide::Expr t);
    void split(Halide::Var x, Halide::Var y, Halide::Var z, Halide::Expr t);
    void split(std::vector<Halide::Var> vars, Halide::Expr t);
    void split(std::map<std::string, Halide::Expr> dims);
    // @}


    /** @name Reorder scans in the filter or cascade them to produce multiple filters
     *  @brief uses list of list of scans as argument, producing a list of
     *  recursive filters each containing the respective list of scans
     *  (defined in reorder.cpp)
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

    /** @name Scheduling language: */
    // {@
    // @}


    /** @name  Inline all calls to a pure function (defined in reorder.cpp)
     */
    // {@
    void inline_func(
            std::string func_name   ///< name of function to be inlined
            );

    void inline_func(
            Halide::Func a,         ///< function to be inlined
            Halide::Func b          ///< function in which to inline
            );
    // @}


    /** @name Reorder memory layout by swapping two dimensions of a function
     * (defined in reorder.cpp) */
    // {@
    void transpose_dimensions(
            std::string func,   ///< name of function whose dimensions must be swapped
            Halide::Var a,      ///< pure arg of first dimension to swap
            Halide::Var b       ///< pure arg of second dimension to swap
            );
    void transpose_dimensions(
            std::string func,   ///< name of function whose dimensions must be swapped
            std::string a,      ///< pure arg of first dimension to swap
            std::string b       ///< pure arg of second dimension to swap
            );
    // @}

    /**@name Merging and interleaving routines
     *
     * @brief Interleave two functions into a single function with output that contains
     * the buffers of both the input functions.  The functions to be interleaved are
     * searched in the dependency graph of functions required to compute the recursive filter
     * (defined in reorder.cpp)
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

    /**
     * @brief Merge multiple functions into a single function with mutiple outputs
     * The functions to be merged are searched in the dependency graph of functions
     * required to compute the recursive filter (defined in reorder.cpp)
     *
     * Preconditions: functions to be merged must have
     * - same pure args
     * - scans with same update args and update domains in same order
     */
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
    // @}

    /** Remove the pure def of a Function and add it to the first update
     * def; replacing the pure def with zero or undefine */
    void remove_pure_def(
            std::string func_name  ///< name of function
            );

private:
    /** Recursive filter function */
    RecFilterFunc& function(std::string func_name);
};

// -----------------------------------------------------------------------------

#ifndef WIN32
# include <sys/time.h>
# include <sys/stat.h>
# include <sys/types.h>
# include <dirent.h>
#else
# include <windows.h>
# include <psapi.h>
# include <shlwapi.h>
#endif

#include "recfilter.h"


/** @name Printing utils for recursive filter, Halide functions and
 * difference between computed result and reference result  */
// {@
std::ostream &operator<<(std::ostream &s, RecFilter r);
std::ostream &operator<<(std::ostream &s, RecFilterFunc f);
std::ostream &operator<<(std::ostream &s, Halide::Func f);
std::ostream &operator<<(std::ostream &s, Halide::Internal::Function f);
std::ostream &operator<<(std::ostream &s, CheckResult v);
std::ostream &operator<<(std::ostream &s, CheckResultVerbose v);
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

/** Cross platform timer.
 * Platform independent class to record time elapsed. The timer
 * starts counting automatically as soon as an object of this
 * class is allocated and prints the time elapsed as soon as the
 * object is deallocated (or goes out of scope).
 */
class Timer {
protected:
    typedef long long t_time;   ///< data type for storing timestamps
    std::string m_Name;         ///< name of timer, default = Timer
    t_time m_TmStart;           ///< starting time stamp

public:

    /** Start the timer */
    Timer(std::string name="Timer");

    /** Stop the timer */
    ~Timer(void);

    /** Start the timer by recording the initial time stamp */
    void start(void);

    /** Stop the timer and print time elapsed */
    t_time stop(void);

    /** Get the total time elapsed in milliseconds */
    t_time elapsed(void);

    /** Platform independent function to get current timestamp in milliseconds */
    t_time milliseconds(void);
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
