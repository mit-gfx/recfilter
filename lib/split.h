#ifndef _SPLIT_H_
#define _SPLIT_H_

#include <iomanip>
#include <vector>
#include <queue>
#include <string>
#include <stdexcept>
#include <cstdio>

#include <Halide.h>

// ----------------------------------------------------------------------------

#define INTRA_TILE_RESULT     "Intra"
#define INTRA_TILE_SUB_RESULT "Intra2"
#define INTRA_TILE_TAIL_TERM  "Tail"
#define INTER_TILE_TAIL_SUM   "CTail"
#define INTER_TILE_DEPENDENCY "Deps"

#define DELIM_START '$'
#define DELIM_END   '$'

// ----------------------------------------------------------------------------

// Printing Utils

std::ostream &operator<<(std::ostream &s, Halide::Func f);

std::ostream &operator<<(std::ostream &s, Halide::Internal::Function f);

// ----------------------------------------------------------------------------

// Splitting routine

void split(
        Halide::Func& F,
        Halide::Func  filter_weights,
        std::vector<int>          dimensions,
        std::vector<Halide::Var>  vars,
        std::vector<Halide::Var>  inner_vars,
        std::vector<Halide::Var>  outer_vars,
        std::vector<Halide::RDom> rdoms,
        std::vector<Halide::RDom> inner_rdoms,
        std::vector<int> orders);

void split(
        Halide::Func& F,
        std::vector<int>          dimensions,
        std::vector<Halide::Var>  vars,
        std::vector<Halide::Var>  inner_vars,
        std::vector<Halide::Var>  outer_vars,
        std::vector<Halide::RDom> rdoms,
        std::vector<Halide::RDom> inner_rdoms,
        std::vector<int> orders);

// ----------------------------------------------------------------------------

// Reordering routines

void merge_and_inline(Halide::Func S,
        std::string func_a,
        std::string func_b,
        std::string merged);

void merge_and_inline(Halide::Func S,
        std::string func_a,
        std::string func_b,
        std::string func_c,
        string merged);

void merge_and_inline(Halide::Func S,
        std::string func_a,
        std::string func_b,
        std::string func_c,
        std::string func_d,
        string merged);

void merge(Halide::Func S,
        std::string func_a,
        std::string func_b,
        std::string merged);

void merge(Halide::Func S,
        std::string func_a,
        std::string func_b,
        std::string func_c,
        string merged);

void merge(Halide::Func S,
        std::string func_a,
        std::string func_b,
        std::string func_c,
        std::string func_d,
        string merged);

void merge(Halide::Func A, Halide::Func B);

void inline_function(Halide::Func F, std::string func_name);

void inline_function(Halide::Func F, Halide::Func A);

void inline_non_split_functions(Halide::Func F, size_t num_splits);

void inline_pure_functions(Halide::Func F);

void float_dependencies_to_root(Halide::Func F);

void swap_variables(Halide::Func F,
        std::string func_name,
        Halide::Var a,
        Halide::Var b);

void expand_multiple_reductions(Halide::Func S);

// ----------------------------------------------------------------------------

// Modifier routines

/// Check if a given expression depends upon a variable
bool expr_depends_on_var(
        Halide::Expr expr,      /// expression to be checked
        std::string var         /// variable name
        );


/// Check if a given expression contains calls to a Function
bool expr_depends_on_func(
        Halide::Expr expr,              /// expression to be checked
        std::string func_name           /// name of Function
        );


/// Substitute a variable in all calls to a particular Function
/// in a given expression
Halide::Expr substitute_in_func_call(
        std::string func_name,          /// name of Function
        std::string var,                /// variable to be substituted
        Halide::Expr replace,           /// replacement expression
        Halide::Expr original           /// original expression
        );


/// Add a calling argument to all calls to a particular Function
Halide::Expr insert_arg_to_func_call(
        std::string func_name,          /// name of Function
        size_t pos,                     /// position to add calling arg within list
        Halide::Expr arg,               /// calling argument to add
        Halide::Expr original           /// original expression
        );


/// Mathematically add an expression to all calls to a particular
/// Function; the expression to be added is selected from a list of
/// expressions depending upon the value index of the Function call;
/// all occurances of arguments of the Function must be replaced by
/// calling arguments in the selected expression before adding
Halide::Expr augment_func_call(
        std::string func_name,              /// name of Function
        std::vector<std::string> func_args, /// arguments of Function
        std::vector<Halide::Expr> extra,    /// list of expression to be mathematically added
        Halide::Expr original               /// original expression
        );


/// Substitute all calls to a particular Function by
/// a new Function with the same calling arguments
Halide::Expr substitute_func_call(
        std::string func_name,              /// name of Function
        Halide::Internal::Function replace, /// replacement Function
        Halide::Expr original               /// original expression
        );


/// Remove all calls to a particular Function or to all
/// Functions except for a particular Function by identity
/// as determined by a boolean flag argument, all calls
/// to the Function are removed if flag is set
Halide::Expr remove_func_calls(
        std::string func_name,              /// name of Function
        bool matching,                      /// remove calls matching or not matching
        Halide::Expr original               /// original expression
        );


/// Find all calls to a particular Function and increment
/// the value_index (Halide::Internal::Call::value_index)
/// of the call by a given offset
Halide::Expr increment_value_index_in_func_call(
        std::string func_name,              /// name of Function
        int increment,                      /// increment to value_index, can be negative
        Halide::Expr original               /// original expression
        );


/// Check if an expression has calls to a particular Function
/// and inline the Function within the expression
Halide::Expr inline_func_calls(
        Halide::Internal::Function func,    /// Function to be inlined
        Halide::Expr original               /// original expression
        );

/// Swaps two calling arguments of given function in the expression
Halide::Expr swap_callargs_in_func_call(
        string func_name,       /// name of function to modify
        int va_idx,             ///.index of first calling arg
        int vb_idx,             /// index of second calling arg
        Halide::Expr original   /// original expression
        );

/// Swaps two variables in an expression
Halide::Expr swap_vars_in_expr(
        string a,               /// first variable name for swapping
        string b,               /// second variable name for swapping
        Halide::Expr original   /// original expression
        );

/// Extract all the variables referenced in a Expr
std::vector<string> extract_vars_in_expr(Halide::Expr expr);

/// Extract the list of dependencies of a Function -
/// Functions called in the pure and reduction definitions
/// of the given Function as well as its dependecies -
/// returns the complete dependency graph of the function.
void extract_func_calls(
        Halide::Func func,                  /// Function to be analyzed
        std::vector<Halide::Func>& f_list   /// list returned as output
        );

// ----------------------------------------------------------------------------

// Command line args

class Arguments {
public:
    int width;       // image width
    int height;      // image height
    int block;       // block size
    bool debug;      // display intermediate stages
    bool verbose;    // display input, reference output, halide output, difference
    bool nocheck;    // skip check Halide result against reference solution
    float weight;    // First order filter weight
    int  iterations; // profiling iterations

    Arguments(string app_name, int argc, char** argv);
};

// ----------------------------------------------------------------------------

// Cross platform timer

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

/** Timer class.
 * Platform independent class to record time elapsed. The timer
 * starts counting automatically as soon as an object of this
 * class is allocated and prints the time elapsed as soon as the
 * object is deallocated (or goes out of scope).
 */
class Timer {
protected:

    /** Data type for storing timestamps. */
    typedef long long t_time;

    /** name of timer, default = [Timer]*/
    std::string m_Name;

    /** Starting time stamp */
    t_time m_TmStart;

public:

    /** Start the timer. */
    Timer(std::string name="Timer");

    /** Stop the timer. */
    ~Timer(void);

    /** Start the timer by recording the initial time stamp */
    void start(void);

    /** Stop the timer and print time elapsed. */
    t_time stop(void);

    /** Get the total time elapsed in milliseconds */
    t_time elapsed(void);

    /** Platform independent function to get current timestamp in milliseconds */
    t_time milliseconds(void);

};

// ----------------------------------------------------------------------------

// Reference image computation utils

#define MIN_ELEMENT 1
#define MAX_ELEMENT 1
#define PRINT_WIDTH 3

template<typename T>
Halide::Image<T> generate_random_image(int width) {
    Halide::Image<T> image(width);
    for (int x=0; x<width; x++) {
        image(x) = T(MIN_ELEMENT + (rand() % MAX_ELEMENT));
    }
    return image;
}

template<typename T>
Halide::Image<T> generate_random_image(int width, int height) {
    Halide::Image<T> image(width,height);
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            image(x,y) = T(MIN_ELEMENT + (rand() % MAX_ELEMENT));
        }
    }
    return image;
}


template<typename T>
std::ostream &operator<<(std::ostream &s, Halide::Image<T> image) {
    if (image.dimensions() > 1) {
        for (size_t y=0; y<image.height(); y++) {
            for (size_t x=0; x<image.width(); x++) {
                s << std::setw(PRINT_WIDTH) << image(x,y) << " ";
            }
            s << "\n";
        }
    } else {
        for (size_t x=0; x<image.width(); x++)
            s << std::setw(PRINT_WIDTH) << image(x) << " ";
        s << "\n";
    }
    return s;
}

template<typename T>
Halide::Image<T> reference_recursive_filter(Halide::Image<T> in,
        Halide::Image<T> weights)
{
    int width = in.width();
    int height= in.height();

    int order_x = weights.height();
    int order_y = weights.height();

    Halide::Image<T> ref(width,height);

    for (int y=0; y<height; y++) {          // init the solution
        for (int x=0; x<width; x++) {
            ref(x,y) = in(x,y);
        }
    }

    for (int y=0; y<height; y++) {          // x filtering
        for (int x=0; x<width; x++) {
            for (int k=1; k<=order_x; k++) {
                ref(x,y) += (x>=k ? weights(0,k-1)*ref(x-k,y) : T(0));
            }
        }
    }

    for (int y=0; y<height; y++) {          // y filtering
        for (int x=0; x<width; x++) {
            for (int k=1; k<=order_y; k++) {
                ref(x,y) += (y>=k ? weights(1,k-1)*ref(x,y-k) : T(0));
            }
        }
    }

    return ref;
}

#endif // _SPLIT_H_
