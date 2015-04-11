#ifndef _MODIFIERS_H_
#define _MODIFIERS_H_

#include <Halide.h>

/** Return true if all Expr in the vector are undef */
bool is_undef(Halide::Expr e);

/** Return true if all Expr in the vector are undef */
bool is_undef(std::vector<Halide::Expr> e);

/** Return true if all elements of the Tuple are undef */
bool is_undef(Halide::Tuple t);

/** Check if a given expression depends upon a variable */
bool expr_depends_on_var(
        Halide::Expr expr,      ///< expression to be checked
        std::string var         ///< variable name
        );

/** Check if a given expression contains calls to a Function */
bool expr_depends_on_func(
        Halide::Expr expr,      ///< expression to be checked
        std::string func_name   ///< name of Function
        );


/** Substitute a variable in all calls to a particular Function
 * in a given expression
 */
Halide::Expr substitute_in_func_call(
        std::string func_name,  ///< name of Function
        std::string var,        ///< variable to be substituted
        Halide::Expr replace,   ///< replacement expression
        Halide::Expr original   ///< original expression
        );

/** Substitute all calls to a particular Function by a given expression
 * the calling args match the given list
 */
Halide::Expr substitute_func_call_with_args(
        std::string func_name,                  ///< name of Function
        std::vector<Halide::Expr> call_args,    ///< list of calling args to match
        Halide::Expr replace,                   ///< replacement expression
        Halide::Expr original                   ///< original expression
        );

/** Add a calling argument to all calls to a particular Function */
Halide::Expr insert_arg_in_func_call(
        std::string func_name,  ///< name of Function
        size_t pos,             ///< position to add calling arg within list
        Halide::Expr arg,       ///< calling argument to add
        Halide::Expr original   ///< original expression
        );

/** Remove a calling argument from all calls to a particular Function */
Halide::Expr remove_arg_from_func_call(
        std::string func_name,  ///< name of Function
        size_t pos,             ///< calling arg index within list to be removed
        Halide::Expr original   ///< original expression
        );

/** Swap two calling arguments of given function in the expression */
Halide::Expr swap_args_in_func_call(
        std::string func_name,  ///< name of function to modify
        size_t va_idx,          ///< index of first calling arg
        size_t vb_idx,          ///< index of second calling arg
        Halide::Expr original   ///< original expression
        );

/** Substitute a calling arg in feed forward recursive calls
 * to a Func; feedforward calls are those where all args in the
 * Function call are identical to args in Function definition
 */
Halide::Expr substitute_arg_in_feedforward_func_call(
        std::string func_name,              ///< name of function to modify
        std::vector<Halide::Expr> def_arg,  ///< args in Function definition
        size_t pos,                         ///< index of calling arg to be replaced
        Halide::Expr new_arg,               ///< new calling arg
        Halide::Expr original               ///< original expression
        );

/** Apply zero boundary condition on all tiles except tiles that touch image
 * borders; this pads tiles by zeros if the intra tile index is out of range -
 * - less than zero or greater than tile width.
 * Usually, if the feedforward coeff is 1.0 zero padding is required for in
 * the Function definition itself - which automatically pads all tiles by zeros.
 * But in Gaussian filtering feedforward coeff is not 1.0; so Function definition
 * cannot pad the image by zeros; in such case it is important to forcibly pad
 * all inner tiles by zeros.
 */
Halide::Expr apply_zero_boundary_in_func_call(
        std::string func_name,  ///< name of function to modify
        size_t dim,             ///< dimension containing the RVar
        Halide::Expr def_arg,   ///< args in Function definition
        Halide::Expr boundary,  ///< tile width
        Halide::Expr cond,      ///< condition to check tiles touching borders
        Halide::Expr original   ///< original expression
        );

/** Mathematically add an expression to all calls to a particular
 *  Function; the expression to be added is selected from a list of
 *  expressions depending upon the value index of the Function call;
 *  all occurances of arguments of the Function must be replaced by
 * calling arguments in the selected expression before adding
 */
Halide::Expr augment_func_call(
        std::string func_name,              ///< name of Function
        std::vector<std::string> func_args, ///< arguments of Function
        std::vector<Halide::Expr> extra,    ///< list of exprs to be algebraically added
        Halide::Expr original               ///< original expression
        );


/** Substitute all calls to a particular Function by
 * a new Function with the same calling arguments
 */
Halide::Expr substitute_func_call(
        std::string func_name,              ///< name of Function
        Halide::Internal::Function replace, ///< replacement Function
        Halide::Expr original               ///< original expression
        );


/** Remove all calls to a particular Function or to all
 * Functions except for a particular Function by identity
 * as determined by a boolean flag argument, all calls
 * to the Function are removed if flag is set
 */
Halide::Expr remove_func_calls(
        std::string func_name,  ///< name of Function
        bool matching,          ///< remove calls matching or not matching
        Halide::Expr original   ///< original expression
        );


/** Find all calls to a particular Function and increment
 * the value_index (Halide::Internal::Call::value_index)
 * of the call by a given offset
 */
Halide::Expr increment_value_index_in_func_call(
        std::string func_name,  ///< name of Function
        int increment,          ///< increment to value_index, can be negative
        Halide::Expr original   ///< original expression
        );

/** Swaps two variables in an expression */
Halide::Expr swap_vars_in_expr(
        std::string a,          ///< first variable name for swapping
        std::string b,          ///< second variable name for swapping
        Halide::Expr original   ///< original expression
        );

/** Remove all lets by inlining; all sub-expressions will exist once
 * in memory, but may have many pointers to them, so this doesn't cause
 * a combinatorial explosion; if you walk over this as if it were a tree,
 * however, you're going to have a bad time */
Halide::Expr remove_lets(Halide::Expr);

/** Extract the list of buffers called in the definition a function */
std::map<std::string, Halide::Buffer> extract_buffer_calls(Halide::Func func);

/** Make sure that all vars with tags INNER, OUTER or FULL have VarTag count
 * in the same order as they appear in the function definition; this continuity
 * was broken during tiling or by RecFilterSchedule::split()
 *
 * \param[in,out] var_tags list of variable tags to be modified
 * \param[in] args pure definition args
 * \param[in] var_splits list of variables created by RecFilterSchedule::split() and the corresponding old variables for the pure definition
 */
void reassign_vartag_counts(
        std::map<std::string,VarTag>& var_tags,
        std::vector<std::string> args,
        std::map<std::string,std::string> var_splits);

/** Make sure that all vars with tags INNER, OUTER or FULL have VarTag count
 * in the same order as they appear in the function definition; this continuity
 * was broken during tiling or by RecFilterSchedule::split()
 *
 * \param[in,out] var_tags list of variable tags to be modified
 * \param[in] args update definition args
 * \param[in] var_splits list of variables created by RecFilterSchedule::split() and the corresponding old variables for this update definition
 */
void reassign_vartag_counts(
        std::map<std::string,VarTag>& var_tags,
        std::vector<Halide::Expr> args,
        std::map<std::string,std::string> var_splits);

#endif // _MODIFIERS_H_
