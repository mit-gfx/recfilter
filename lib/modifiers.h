#ifndef _MODIFIERS_H_
#define _MODIFIERS_H_

#include <Halide.h>

/** @brief Check if a given expression depends upon a variable
 *  (defined in modifiers.cpp) */
bool expr_depends_on_var(
        Halide::Expr expr,      ///< expression to be checked
        std::string var         ///< variable name
        );

/** @brief Check if a given expression contains calls to a Function
 *  (defined in modifiers.cpp) */
bool expr_depends_on_func(
        Halide::Expr expr,      ///< expression to be checked
        std::string func_name   ///< name of Function
        );


/** @brief Substitute a variable in all calls to a particular Function
 * in a given expression
 *  (defined in modifiers.cpp)
 */
Halide::Expr substitute_in_func_call(
        std::string func_name,  ///< name of Function
        std::string var,        ///< variable to be substituted
        Halide::Expr replace,   ///< replacement expression
        Halide::Expr original   ///< original expression
        );

/** @brief Add a calling argument to all calls to a particular Function
 *  (defined in modifiers.cpp) */
Halide::Expr insert_arg_in_func_call(
        std::string func_name,  ///< name of Function
        size_t pos,             ///< position to add calling arg within list
        Halide::Expr arg,       ///< calling argument to add
        Halide::Expr original   ///< original expression
        );

/** @brief Remove a calling argument from all calls to a particular Function
 *  (defined in modifiers.cpp) */
Halide::Expr remove_arg_from_func_call(
        std::string func_name,  ///< name of Function
        size_t pos,             ///< calling arg index within list to be removed
        Halide::Expr original   ///< original expression
        );

/** @brief Swap two calling arguments of given function in the expression
 *  (defined in modifiers.cpp) */
Halide::Expr swap_args_in_func_call(
        std::string func_name,  ///< name of function to modify
        size_t va_idx,          ///< index of first calling arg
        size_t vb_idx,          ///< index of second calling arg
        Halide::Expr original   ///< original expression
        );

/** @brief Substitute a calling arg in feed forward recursive calls
 * to a Func; feedforward calls are those where all args in the
 * Function call are identical to args in Function definition
 *  (defined in modifiers.cpp)
 */
Halide::Expr substitute_arg_in_feedforward_func_call(
        std::string func_name,              ///< name of function to modify
        std::vector<Halide::Expr> def_arg,  ///< args in Function definition
        size_t pos,                         ///< index of calling arg to be replaced
        Halide::Expr new_arg,               ///< new calling arg
        Halide::Expr original               ///< original expression
        );

/** @brief Apply zero boundary condition on all tiles except tiles that touch image
 * borders; this pads tiles by zeros if the intra tile index is out of range -
 * - less than zero or greater than tile width (defined in modifiers.cpp).
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

/** @brief Mathematically add an expression to all calls to a particular
 *  Function; the expression to be added is selected from a list of
 *  expressions depending upon the value index of the Function call;
 *  all occurances of arguments of the Function must be replaced by
 * calling arguments in the selected expression before adding
 *  (defined in modifiers.cpp)
 */
Halide::Expr augment_func_call(
        std::string func_name,              ///< name of Function
        std::vector<std::string> func_args, ///< arguments of Function
        std::vector<Halide::Expr> extra,    ///< list of exprs to be algebraically added
        Halide::Expr original               ///< original expression
        );


/** @brief Substitute all calls to a particular Function by
 * a new Function with the same calling arguments
 *  (defined in modifiers.cpp)
 */
Halide::Expr substitute_func_call(
        std::string func_name,              ///< name of Function
        Halide::Internal::Function replace, ///< replacement Function
        Halide::Expr original               ///< original expression
        );


/** @brief Remove all calls to a particular Function or to all
 * Functions except for a particular Function by identity
 * as determined by a boolean flag argument, all calls
 * to the Function are removed if flag is set
 *  (defined in modifiers.cpp)
 */
Halide::Expr remove_func_calls(
        std::string func_name,  ///< name of Function
        bool matching,          ///< remove calls matching or not matching
        Halide::Expr original   ///< original expression
        );


/** @brief Find all calls to a particular Function and increment
 * the value_index (Halide::Internal::Call::value_index)
 * of the call by a given offset
 *  (defined in modifiers.cpp)
 */
Halide::Expr increment_value_index_in_func_call(
        std::string func_name,  ///< name of Function
        int increment,          ///< increment to value_index, can be negative
        Halide::Expr original   ///< original expression
        );

/** @brief Swaps two variables in an expression
 *  (defined in modifiers.cpp) */
Halide::Expr swap_vars_in_expr(
        std::string a,          ///< first variable name for swapping
        std::string b,          ///< second variable name for swapping
        Halide::Expr original   ///< original expression
        );

/**@name Extract vars referenced in a Expr
 *  (defined in modifiers.cpp) */
// {@
std::vector<std::string> extract_rvars_in_expr(Halide::Expr expr);
std::vector<std::string> extract_params_in_expr(Halide::Expr expr);
std::vector<std::string> extract_vars_or_rvars_in_expr(Halide::Expr expr);
// @}

/** @name Extract the list of dependencies of a Function
 *  (defined in modifiers.cpp) */
// {@
std::map<std::string, Halide::Func>   extract_func_calls  (Halide::Func func);
std::map<std::string, Halide::Buffer> extract_buffer_calls(Halide::Func func);
// @}

#endif // _MODIFIERS_H_
