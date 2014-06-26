#include "split.h"
#include "split_macros.h"

using namespace Halide;
using namespace Halide::Internal;

using std::string;
using std::cerr;
using std::endl;
using std::vector;
using std::map;

// -----------------------------------------------------------------------------

static vector<Function> extract_called_functions(Function function) {
    vector<Function> func_list;
    map<string, Function> func_map = find_transitive_calls(function);
    map<string, Function>::iterator f_it  = func_map.begin();
    map<string, Function>::iterator f_end = func_map.end();
    while (f_it != f_end) {
        func_list.push_back(f_it->second);
        f_it++;
    }
}

static vector<Function> extract_called_functions(Func f) {
    return extract_called_functions(f.function());
}

// -----------------------------------------------------------------------------

static bool merge_feasible(Function A, Function B, string& error) {
    bool can_merge = true;

    // pure definitions must have same args
    can_merge &= (A.args().size() == B.args().size());
    for (int i=0; can_merge && i<A.args().size(); i++) {
        can_merge &= (A.args()[i] == B.args()[i]);
    }
    if (!can_merge) {
        error = "Functions to be merged must have same args in pure defs";
    }

    // each reduction definition must have same args and reduction domain
    vector<ReductionDefinition> Ar = A.reductions();
    vector<ReductionDefinition> Br = B.reductions();
    for (int i=0; can_merge && i<std::min(Ar.size(),Br.size()); i++) {
        can_merge &= Ar[i].domain.same_as(Br[i].domain);
        can_merge &= (Ar[i].args.size() == Br[i].args.size());
        for (int j=0; can_merge && j<Ar[i].args.size(); j++) {
            can_merge &= equal(Ar[i].args[j], Br[i].args[j]);
        }
    }
    if (!can_merge) {
        error = "Functions to be merged must have same args in reduction defs";
    }

    // A must not depend upon B
    for (int i=0; can_merge && i<A.values().size(); i++) {
        can_merge &= (!expr_depends_on_func(A.values()[i], B.name()));
    }
    for (int i=0; can_merge && i<A.reductions().size(); i++) {
        for (int j=0; can_merge && j<A.reductions()[i].values.size(); j++) {
            can_merge &= (!expr_depends_on_func(A.reductions()[i].values[j], B.name()));
        }
    }
    if (!can_merge) {
        error = "Functions to be merged must not call each other";
    }

    // B must not depend upon A
    for (int i=0; can_merge && i<B.values().size(); i++) {
        can_merge &= (!expr_depends_on_func(B.values()[i], A.name()));
    }
    for (int i=0; can_merge && i<B.reductions().size(); i++) {
        for (int j=0; can_merge && j<B.reductions()[i].values.size(); j++) {
            can_merge &= (!expr_depends_on_func(B.reductions()[i].values[j], A.name()));
        }
    }
    if (!can_merge) {
        error = "Functions to be merged must not call each other";
    }

    return can_merge;
}

static bool check_duplicate(Function A, Function B) {
    // duplicate functions can always be merged
    string err;
    bool duplicate = merge_feasible(A,B, err);

    // no. of outputs and reduction definitions must be same
    duplicate &= (A.outputs() == B.outputs());
    duplicate &= (A.reductions().size() == B.reductions().size());

    // compare pure defs
    for (int i=0; duplicate && i<A.outputs(); i++) {
        duplicate &= equal(A.values()[i], B.values()[i]);
    }

    // compare reduction definition outputs
    for (int i=0; duplicate && i<A.reductions().size(); i++) {
        for (int j=0; duplicate && j<A.reductions()[i].values.size(); j++) {
            Expr a = A.reductions()[i].values[j];
            Expr b = B.reductions()[i].values[j];
            duplicate &= equal(a, substitute_func_call(B.name(),A,b));
        }
    }
    return duplicate;
}

static void merge(Func S, Function A, Function B, string merged_name) {
    string merge_error;
    if (!merge_feasible(A,B,merge_error)) {
        cerr << merge_error << std::endl;
        assert(false);
    }

    int num_outputs_A = A.outputs();
    int num_outputs_B = B.outputs();

    bool duplicate = check_duplicate(A,B);

    // merge procedure
    Function AB(merged_name);

    // merged tuple for pure definition
    {
        vector<Expr> values;

        // RHS of pure definitions of A
        for (int j=0; j<num_outputs_A; j++) {
            values.push_back(A.values()[j]);
        }

        // RHS of pure definitions of B
        // if A and B are identical, no need to create a Tuple
        // with RHS of both A and B, just RHS of A will suffice
        for (int j=0; !duplicate && j<num_outputs_B; j++) {
            values.push_back(B.values()[j]);
        }

        AB.define(A.args(), values);
    }

    // merged tuple for each of reduction definitions
    {
        vector<ReductionDefinition> Ar = A.reductions();
        vector<ReductionDefinition> Br = B.reductions();
        for (int i=0; i<std::max(Ar.size(),Br.size()); i++) {
            vector<Expr> values;

            // RHS of reduction definitions of A
            for (int j=0; j<num_outputs_A; j++) {
                if (i < Ar.size()) {
                    // replace all recursive calls to A with calls to AB
                    Expr val = substitute_func_call(A.name(), AB, Ar[i].values[j]);
                    values.push_back(val);
                } else {
                    // add redundant reduction if A has no more reduction definitions
                    values.push_back(Call::make(AB, Ar[i].args, j));
                }
            }

            // RHS of reduction definitions of B
            // if A and B are identical, no need to create a Tuple
            // with RHS of both A and B, just RHS of A will suffice
            for (int j=0; !duplicate && j<num_outputs_B; j++) {
                if (i < Br.size()) {
                    // replace all recursive calls to B with calls to AB and
                    // increment value indices by number of outputs of A
                    Expr val = substitute_func_call(B.name(), AB, Br[i].values[j]);
                    val = increment_value_index_in_func_call(AB.name(), num_outputs_A, val);
                    values.push_back(val);
                } else {
                    // add redundant reduction if B has no more reduction definitions
                    values.push_back(Call::make(AB, Br[i].args, j+num_outputs_A));
                }
            }

            AB.define_reduction(Ar[i].args, values);
        }
    }

    // mutate A to index into merged function
    {
        vector<string> args = A.args();
        vector<Expr> call_args;
        vector<Expr> values;
        for (int i=0; i<A.args().size(); i++) {
            call_args.push_back(Var(A.args()[i]));
        }
        for (int i=0; i<A.values().size(); i++) {
            values.push_back(Call::make(AB, call_args, i));
        }
        A.clear_all_definitions();
        A.define(args, values);
    }

    // mutate B to index into merged function
    {
        vector<string> args = B.args();
        vector<Expr> call_args;
        vector<Expr> values;
        for (int i=0; i<B.args().size(); i++) {
            call_args.push_back(Var(B.args()[i]));
        }
        for (int i=0; i<B.values().size(); i++) {
            if (duplicate) {
                values.push_back(Call::make(AB, call_args, i));
            } else {
                values.push_back(Call::make(AB, call_args, i+num_outputs_A));
            }
        }
        B.clear_all_definitions();
        B.define(args, values);
    }

    // inline A and B
    inline_function(S, Func(A));
    inline_function(S, Func(B));
}

// -----------------------------------------------------------------------------

void merge(Func S, string func_a, string func_b, string merged_name) {
    Function FA;
    Function FB;

    vector<Function> func_list = extract_called_functions(S);

    for (int i=0; i<func_list.size(); i++) {
        if (func_a == func_list[i].name())
            FA = func_list[i];
        if (func_b == func_list[i].name())
            FB = func_list[i];
    }

    if (!FA.has_pure_definition()) {
        cerr << func_a << " to be merged not found" << endl;
        assert(false);
    }
    if (!FB.has_pure_definition()) {
        cerr << func_b << " to be merged not found" << endl;
        assert(false);
    }

    merge(S, FA, FB, merged_name);
}

void merge(Func S, string func_a, string func_b, string func_c, string merged_name) {
    string func_ab = "Merged_%d" + int_to_string(rand());
    merge(S, func_a, func_b, func_ab);
    merge(S, func_c, func_ab, merged_name);
}

void merge(Func S, string func_a, string func_b, string func_c, string func_d, string merged_name) {
    string func_ab = "Merged_%d" + int_to_string(rand());
    string func_cd = "Merged_%d" + int_to_string(rand());
    merge(S, func_a, func_b, func_ab);
    merge(S, func_c, func_d, func_cd);
    merge(S, func_ab,func_cd,merged_name);
}

void merge(Func S, std::vector<std::string> funcs, string merged) {
    assert(funcs.size() > 1);
    string func_prev_merge = funcs[0];
    string func_next_merge;
    for (int i=1; i<funcs.size(); i++) {
        if (i == funcs.size()-1) {
            func_next_merge = merged;
        } else {
            func_next_merge = "Merged_%d" + int_to_string(rand());
        }
        merge(S, funcs[i], func_prev_merge, func_next_merge);
        func_prev_merge = func_next_merge;
    }
}

void merge_duplicates_with_substring(Func S, string pattern) {
    vector<Function> func_list = extract_called_functions(S);

    for (int i=0; i<func_list.size(); i++) {
        bool rebuild_func_list = false;
        Function A = func_list[i];

        if (A.name().find(pattern) == string::npos)
            continue;

        for (int j=0; !rebuild_func_list && j<func_list.size(); j++) {
            Function B = func_list[j];

            if (A.name() == B.name())
                continue;

            //cerr << "Testing for duplicates " << A.name() << " " << B.name() << endl;
            if (check_duplicate(A, B)) {
                string merged_name;
                if (A.name().length() < B.name().length())
                    merged_name = A.name();
                else
                    merged_name = B.name();

                merge(S, A, B, merged_name);
                rebuild_func_list = true;
            }
        }

        if (rebuild_func_list) {
            i = 0;
            func_list.clear();
            func_list = extract_called_functions(S);
        }
    }
}

// -----------------------------------------------------------------------------

void inline_function(Function F, Function A) {
    inline_function(Func(F), Func(A));
}

void inline_function(Func F, string func_name) {
    // extract all Function calls
    bool found = false;
    vector<Function> func_list = extract_called_functions(F);
    for (int i=0; !found && i<func_list.size(); i++) {
        if (func_name == func_list[i].name()) {
            inline_function(F, func_list[i]);
            found = true;
        }
    }
    if (!found) {
        cerr << "Function " << func_name << " to be inlined not found" << endl;
        assert(false);
    }
}

void inline_function(Func F, Func A) {
    if (F.name() == A.name())
        return;

    vector<Function> func_list = extract_called_functions(F);

    // function to be inlined must be pure
    Function f = A.function();
    assert(f.is_pure() && "Function to be inlined must be pure");

    // go to all other functions and inline calls to f
    for (int j=0; j<func_list.size(); j++) {
        Function g = func_list[j];

        vector<string> args   = g.args();
        vector<Expr>   values = g.values();
        vector<ReductionDefinition> reductions = g.reductions();

        for (int k=0; k<values.size(); k++) {
            values[k] = inline_func_calls(f, values[k]);
        }
        g.clear_all_definitions();
        g.define(args, values);

        for (int k=0; k<reductions.size(); k++) {
            vector<Expr> reduction_args   = reductions[k].args;
            vector<Expr> reduction_values = reductions[k].values;
            for (int u=0; u<reduction_args.size(); u++) {
                reduction_args[u] = inline_func_calls(f, reduction_args[u]);
            }
            for (int u=0; u<reduction_values.size(); u++) {
                reduction_values[u] = inline_func_calls(f, reduction_values[u]);
            }
            g.define_reduction(reduction_args, reduction_values);
        }
    }
}

// -----------------------------------------------------------------------------

void inline_functions_with_substring(Func F, string pattern) {
    vector<Function> func_list = extract_called_functions(F);

    // find all Funcs containing pattern in their name
    for (int i=0; i<func_list.size(); i++) {
        if (func_list[i].is_pure() &&
                func_list[i].name().find(pattern) != string::npos)
        {
            inline_function(F, func_list[i]);
        }
    }
}

// -----------------------------------------------------------------------------

void float_dependencies_to_root(Func F) {
    vector<Function> func_list = extract_called_functions(F);

    // list of Functions which compute dependencies across tiles
    vector<Function> dependency_func_list;
    for (int i=0; i<func_list.size(); i++) {
        Function f = func_list[i];
        if (f.name().find(COMPLETE_TAIL_RESIDUAL)!=string::npos ||
            f.name().find(FINAL_RESULT_RESIDUAL )!=string::npos)
        {
            dependency_func_list.push_back(f);
            func_list.erase(func_list.begin()+i);
            i--;
        }
    }

    // process all dependency functions
    for (int i=0; i<dependency_func_list.size(); i++) {
        Function f_dep = dependency_func_list[i];

        // iterate until f_dep has floated to the root node
        bool function_hierarchy_changed = true;

        while (function_hierarchy_changed) {
            function_hierarchy_changed = false;

            // find a function that calls f_dep
            // f_dep calls can be extracted from f only if the call appears
            // in a pure definition, and not reduction defintions
            for (int i=0; i<func_list.size(); i++) {
                Function f = func_list[i];

                // check if f calls f_dep in pure def
                bool f_pure_calls_fdep = false;
                for (int j=0; j<f.values().size(); j++) {
                    f_pure_calls_fdep |= expr_depends_on_func(f.values()[j], f_dep.name());
                }

                // f is not pure, then f_dep cannot be pulled out of f
                // f does not call f_dep, no mutation required
                if (!f.is_pure() || !f_pure_calls_fdep)
                    continue;

                // extract all calls to f_dep from f, dont modify f already
                // because f might be the root node. if there is a function
                // g which calls f then f is not root and it is ok to modify it
                vector<Expr> fdep_expr(f.values().size());
                for (int k=0; k<f.values().size(); k++) {
                    fdep_expr[k] = remove_func_calls(f_dep.name(),false,f.values()[k]);
                }

                // f needs to be modified only if some other function that called f
                // is changed in the following for loop
                bool modify_f = false;

                // find all g that call f and append f_dep calls to their pure
                // and reduction defs where f is called
                for (int j=0; j<func_list.size(); j++) {
                    Function g = func_list[j];

                    // check if g calls f
                    bool g_calls_f = false;
                    for (int k=0; k<g.values().size(); k++) {
                        g_calls_f |= expr_depends_on_func(g.values()[k], f.name());
                    }
                    for (int k=0; k<g.reductions().size(); k++) {
                        ReductionDefinition gr = g.reductions()[k];
                        for (int u=0; u<gr.values.size(); u++) {
                            g_calls_f |= expr_depends_on_func(gr.values[u], f.name());
                        }
                    }

                    // nothing to change if g does not call f
                    // add call of fdep at every location where g calls f
                    if (g_calls_f) {
                        vector<string> pure_args = g.args();
                        vector<Expr>   pure_vals = g.values();
                        vector<ReductionDefinition> reductions = g.reductions();

                        for (int k=0; k<pure_vals.size(); k++) {
                            pure_vals[k] = augment_func_call(f.name(), f.args(),
                                    fdep_expr, pure_vals[k]);
                        }
                        g.clear_all_definitions();
                        g.define(pure_args, pure_vals);

                        for (int k=0; k<reductions.size(); k++) {
                            ReductionDefinition gr = reductions[k];
                            vector<Expr> args   = gr.args;
                            vector<Expr> values = gr.values;
                            for (int u=0; u<gr.values.size(); u++) {
                                gr.values[k] = augment_func_call(f.name(), f.args(),
                                        fdep_expr, gr.values[k]);
                            }
                            g.define_reduction(args, values);
                        }

                        // since calls to f have been augmented with f_dep
                        // it is now necessary to remove f_dep from def of f
                        modify_f = true;
                    }
                }

                // remove all f_dep from f because some g which called f has
                // been modified
                if (modify_f) {
                    vector<string> pure_args = f.args();
                    vector<Expr>   pure_vals = f.values();
                    vector<ReductionDefinition> reductions = f.reductions();
                    for (int k=0; k<pure_vals.size(); k++) {
                        pure_vals[k] = remove_func_calls(f_dep.name(), true, pure_vals[k]);
                    }
                    f.clear_all_definitions();
                    f.define(pure_args, pure_vals);

                    // leave reduction defs untouched as they don't call f_dep
                    for (int k=0; k<reductions.size(); k++) {
                        ReductionDefinition gr = reductions[k];
                        f.define_reduction(gr.args, gr.values);
                    }
                    function_hierarchy_changed = true;
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------

void swap_variables(Func S, string func_name, Var a, Var b) {
    assert(!a.same_as(b) && "Variables to be swapped must be different");

    vector<Function> func_list = extract_called_functions(S);

    Function F;
    for (int i=0; i<func_list.size(); i++) {
        if (func_name == func_list[i].name())
            F = func_list[i];
    }

    assert(F.has_pure_definition() && "Function to swap variables not found");

    int va_idx = -1;
    int vb_idx = -1;

    // basic checks
    {
        assert(F.is_pure() && "Variable swapping can only be applied to pure functions");

        // check that both a and b are variables of the function
        for (int i=0; i<F.args().size(); i++) {
            if (F.args()[i] == a.name()) {
                va_idx = i;
            }
            if (F.args()[i] == b.name()) {
                vb_idx = i;
            }
        }
        assert(va_idx>=0 && vb_idx>=0 && "Both variables must be args of function");
    }

    // change all RHS values of the function
    vector<string> args = F.args();
    vector<Expr>   values = F.values();
    for (int i=0; i<values.size(); i++) {
        string temp_var_name = "temp_var_" + int_to_string(rand());
        Var t(temp_var_name);
        values[i] = swap_vars_in_expr(a.name(), b.name(), values[i]);
    }
    F.clear_all_definitions();
    F.define(args, values);

    // find all function calls and swap the calling args
    // at indices va_idx and vb_idx
    for (int i=0; i<func_list.size(); i++) {
        if (func_name != func_list[i].name()) {
            bool modified = false;
            Function f = func_list[i];

            // change all RHS values of the function
            vector<string> pure_args = f.args();
            vector<Expr>   pure_values = f.values();
            vector<ReductionDefinition> reductions = f.reductions();
            for (int i=0; i<pure_values.size(); i++) {
                if (expr_depends_on_func(pure_values[i], F.name())) {
                    pure_values[i] = swap_args_in_func_call(
                            F.name(), va_idx, vb_idx, pure_values[i]);
                    modified = true;
                }
            }
            for (int k=0; k<reductions.size(); k++) {
                for (int u=0; u<reductions[k].values.size(); u++) {
                    if (expr_depends_on_func(reductions[k].values[u], F.name())) {
                        reductions[k].values[u] = swap_args_in_func_call(
                                F.name(), va_idx, vb_idx, reductions[k].values[u]);
                        modified = true;
                    }
                }
            }
            if (modified) {
                f.clear_all_definitions();
                f.define(pure_args, pure_values);
                for (int k=0; k<reductions.size(); k++) {
                    vector<Expr> reduction_args   = reductions[k].args;
                    vector<Expr> reduction_values = reductions[k].values;
                    f.define_reduction(reduction_args, reduction_values);
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------

void expand_multiple_reductions(Func S) {
    vector<Function> func_list = extract_called_functions(S);

    for (int u=0; u<func_list.size(); u++) {
        Function F = func_list[u];

        // only process functions with multiple reduction defs
        if (F.reductions().size() < 2)
            continue;

        int num_reductions = F.reductions().size();

        vector<Function> Fsub;

        for (int k=0; k<num_reductions; k++) {
            Function function(F.name() + DELIMITER + int_to_string(k));

            // pure args same as pure args of original function
            // pure val is call to function corresponding to previous
            // reduction def
            vector<string> pure_args;
            vector<Expr>   call_args;
            vector<Expr>   pure_values;
            for (int i=0; i<F.args().size(); i++) {
                pure_args.push_back(F.args()[i]);
                call_args.push_back(Var(F.args()[i]));
            }
            for (int i=0; i<F.values().size(); i++) {
                if (k==0) {
                    pure_values.push_back(F.values()[i]);
                } else {
                    pure_values.push_back(Call::make(Fsub[k-1], call_args, i));
                }
            }
            function.define(pure_args, pure_values);

            // extract the reduction from the original function
            // and replace all calls to original function by calls to
            // current function
            ReductionDefinition reduction = F.reductions()[k];
            for (int i=0; i<reduction.values.size(); i++) {
                if (k != num_reductions-1) {
                    reduction.values[i] = substitute_func_call(F.name(),
                            function, reduction.values[i]);
                }
            }
            function.define_reduction(reduction.args, reduction.values);

            Fsub.push_back(function);

            if (k == num_reductions-1) {
                F.clear_all_definitions();
                F.define(Fsub[k].args(), Fsub[k].values());
                F.define_reduction(Fsub[k].reductions()[0].args, Fsub[k].reductions()[0].values);
            }
        }
    }
}
