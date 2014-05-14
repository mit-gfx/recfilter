#include "split.h"

using namespace Halide;
using namespace Halide::Internal;

using std::string;
using std::cerr;
using std::endl;
using std::vector;
using std::map;


// -----------------------------------------------------------------------------

static void merge(Function A, Function B, string merged_name) {
    // basic checks
    {
        bool can_merge = true;

        bool brace_present = (
                (merged_name.find(DELIM_START) != string::npos) ||
                (merged_name.find(DELIM_START) != string::npos));
        if (brace_present) {
            cerr << "Bad Function name " << merged_name << ". ";
            cerr << DELIM_START << " and " << DELIM_END << " ";
            cerr << "are not allowed in Function names as they are used ";
            cerr << "by splitting routines and carry special meaning";
            assert(false);
        }

        assert(A.name()!=merged_name && B.name()!=merged_name &&
                "Name of merged function cannot be same as either of the merged functions");

        // pure definitions must have same args
        can_merge &= (A.args().size() == B.args().size());
        for (size_t i=0; can_merge && i<A.args().size(); i++) {
            can_merge &= (A.args()[i] == B.args()[i]);
        }
        assert(can_merge && "Functions to be merged must have same args in pure defs");

        // each reduction definition must have same args and reduction domain
        vector<ReductionDefinition> Ar = A.reductions();
        vector<ReductionDefinition> Br = B.reductions();
        for (size_t i=0; can_merge && i<std::min(Ar.size(),Br.size()); i++) {
            can_merge &= Ar[i].domain.same_as(Br[i].domain);
            can_merge &= (Ar[i].args.size() == Br[i].args.size());
            for (size_t j=0; can_merge && j<Ar[i].args.size(); j++) {
                can_merge &= same_expr(Ar[i].args[j], Br[i].args[j]);
            }
        }
        assert(can_merge && "Functions to be merged must have same args in reduction defs");

        // A must not depend upon B
        for (size_t i=0; can_merge && i<A.values().size(); i++) {
            can_merge &= (!expr_depends_on_func(A.values()[i], B.name()));
        }
        for (size_t i=0; can_merge && i<A.reductions().size(); i++) {
            for (size_t j=0; can_merge && j<A.reductions()[i].values.size(); j++) {
                can_merge &= (!expr_depends_on_func(A.reductions()[i].values[j], B.name()));
            }
        }
        assert(can_merge && "Functions to be merged must not call each other, first function calls second");

        // B must not depend upon A
        for (size_t i=0; can_merge && i<B.values().size(); i++) {
            can_merge &= (!expr_depends_on_func(B.values()[i], A.name()));
        }
        for (size_t i=0; can_merge && i<B.reductions().size(); i++) {
            for (size_t j=0; can_merge && j<B.reductions()[i].values.size(); j++) {
                can_merge &= (!expr_depends_on_func(B.reductions()[i].values[j], A.name()));
            }
        }
        assert(can_merge && "Functions to be merged must not call each other, second function calls first");
    }

    size_t num_outputs_A = A.outputs();
    size_t num_outputs_B = B.outputs();

    // merge procedure
    {
        if (merged_name.empty()) {
            merged_name = "Merged{" + A.name() + "}{" + B.name() + "}";
        }

        Function AB(merged_name);

        // merged tuple for pure definition
        {
            vector<Expr> values;

            // RHS of pure definitions of A
            for (size_t j=0; j<num_outputs_A; j++) {
                values.push_back(A.values()[j]);
            }

            // RHS of pure definitions of B
            for (size_t j=0; j<num_outputs_B; j++) {
                values.push_back(B.values()[j]);
            }

            AB.define(A.args(), values);
        }

        // merged tuple for each of reduction definitions
        {
            vector<ReductionDefinition> Ar = A.reductions();
            vector<ReductionDefinition> Br = B.reductions();
            for (size_t i=0; i<std::max(Ar.size(),Br.size()); i++) {
                vector<Expr> values;

                // RHS of reduction definitions of A
                for (size_t j=0; j<num_outputs_A; j++) {
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
                for (size_t j=0; j<num_outputs_B; j++) {
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

        // if both A and B are identical, no need to create a Tuple
        // with RHS of both A and B, just RHS of either will suffice
        {
            bool identical = true;

            // no. of outputs and reduction definitions must be same
            identical &= (num_outputs_A == num_outputs_B);
            identical &= (A.reductions().size() == B.reductions().size());

            // compare pure defs (0, num_outputs_A-1) to
            // (num_outputs_A, num_outputs_A+num_outputs_B-1)
            for (size_t i=0; identical && i<num_outputs_A; i++) {
                Expr a = AB.values()[i];
                Expr b = AB.values()[num_outputs_A+i];
                identical &= same_expr(a,b);
                identical &= same_expr(AB.values()[i], AB.values()[num_outputs_A+i]);
            }

            // compare reduction definition outputs (0, num_outputs_A-1) to
            // (num_outputs_A, num_outputs_A+num_outputs_B-1), for all reduction defs
            for (size_t i=0; identical && i<AB.reductions().size(); i++) {
                for (size_t j=0; identical && j<num_outputs_A; j++) {
                    Expr a = AB.reductions()[i].values[j];
                    Expr b = AB.reductions()[i].values[num_outputs_A+j];
                    a = increment_value_index_in_func_call(AB.name(), num_outputs_A, a);
                    identical &= same_expr(a,b);
                }
            }

            // if A and B are identical remove all Tuple entries (num_outputs_A,end)
            if (identical) {
                vector<string> args = AB.args();
                vector<Expr> values = AB.values();
                vector<ReductionDefinition> reductions = AB.reductions();

                values.erase(values.begin()+num_outputs_A, values.end());
                AB.clear_all_definitions();
                AB.define(args,values);

                for (size_t i=0; i<reductions.size(); i++) {
                    vector<Expr> a = reductions[i].args;
                    vector<Expr> v = reductions[i].values;
                    v.erase(v.begin()+num_outputs_A, v.end());
                    AB.define_reduction(a,v);
                }

                // set number of outputs of A to 0 since A and B's outputs are same
                num_outputs_A = 0;
            }
        }

        // mutate A to index into merged function
        {
            vector<string> args = A.args();
            vector<Expr> call_args;
            vector<Expr> values;
            for (size_t i=0; i<A.args().size(); i++) {
                call_args.push_back(Var(A.args()[i]));
            }
            for (size_t i=0; i<A.values().size(); i++) {
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
            for (size_t i=0; i<B.args().size(); i++) {
                call_args.push_back(Var(B.args()[i]));
            }
            for (size_t i=0; i<B.values().size(); i++) {
                values.push_back(Call::make(AB, call_args, i+num_outputs_A));
            }
            B.clear_all_definitions();
            B.define(args, values);
        }
    }
}

// -----------------------------------------------------------------------------

void merge(Func S, string func_a, string func_b, string merged_name) {
    Function FA;
    Function FB;

    vector<Func> func_list;
    extract_func_calls(S, func_list);

    for (size_t i=0; i<func_list.size(); i++) {
        if (func_a == func_list[i].name())
            FA = func_list[i].function();
        if (func_b == func_list[i].name())
            FB = func_list[i].function();
    }

    if (!FA.has_pure_definition()) {
        cerr << func_a << " to be merged not found" << endl;
        assert(false);
    }
    if (!FB.has_pure_definition()) {
        cerr << func_b << " to be merged not found" << endl;
        assert(false);
    }

    merge(FA, FB, merged_name);
}

void merge(Func S, string func_a, string func_b, string func_c, string merged_name) {
    string func_ab = "Merged_%d" + int_to_string(rand());
    merge(S, func_a, func_b, func_ab);
    merge(S, func_c, func_ab, merged_name);
    inline_function(S, func_ab);
}

void merge(Func S, string func_a, string func_b, string func_c, string func_d, string merged_name) {
    string func_ab = "Merged_%d" + int_to_string(rand());
    string func_cd = "Merged_%d" + int_to_string(rand());
    merge(S, func_a, func_b, func_ab);
    merge(S, func_c, func_d, func_cd);
    merge(S, func_ab,func_cd,merged_name);
    inline_function(S, func_ab);
    inline_function(S, func_cd);
}

void merge(Func A, Func B, string merged_name) {
    Function FA = A.function();
    Function FB = B.function();
    merge(FA, FB, merged_name);
}

void merge_and_inline(Func S, string func_a, string func_b, string merged_name) {
    merge(S, func_a, func_b, merged_name);
    inline_function(S, func_a);
    inline_function(S, func_b);
}

void merge_and_inline(Func S, string func_a, string func_b, string func_c, string func_d, string merged_name) {
    string func_ab = "Merged_%d" + int_to_string(rand());
    string func_cd = "Merged_%d" + int_to_string(rand());
    merge_and_inline(S, func_a, func_b, func_ab);
    merge_and_inline(S, func_c, func_d, func_cd);
    merge_and_inline(S, func_ab,func_cd,merged_name);
}

void merge_and_inline(Func S, string func_a, string func_b, string func_c, string merged_name) {
    string func_ab = "Merged_%d" + int_to_string(rand());
    merge_and_inline(S, func_a, func_b, func_ab);
    merge_and_inline(S, func_c, func_ab, merged_name);
}


// -----------------------------------------------------------------------------

void inline_function(Func F, string func_name) {
    // extract all Function calls
    vector<Func> func_list;
    extract_func_calls(F, func_list);
    for (size_t i=0; i<func_list.size(); i++) {
        if (func_name == func_list[i].name()) {
            inline_function(F, func_list[i]);
            return;
        }
    }
}

void inline_function(Func F, Func A) {
    if (F.name() == A.name())
        return;

    // extract all Function calls
    vector<Func> func_list;
    extract_func_calls(F, func_list);

    // function to be inlined must be pure
    Function f = A.function();
    assert(f.is_pure() && "Function to be inlined must be pure");

    // go to all other functions and inline calls to f
    for (size_t j=0; j<func_list.size(); j++) {
        Function g = func_list[j].function();

        vector<string> args   = g.args();
        vector<Expr>   values = g.values();
        vector<ReductionDefinition> reductions = g.reductions();

        for (size_t k=0; k<values.size(); k++) {
            values[k] = inline_func_calls(f, values[k]);
        }
        g.clear_all_definitions();
        g.define(args, values);

        for (size_t k=0; k<reductions.size(); k++) {
            vector<Expr> reduction_args   = reductions[k].args;
            vector<Expr> reduction_values = reductions[k].values;
            for (size_t u=0; u<reduction_values.size(); u++) {
                reduction_values[u] = inline_func_calls(f, reduction_values[u]);
            }
            g.define_reduction(reduction_args, reduction_values);
        }
    }
}

// -----------------------------------------------------------------------------

void inline_pure_functions(Func F) {
    // extract all Function calls
    vector<Func> func_list;
    extract_func_calls(F, func_list);

    // find all Funcs which are pure
    for (size_t i=0; i<func_list.size(); i++) {
        if (!func_list[i].is_reduction()) {
            inline_function(F, func_list[i]);
        }
    }
}

void inline_non_split_functions(Func F, size_t num_splits) {
    // extract all Function calls
    vector<Func> func_list;
    extract_func_calls(F, func_list);

    // find all Funcs which are not split in all dimensions
    for (size_t i=0; i<func_list.size(); i++) {
        string name = func_list[i].name();
        size_t delims = 0;
        for (size_t j=0; j<name.length(); j++) {
            char c = name[j];
            if (c==DELIM_START || c==DELIM_END)
                delims++;
        }

        assert(delims%2==0 && "Error in split function names");

        // 2 delimiters are added initially, then each split adds 2 delimliters
        if (delims>0 && delims!=(2+2*num_splits)) {
            if (!func_list[i].is_reduction()) {             // we can only inline
                inline_function(F, func_list[i]);           // pure functions
            }
        }
    }
}

// -----------------------------------------------------------------------------

void float_dependencies_to_root(Func F) {
    // extract all Function calls
    vector<Func> func_list;
    extract_func_calls(F, func_list);

    // list of Functions which compute dependencies across tiles
    vector<Function> dependency_func_list;
    for (size_t i=0; i<func_list.size(); i++) {
        Function f = func_list[i].function();
        if (f.name().find(INTER_TILE_DEPENDENCY) != string::npos) {
            dependency_func_list.push_back(f);
            func_list.erase(func_list.begin()+i);
            i--;
        }
    }

    // process all dependency functions
    for (size_t i=0; i<dependency_func_list.size(); i++) {
        Function f_dep = dependency_func_list[i];

        // iterate until f_dep has floated to the root node
        bool function_hierarchy_changed = true;

        while (function_hierarchy_changed) {
            function_hierarchy_changed = false;

            // find a function that calls f_dep
            // f_dep calls can be extracted from f only if the call appears
            // in a pure definition, and not reduction defintions
            for (size_t i=0; i<func_list.size(); i++) {
                Function f = func_list[i].function();

                // check if f calls f_dep in pure def
                bool f_pure_calls_fdep = false;
                for (size_t j=0; j<f.values().size(); j++) {
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
                for (size_t k=0; k<f.values().size(); k++) {
                    fdep_expr[k] = remove_func_calls(f_dep.name(),false,f.values()[k]);
                }

                // f needs to be modified only if some other function that called f
                // is changed in the following for loop
                bool modify_f = false;

                // find all g that call f and append f_dep calls to their pure
                // and reduction defs where f is called
                for (size_t j=0; j<func_list.size(); j++) {
                    Function g = func_list[j].function();

                    // check if g calls f
                    bool g_calls_f = false;
                    for (size_t k=0; k<g.values().size(); k++) {
                        g_calls_f |= expr_depends_on_func(g.values()[k], f.name());
                    }
                    for (size_t k=0; k<g.reductions().size(); k++) {
                        ReductionDefinition gr = g.reductions()[k];
                        for (size_t u=0; u<gr.values.size(); u++) {
                            g_calls_f |= expr_depends_on_func(gr.values[u], f.name());
                        }
                    }

                    // nothing to change if g does not call f
                    // add call of fdep at every location where g calls f
                    if (g_calls_f) {
                        vector<string> pure_args = g.args();
                        vector<Expr>   pure_vals = g.values();
                        vector<ReductionDefinition> reductions = g.reductions();

                        for (size_t k=0; k<pure_vals.size(); k++) {
                            pure_vals[k] = augment_func_call(f.name(), f.args(),
                                    fdep_expr, pure_vals[k]);
                        }
                        g.clear_all_definitions();
                        g.define(pure_args, pure_vals);

                        for (size_t k=0; k<reductions.size(); k++) {
                            ReductionDefinition gr = reductions[k];
                            vector<Expr> args   = gr.args;
                            vector<Expr> values = gr.values;
                            for (size_t u=0; u<gr.values.size(); u++) {
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
                    for (size_t k=0; k<pure_vals.size(); k++) {
                        pure_vals[k] = remove_func_calls(f_dep.name(), true, pure_vals[k]);
                    }
                    f.clear_all_definitions();
                    f.define(pure_args, pure_vals);

                    // leave reduction defs untouched as they don't call f_dep
                    for (size_t k=0; k<reductions.size(); k++) {
                        ReductionDefinition gr = reductions[k];
                        f.define_reduction(gr.args, gr.values);
                    }
                    function_hierarchy_changed = true;
                }
            }
        }
    }
}

void swap_variables(Func S, string func_name, Var a, Var b) {
    assert(!a.same_as(b) && "Variables to be swapped must be different");

    vector<Func> func_list;
    extract_func_calls(S, func_list);

    Function F;
    for (size_t i=0; i<func_list.size(); i++) {
        if (func_name == func_list[i].name())
            F = func_list[i].function();
    }

    assert(F.has_pure_definition() && "Function to swap variables not found");

    int va_idx = -1;
    int vb_idx = -1;

    // basic checks
    {
        assert(F.is_pure() && "Variable swapping can only be applied to pure functions");

        // check that both a and b are variables of the function
        for (size_t i=0; i<F.args().size(); i++) {
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
    for (size_t i=0; i<values.size(); i++) {
        string temp_var_name = "temp_var_" + int_to_string(rand());
        Var t(temp_var_name);
        values[i] = swap_vars_in_expr(a.name(), b.name(), values[i]);
    }
    F.clear_all_definitions();
    F.define(args, values);

    // find all function calls and swap the calling args
    // at indices va_idx and vb_idx
    for (size_t i=0; i<func_list.size(); i++) {
        if (func_name != func_list[i].name()) {
            bool modified = false;
            Function f = func_list[i].function();

            // change all RHS values of the function
            vector<string> pure_args = f.args();
            vector<Expr>   pure_values = f.values();
            vector<ReductionDefinition> reductions = f.reductions();
            for (size_t i=0; i<pure_values.size(); i++) {
                if (expr_depends_on_func(pure_values[i], F.name())) {
                    pure_values[i] = swap_callargs_in_func_call(
                            F.name(), va_idx, vb_idx, pure_values[i]);
                    modified = true;
                }
            }
            for (size_t k=0; k<reductions.size(); k++) {
                for (size_t u=0; u<reductions[k].values.size(); u++) {
                    if (expr_depends_on_func(reductions[k].values[u], F.name())) {
                        reductions[k].values[u] = swap_callargs_in_func_call(
                                F.name(), va_idx, vb_idx, reductions[k].values[u]);
                        modified = true;
                    }
                }
            }
            if (modified) {
                f.clear_all_definitions();
                f.define(pure_args, pure_values);
                for (size_t k=0; k<reductions.size(); k++) {
                    vector<Expr> reduction_args   = reductions[k].args;
                    vector<Expr> reduction_values = reductions[k].values;
                    f.define_reduction(reduction_args, reduction_values);
                }
            }
        }
    }
}


void expand_multiple_reductions(Func S) {
    vector<Func> func_list;
    extract_func_calls(S, func_list);

    for (size_t u=0; u<func_list.size(); u++) {
        Function F = func_list[u].function();

        // only process functions with multiple reduction defs
        if (F.reductions().size() < 2)
            continue;

        int num_reductions = F.reductions().size();

        vector<Function> Fsub;

        for (size_t k=0; k<num_reductions; k++) {
            Function function(F.name() + DELIM_START+int_to_string(k)+DELIM_END);

            // pure args same as pure args of original function
            // pure val is call to function corresponding to previous
            // reduction def
            vector<string> pure_args;
            vector<Expr>   call_args;
            vector<Expr>   pure_values;
            for (size_t i=0; i<F.args().size(); i++) {
                pure_args.push_back(F.args()[i]);
                call_args.push_back(Var(F.args()[i]));
            }
            for (size_t i=0; i<F.values().size(); i++) {
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
            for (size_t i=0; i<reduction.values.size(); i++) {
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
