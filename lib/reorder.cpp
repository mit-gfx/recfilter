#include "recfilter.h"
#include "modifiers.h"

using namespace Halide;
using namespace Halide::Internal;

using std::string;
using std::cerr;
using std::endl;
using std::vector;
using std::map;

// -----------------------------------------------------------------------------

/** Infer the dimension which contains the reduction variable in a given
 * reduction definition */
static int get_scan_dimension(ReductionDefinition r) {
    vector<int> scan_dim;

    // return -1 if there is no RDom associated with the reduction
    if (r.domain.defined()) {
        vector<ReductionVariable> vars = r.domain.domain();
        for (int i=0; i<vars.size(); i++) {
            for (int j=0; j<r.args.size(); j++) {
                if (expr_depends_on_var(r.args[j], vars[i].var)) {
                    scan_dim.push_back(j);
                }
            }
        }

        // return the scan dimension if the scan is in a single
        // dimension, else return -1
        if (scan_dim.size() == 1) {
            return scan_dim[0];
        } else {
            return -1;
        }
    } else {
        return -1;
    }
}

// -----------------------------------------------------------------------------

std::vector<Func> cascade_dimensions(Func& func) {
    Function F = func.function();
    int num_dimensions = F.args().size();

    vector<Func> func_list;

    // loop over all dimensions of the function
    // and create a separate function for scans in each dimension
    for (int i=0; i<num_dimensions; i++) {
        Function f(F.name() + "_" + int_to_string(i));

        if (i==0) { // pure def of first dimension uses the original pure def
            f.define(F.args(), F.values());
        }
        else {  // pure defs of other dimension use previous dimension as pure def
            vector<Expr> call_args;
            vector<Expr> values;
            for (int j=0; j<F.args().size(); j++) {
                call_args.push_back(Var(F.args()[j]));
            }
            for (int j=0; j<F.values().size(); j++) {
                values.push_back(Call::make(func_list[i-1].function(), call_args, j));
            }
            f.define(F.args(), values);
        }

        // add the function to the list of cascaded functions
        func_list.push_back(Func(f));
    }

    // loop over all the redunction defs and add each to the function
    // corresponding to the scan dimension
    for (int j=0; j<F.reductions().size(); j++) {
        int dim = get_scan_dimension(F.reductions()[j]);
        if (dim<0) {
            cerr << "Function cascading can be used if each reduction has exactly one reduction "
                " variable in exactly one dimension" << endl;
            assert(false);
        }

        Function f = func_list[dim].function();

        // add the scan replacing all calls to original function with the new function
        vector<Expr> values = F.reductions()[j].values;
        for (int k=0; k<values.size(); k++) {
            values[k] = substitute_func_call(F.name(), f, values[k]);
        }
        f.define_reduction(F.reductions()[j].args, values);
    }

    // change the original function to index into the last function in the list
    {
        F.clear_all_definitions();
        Function f = func_list[func_list.size()-1].function();

        vector<Expr> call_args;
        vector<Expr> values;
        for (int j=0; j<f.args().size(); j++) {
            call_args.push_back(Var(f.args()[j]));
        }
        for (int j=0; j<f.values().size(); j++) {
            values.push_back(Call::make(f, call_args, j));
        }
        F.define(f.args(), values);
        func = Func(F);
    }

    return func_list;
}

std::vector<Func> cascade_repeated_scans(Func& func) {
    Function F = func.function();

    // there can be at most as many functions as reductions defs
    vector<Func> func_list;
    for (int i=0; i<F.reductions().size(); i++) {
        Function f(F.name() + "_" + int_to_string(i));

        if (i==0) { // pure def of first dimension uses the original pure def
            f.define(F.args(), F.values());
        }
        else {  // pure defs of other dimension use previous dimension as pure def
            vector<Expr> call_args;
            vector<Expr> values;
            for (int j=0; j<F.args().size(); j++) {
                call_args.push_back(Var(F.args()[j]));
            }
            for (int j=0; j<F.values().size(); j++) {
                values.push_back(Call::make(func_list[i-1].function(), call_args, j));
            }
            f.define(F.args(), values);
        }

        // add the function to the list of cascaded functions
        func_list.push_back(Func(f));
    }

    // the index of the function which should get the next reduction
    // definition in each dimension; the first reduction in each dimension
    // can go to the first function
    vector<int> func_id(F.args().size(), 0);

    // loop over all the reduction defs and add each
    // corresponding to the scan dimension
    for (int j=0; j<F.reductions().size(); j++) {
        int dim = get_scan_dimension(F.reductions()[j]);
        if (dim<0) {
            cerr << "Function cascading can be used if each reduction has exactly one reduction "
                " variable in exactly one dimension" << endl;
            assert(false);
        }

        // get the function to which this reduction def should be added
        Function f = func_list[func_id[dim]].function();

        // add the scan replacing all calls to original function with the new function
        vector<Expr> values = F.reductions()[j].values;
        for (int k=0; k<values.size(); k++) {
            values[k] = substitute_func_call(F.name(), f, values[k]);
        }
        f.define_reduction(F.reductions()[j].args, values);

        // next repeated scan in this dimension should be added to
        // the next function
        func_id[dim]++;
    }

    // remove the functions which do not have any reductions - redundant
    for (int i=0; i<func_list.size(); i++) {
        if (!func_list[i].is_reduction()) {
            func_list.erase(func_list.begin() + i);
            i--;
        }
    }

    // change the original function to index into the last function in the list
    {
        F.clear_all_definitions();
        Function f = func_list[func_list.size()-1].function();

        vector<Expr> call_args;
        vector<Expr> values;
        for (int j=0; j<f.args().size(); j++) {
            call_args.push_back(Var(f.args()[j]));
        }
        for (int j=0; j<f.values().size(); j++) {
            values.push_back(Call::make(f, call_args, j));
        }
        F.define(f.args(), values);
        func = Func(F);
    }

    return func_list;
}

// -----------------------------------------------------------------------------

/** Inline a pure function in a list of other functions
 * \param f function to be inlined
 * \param func_list list of functions in which calls to first parameter must be inlined
 */
static void inline_function(Function f, vector<Function> func_list) {
    if (!f.is_pure()) {
        cerr << "Function " << f.name() << " to be inlined must be pure" << endl;
        assert(false);
    }

    // go to all other functions and inline calls to f
    for (int j=0; j<func_list.size(); j++) {
        Function g = func_list[j];

        // check if g not same as f and g calls f
        map<string,Function> called_funcs = find_direct_calls(g);
        if (g.name()==f.name() || called_funcs.find(f.name())==called_funcs.end()) {
            continue;
        }

        vector<string> args   = g.args();
        vector<Expr>   values = g.values();
        vector<ReductionDefinition> reductions = g.reductions();

        for (int k=0; k<values.size(); k++) {
            values[k] = inline_function(values[k], f);
        }
        g.clear_all_definitions();
        g.define(args, values);

        for (int k=0; k<reductions.size(); k++) {
            vector<Expr> reduction_args   = reductions[k].args;
            vector<Expr> reduction_values = reductions[k].values;
            for (int u=0; u<reduction_args.size(); u++) {
                reduction_args[u] = inline_function(reduction_args[u], f);
            }
            for (int u=0; u<reduction_values.size(); u++) {
                reduction_values[u] = inline_function(reduction_values[u], f);
            }
            g.define_reduction(reduction_args, reduction_values);
        }
    }
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

// -----------------------------------------------------------------------------

/** Merge two functions to create a new function which contains the outputs
 * of both the functions in a Tuple; and replace calls to the two original
 * functions by the merged function in all calls in
 */
static void merge_function(
        Function A,              ///< first function to be merged
        Function B,              ///< second function to be merged
        string merged_name,      ///< name of the merged function
        vector<Function> func_list = vector<Function>() ///< list of functions where A and B must be replaced by the merged function
        )
{
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
    if (!func_list.empty()) {
        inline_function(A, func_list);
        inline_function(B, func_list);
    }
}

// -----------------------------------------------------------------------------

void RecFilter::inline_func(string func_name) {
    if (contents.ptr->recfilter.name() == func_name) {
        return;
    }

    bool found = false;
    vector<Function> func_list = contents.ptr->func_list;

    for (int i=0; !found && i<func_list.size(); i++) {
        if (func_name == func_list[i].name()) {
            inline_function(func_list[i], func_list);
            found = true;
        }
    }
    if (!found) {
        cerr << "Function " << func_name << " to be inlined not found" << endl;
        assert(false);
    }
}


void RecFilter::merge_func(string func_a, string func_b, string merged_name) {
    Function FA;
    Function FB;

    vector<Function> func_list = contents.ptr->func_list;

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

    merge_function(FA, FB, merged_name);
}

void RecFilter::merge_func(string func_a, string func_b, string func_c, string merged_name) {
    string func_ab = "Merged_%d" + int_to_string(rand());
    merge_func(func_a, func_b, func_ab);
    merge_func(func_c, func_ab, merged_name);
}

void RecFilter::merge_func(string func_a, string func_b, string func_c, string func_d, string merged_name) {
    string func_ab = "Merged_%d" + int_to_string(rand());
    string func_cd = "Merged_%d" + int_to_string(rand());
    merge_func(func_a, func_b, func_ab);
    merge_func(func_c, func_d, func_cd);
    merge_func(func_ab,func_cd,merged_name);
}

void RecFilter::merge_func(std::vector<std::string> funcs, string merged) {
    assert(funcs.size() > 1);
    string func_prev_merge = funcs[0];
    string func_next_merge;
    for (int i=1; i<funcs.size(); i++) {
        if (i == funcs.size()-1) {
            func_next_merge = merged;
        } else {
            func_next_merge = "Merged_%d" + int_to_string(rand());
        }
        merge_func(funcs[i], func_prev_merge, func_next_merge);
        func_prev_merge = func_next_merge;
    }
}

void RecFilter::swap_variables(string func_name, Var a, Var b) {
    assert(!a.same_as(b) && "Variables to be swapped must be different");

    vector<Function> func_list = contents.ptr->func_list;

    Function F;
    for (int i=0; i<func_list.size(); i++) {
        if (func_name == func_list[i].name()) {
            F = func_list[i];
        }
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

vector<RecFilter> RecFilter::cascade(vector<vector<int> > scans) {
    // check that the order does not violate
    {
        map<int, bool> scan_causal;
        map<int, bool> scan_dimension;
        map<int, int>  scan_occurance;
        vector<int> reordered_scans;
        for (int i=0; i<scans.size(); i++) {
            for (int j=0; j<scans[i].size(); j++) {
                reordered_scans.push_back(scans[i][j]);
            }
        }
        for (int i=0; i<contents.ptr->split_info.size(); i++) {
            int scan_dim = contents.ptr->split_info[i].filter_dim;
            for (int j=0; j<contents.ptr->split_info[i].num_splits; j++) {
                int scan_id = contents.ptr->split_info[i].scan_id[j];
                bool causal = contents.ptr->split_info[i].scan_causal[j];
                scan_causal[scan_id] = causal;
                scan_dimension[scan_id] = scan_dim;
            }
        }
        // order is violated only if the relative order of two scans in
        // same dimension and opposite causality changes
        for (int u=0; u<reordered_scans.size(); u++) {
            int scan_a = reordered_scans[u];
            for (int v=u+1; v<reordered_scans.size(); v++) {
                int scan_b = reordered_scans[v];
                if (scan_dimension.find(scan_a) == scan_dimension.end()) {
                    cerr << "Scan " << scan_a << " not found in recursive filter " << endl;
                    assert(false);
                }
                if (scan_dimension.find(scan_b) == scan_dimension.end()) {
                    cerr << "Scan " << scan_b << " not found in recursive filter " << endl;
                    assert(false);
                }
                int dim_a  = scan_dimension[scan_a];
                int dim_b  = scan_dimension[scan_b];
                bool causal_a = scan_causal[scan_a];
                bool causal_b = scan_causal[scan_b];
                if (dim_a==dim_b && causal_a!=causal_b && scan_b<scan_a) {
                    cerr << "Scans " << scan_a << " " << scan_b << " cannot"
                        << " be reordered" << endl;
                    assert(false);
                }
            }

            // scan_a occurs once more in the list of args
            scan_occurance[scan_a] += 1;
        }

        // check that each scan has occured exactly once
        map<int,int>::iterator so = scan_occurance.begin();
        for (; so!=scan_occurance.end(); so++) {
            if (so->second == 0) {
                cerr << "Scan " << so->first << " does not appear in the list "
                    << "of scans for cascading" << endl;
                assert(false);
            }
            if (so->second > 1) {
                cerr << "Scan " << so->first << " appears multiple times in the list "
                    << "of scans for cascading" << endl;
                assert(false);
            }
        }
    }

    // create the cascaded recursive filters
    vector<Var> args = func().args();

    vector<RecFilter> recfilters;

    for (int i=0; i<scans.size(); i++) {
        RecFilter rf(func().name() + "_" + int_to_string(i));

        // dimensions same as original filter
        rf.setArgs(args);

        // same pure def as original filter for the first
        // subsequent filters call the result of prev recfilter
        if (i == 0) {
            rf.define(func().values());
        } else {
            vector<Expr> call_args;
            vector<Expr> pure_values;
            Function f_prev = recfilters[i-1].func().function();
            for (int j=0; j<args.size(); j++) {
                call_args.push_back(args[j]);
            }
            for (int j=0; j<f_prev.outputs(); j++) {
                pure_values.push_back(Call::make(f_prev, call_args, j));
            }
            rf.define(pure_values);
        }

        // extract the scans from the filter and
        // add them to the new filter
        for (int j=0; j<scans[i].size(); j++) {
            int scan_id = scans[i][j];

            // find the split info struct that corresponds
            // to this scan
            int dim = -1;
            int idx = -1;
            for (int u=0; dim<0 && u<contents.ptr->split_info.size(); u++) {
                for (int v=0; idx<0 && v<contents.ptr->split_info[u].num_splits; v++) {
                    if (scan_id == contents.ptr->split_info[u].scan_id[v]) {
                        dim = u;
                        idx = v;
                    }
                }
            }

            if (dim<0 || idx<0) {
                cerr << "Scan " << scan_id << " not found in recursive filter " << endl;
                assert(false);
            }

            Var x           = contents.ptr->split_info[dim].var;
            RDom rx         = contents.ptr->split_info[dim].rdom;
            bool c          = contents.ptr->split_info[dim].scan_causal[idx];
            int order       = contents.ptr->split_info[dim].filter_order;
            float feedfwd   = contents.ptr->split_info[dim].feedfwd_coeff(scan_id);
            Expr border_expr= contents.ptr->split_info[dim].border_expr[idx];

            vector<float> feedback;
            for (int u=0; u<order; u++) {
                feedback.push_back(contents.ptr->split_info[dim].
                        feedback_coeff(scan_id,u));
            }

            Causality causal = (c ? CAUSAL : ANTICAUSAL);

            Border border_mode;
            if (!border_expr.defined()) {
                border_mode = CLAMP_TO_SELF;
            } else if (equal(border_expr, FLOAT_ZERO)) {
                border_mode = CLAMP_TO_ZERO;
            } else {
                border_mode = CLAMP_TO_EXPR;
            }

            rf.addScan(x, rx, feedfwd, feedback, causal, border_mode, border_expr);
        }

        recfilters.push_back(rf);
    }

    return recfilters;
}

RecFilter RecFilter::cascade(vector<int> a) {
    vector<vector<int> > scans;
    scans.push_back(a);
    vector<RecFilter> filters = cascade(scans);
    return filters[0];
}

vector<RecFilter> RecFilter::cascade(vector<int> a, vector<int> b) {
    vector<vector<int> > scans;
    scans.push_back(a);
    scans.push_back(b);
    return cascade(scans);
};
