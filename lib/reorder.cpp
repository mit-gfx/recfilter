#include "recfilter.h"
#include "recfilter_internals.h"
#include "modifiers.h"

using namespace Halide;
using namespace Halide::Internal;

using std::string;
using std::cerr;
using std::endl;
using std::vector;
using std::map;
using std::pair;
using std::make_pair;

// -----------------------------------------------------------------------------

/** Inline a pure function in a list of other functions
 * \param f function to be inlined
 * \param func_list list of functions in which calls to first parameter must be inlined
 */
static void inline_function(Function f, vector<Func> func_list) {
    if (!f.is_pure()) {
        cerr << "Function " << f.name() << " to be inlined must be pure" << endl;
        assert(false);
    }

    // go to all other functions and inline calls to f
    for (int j=0; j<func_list.size(); j++) {
        Function g = func_list[j].function();

        // check if g not same as f and g calls f
        map<string,Function> called_funcs = find_direct_calls(g);
        if (g.name()==f.name() || called_funcs.find(f.name())==called_funcs.end()) {
            continue;
        }

        vector<string> args   = g.args();
        vector<Expr>   values = g.values();
        vector<UpdateDefinition> updates = g.updates();

        for (int k=0; k<values.size(); k++) {
            values[k] = inline_function(values[k], f);
        }
        g.clear_all_definitions();
        g.define(args, values);

        for (int k=0; k<updates.size(); k++) {
            vector<Expr> update_args   = updates[k].args;
            vector<Expr> update_values = updates[k].values;
            for (int u=0; u<update_args.size(); u++) {
                update_args[u] = inline_function(update_args[u], f);
            }
            for (int u=0; u<update_values.size(); u++) {
                update_values[u] = inline_function(update_values[u], f);
            }
            g.define_update(update_args, update_values);
        }
    }
}

/** Transpose the dimensions of a function to reshape the output buffer */
static void transpose_function_dimensions(
        RecFilterFunc& rF,      ///> recursive filter function whose dimensions must be transposed
        string a,               ///< name of first dimension to be transposed
        string b,               ///< name of second dimension to be transposed
        vector<Func> func_list  ///< list of functions in which calls to first parameter must be changed
        )
{
    if (a == b) {
        cerr << "Variables to be swapped must be different" << endl;
        assert(false);
    }

    Function F = rF.func;

    int va_idx = -1;
    int vb_idx = -1;

    // basic checks
    {
        assert(F.is_pure() && "Variable swapping can only be applied to pure functions");

        // check that both a and b are variables of the function
        for (int i=0; i<F.args().size(); i++) {
            if (F.args()[i] == a) {
                va_idx = i;
            }
            if (F.args()[i] == b) {
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
        values[i] = swap_vars_in_expr(a, b, values[i]);
    }
    F.clear_all_definitions();
    F.define(args, values);

    // find all function calls and swap the calling args
    // at indices va_idx and vb_idx
    for (int i=0; i<func_list.size(); i++) {
        Function f = func_list[i].function();

        if (f.name() == F.name()) {
            continue;
        }

        bool modified = false;

        // change all RHS values of the function
        vector<string> pure_args = f.args();
        vector<Expr>   pure_values = f.values();
        vector<UpdateDefinition> updates = f.updates();
        for (int i=0; i<pure_values.size(); i++) {
            if (expr_depends_on_func(pure_values[i], F.name())) {
                pure_values[i] = swap_args_in_func_call(
                        F.name(), va_idx, vb_idx, pure_values[i]);
                modified = true;
            }
        }
        for (int k=0; k<updates.size(); k++) {
            for (int u=0; u<updates[k].values.size(); u++) {
                if (expr_depends_on_func(updates[k].values[u], F.name())) {
                    updates[k].values[u] = swap_args_in_func_call(
                            F.name(), va_idx, vb_idx, updates[k].values[u]);
                    modified = true;
                }
            }
        }
        if (modified) {
            f.clear_all_definitions();
            f.define(pure_args, pure_values);
            for (int k=0; k<updates.size(); k++) {
                vector<Expr> update_args   = updates[k].args;
                vector<Expr> update_values = updates[k].values;
                f.define_update(update_args, update_values);
            }
        }
    }

    // swap the scheduling tags of the dimensions
    VarTag temp = rF.pure_var_category[a];
    rF.pure_var_category[a] = rF.pure_var_category[b];
    rF.pure_var_category[b] = temp;
}

/** Check if two functions are identical */
static bool check_duplicate(Function A, Function B) {
    bool duplicate = true;

    string err;

    // no. of outputs and update definitions must be same
    duplicate &= (A.outputs() == B.outputs());
    duplicate &= (A.updates().size() == B.updates().size());

    // compare pure defs
    for (int i=0; duplicate && i<A.outputs(); i++) {
        duplicate &= equal(A.values()[i], B.values()[i]);
    }

    // compare update definition outputs
    for (int i=0; duplicate && i<A.updates().size(); i++) {
        for (int j=0; duplicate && j<A.updates()[i].values.size(); j++) {
            Expr a = A.updates()[i].values[j];
            Expr b = B.updates()[i].values[j];
            duplicate &= equal(a, substitute_func_call(B.name(),A,b));
        }
    }
    return duplicate;
}


/** Check if the output buffers of two functions can be merged or interleaved
 * \param A first function to be merged/interleaved
 * \param B second function to be merged/interleaved
 * \param error error message if functions cannot be merged
 */
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

    // no update defs
    if (A.has_update_definition() || B.has_update_definition()) {
        error = "Function to be merged must not have update definitions";
        can_merge = false;
    }

    // A must not depend upon B and B must not depend upon A
    for (int i=0; can_merge && i<A.values().size(); i++) {
        can_merge &= (!expr_depends_on_func(A.values()[i], B.name()));
        if (!can_merge) {
            error = "Functions to be merged must not call each other";
        }
    }
    for (int i=0; can_merge && i<B.values().size(); i++) {
        can_merge &= (!expr_depends_on_func(B.values()[i], A.name()));
        if (!can_merge) {
            error = "Functions to be merged must not call each other";
        }
    }

    return can_merge;
}


/** Check the scheduling tags of two functions before merging or interleaving their output buffers;
 * and transpose the dimensions of one function which will make the output buffers identical
 * \returns true if scheduling tags are same or if dimensions can be transposed to the same effect
 */
static bool check_scheduling_tags(
        RecFilterFunc& fA,      ///< first function to be merged/interleaved
        RecFilterFunc& fB,      ///< second function to be merged/interleaved
        vector<Func> func_list  ///< list of functions where fA and fB must be replaced
        )
{
    Function A = fA.func;
    Function B = fB.func;

    bool same_func_tags = true;
    bool same_pure_def_tags = true;

    vector<string> diff_vars;

    map<string,VarTag>::iterator it;

    // A and B must have the same scheduling tags for the function itself
    same_func_tags &= (fA.func_category == fB.func_category);

    // check pure def scheduling tags and make a list of different tags
    same_pure_def_tags &= (fA.pure_var_category.size()==fB.pure_var_category.size());
    for (it=fB.pure_var_category.begin(); it!=fB.pure_var_category.end(); it++) {
        if (it->second != fA.pure_var_category[it->first]) {
            diff_vars.push_back(it->first);
            same_pure_def_tags &= false;
        }
    }

    if (same_func_tags && !same_pure_def_tags && diff_vars.size()%2==0) {
        // check if transposing dimensions can fix the scheduling tags
        // loop over all combinations of transposes
        do {
            vector< pair<string,string> > var_swaps;
            for (int i=0; i<diff_vars.size(); i+=2) {
                var_swaps.push_back(make_pair(diff_vars[i],diff_vars[i+1]));
            }

            bool same_pure_def_tags_after_dummy_swap = true;

            // apply these swaps on dummy scheduling tags lists
            map<string,VarTag> var_cat_A = fA.pure_var_category;
            map<string,VarTag> var_cat_B = fB.pure_var_category;
            for (int i=0; i<var_swaps.size(); i++) {
                string a = var_swaps[i].first;
                string b = var_swaps[i].second;
                VarTag temp = var_cat_B[a];
                var_cat_B[a] = var_cat_B[b];
                var_cat_B[b] = temp;
            }
            for (it=var_cat_B.begin(); it!=var_cat_B.end(); it++) {
                same_pure_def_tags_after_dummy_swap &= (it->second == var_cat_A[it->first]);
            }

            // actually apply dimension transposes because they will fix the dimensions
            if (same_pure_def_tags_after_dummy_swap) {
                for (int i=0; i<var_swaps.size(); i++) {
                    string a = var_swaps[i].first;
                    string b = var_swaps[i].second;
                    transpose_function_dimensions(fB, a, b, func_list);
                }
                same_pure_def_tags = true;
            }

        } while (!same_pure_def_tags && next_permutation(diff_vars.begin(), diff_vars.end()));
    }

    return (same_func_tags && same_pure_def_tags);
}



/** Merge two functions to create a new function which contains the outputs
 * of both the functions in a Tuple; and replace calls to the two original
 * functions by the merged function in all calls in
 */
static RecFilterFunc merge_function(
        RecFilterFunc fA,        ///< first function to be merged
        RecFilterFunc fB,        ///< second function to be merged
        string merged_name       ///< name of the merged function
        )
{
    Function A = fA.func;
    Function B = fB.func;

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

    // copy the scheduling tags of either of the two functions
    // since tags are identical
    RecFilterFunc fAB;
    fAB.func = AB;
    fAB.func_category     = fA.func_category;
    fAB.pure_var_category = fA.pure_var_category;
    fAB.caller_func       = fA.caller_func;
    fAB.callee_func       = fA.callee_func;
    return fAB;
}

/** Interleave two functions to create a new function which interleaves the outputs
 * of both the functions; and replace calls to the two original functions by new function
 */
static RecFilterFunc interleave_function(
        RecFilterFunc fA,       ///< first function to be interleaved
        RecFilterFunc fB,       ///< second function to be interleaved
        string   merged_name,   ///< name of the interleaved function
        Var      var,           ///< var to the used for interleaving
        Expr     offset         ///< interleaving offset
        )
{
    Function A = fA.func;
    Function B = fB.func;

    Function AB(merged_name);

    if (check_duplicate(A,B)) {
        // create a copy of A as the interleaved function
        AB.define(A.args(), A.values());
        for (int i=0; i<A.updates().size(); i++) {
            UpdateDefinition r = A.updates()[i];
            vector<Expr> args;
            vector<Expr> values;
            for (int j=0; j<r.args.size(); j++) {
                args.push_back(substitute_func_call(A.name(), AB, r.args[j]));
            }
            for (int j=0; j<r.values.size(); j++) {
                values.push_back(substitute_func_call(A.name(), AB, r.values[j]));
            }
            AB.define_update(args, values);
        }

        // mutate A and B to index into interleaved function
        vector<Expr> call_args;
        vector<Expr> values;
        for (int i=0; i<A.args().size(); i++) {
            call_args.push_back(Var(A.args()[i]));
        }
        for (int i=0; i<A.values().size(); i++) {
            values.push_back(Call::make(AB, call_args, i));
        }
        A.clear_all_definitions();
        B.clear_all_definitions();
        A.define(AB.args(), values);
        B.define(AB.args(), values);

    } else {

        // interleave procedure
        int var_idx = -1;
        for (int i=0; i<A.args().size(); i++) {
            if (var.name()==A.args()[i]) {
                var_idx = i;
            }
        }

        if (var_idx<0) {
            cerr << "Var provided for interleaving is not an argument of the function" <<  endl;
            assert(false);
        }

        // interleaved pure definition, A and B have no update defs
        {
            vector<Expr> values;
            for (int j=0; j<A.outputs(); j++) {
                Expr val_a = A.values()[j];
                Expr val_b = substitute(var.name(), var-offset, B.values()[j]);
                values.push_back(select(var<offset, val_a, val_b));
            }
            AB.define(A.args(), values);
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

        // mutate B to index into merged function such have var_idx is offset
        {
            vector<string> args = B.args();
            vector<Expr> call_args;
            vector<Expr> values;
            for (int i=0; i<B.args().size(); i++) {
                call_args.push_back(Var(B.args()[i]));
            }
            call_args[ var_idx ] += offset;
            for (int i=0; i<B.values().size(); i++) {
                values.push_back(Call::make(AB, call_args, i));
            }
            B.clear_all_definitions();
            B.define(args, values);
        }
    }

    // copy the scheduling tags of either of the two functions
    // since tags are identical
    RecFilterFunc fAB;
    fAB.func = AB;
    fAB.func_category     = fA.func_category;
    fAB.pure_var_category = fA.pure_var_category;
    fAB.caller_func       = fA.caller_func;
    fAB.callee_func       = fA.callee_func;
    return fAB;
}

// -----------------------------------------------------------------------------

void RecFilter::transpose_dimensions(string func_name, string a, string b) {
    RecFilterFunc& rF = internal_function(func_name);
    transpose_function_dimensions(rF, a, b, funcs());
}

void RecFilter::inline_func(string func_name) {
    if (contents.ptr->name == func_name) {
        return;
    }
    Function F = internal_function(func_name).func;
    inline_function(F, funcs());
    contents.ptr->func.erase(func_name);
}

void RecFilter::interleave_func(string func_a, string func_b, string merged_name, string var, Expr offset) {
    RecFilterFunc fA = internal_function(func_a);
    RecFilterFunc fB = internal_function(func_b);

    Function FA = fA.func;
    Function FB = fB.func;

    // check if interleaving is possible
    string error;
    if (!merge_feasible(fA.func, fB.func, error)) {
        cerr << error << endl;
        assert(false);
    }

    // check the scheduling tags, transposing dimensions if needed
    if (!check_scheduling_tags(fA, fB, funcs())) {
        cerr << "Functions to be interleaved have different dimensions as " <<
            "inferred from scheduling tags; could not be fixed by dimension transpose" << endl;
        assert(false);
    }

    RecFilterFunc F = interleave_function(fA, fB, merged_name, Var(var), offset);

    contents.ptr->func.insert(make_pair(F.func.name(), F));

    inline_func(FA.name());
    inline_func(FB.name());
}


void RecFilter::merge_func(vector<string> func_list, string merged_name) {
    if (func_list.size() > 1) {
        string func_a;
        string func_b;
        string func_m;

        for (int i=1; i<func_list.size(); i++) {
            func_a = (i==1 ? func_list[0] : func_m);
            func_b = func_list[i];
            func_m = (i==func_list.size()-1 ? merged_name : "Merged_%d" + int_to_string(rand()));

            RecFilterFunc fA = internal_function(func_a);
            RecFilterFunc fB = internal_function(func_b);
            Function FA = fA.func;
            Function FB = fB.func;

            // check if merge is feasible
            string error;
            if (!merge_feasible(fA.func, fB.func, error)) {
                cerr << error << endl;
                assert(false);
            }

            // check the scheduling tags, transposing dimensions if needed
            if (!check_scheduling_tags(fA, fB, funcs())) {
                cerr << "Functions to be merged have different dimensions as " <<
                    "inferred from scheduling tags; could not be fixed by dimension transpose" << endl;
                assert(false);
            }

            RecFilterFunc F = merge_function(fA, fB, func_m);
            inline_func(FA.name());
            inline_func(FB.name());
            contents.ptr->func.insert(make_pair(F.func.name(), F));
        }
    }
}

// -----------------------------------------------------------------------------

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
        for (int i=0; i<contents.ptr->filter_info.size(); i++) {
            int scan_dim = contents.ptr->filter_info[i].filter_dim;
            for (int j=0; j<contents.ptr->filter_info[i].num_scans; j++) {
                int scan_id = contents.ptr->filter_info[i].scan_id[j];
                bool causal = contents.ptr->filter_info[i].scan_causal[j];
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
    vector<RecFilterDim> args;
    for (int i=0; i<contents.ptr->filter_info.size(); i++) {
        args.push_back(RecFilterDim(
                    contents.ptr->filter_info[i].var.name(),
                    contents.ptr->filter_info[i].image_width));
    }

    vector<RecFilter> recfilters;

    for (int i=0; i<scans.size(); i++) {
        RecFilter rf(args, func().name() + "_" + int_to_string(i));

        // set the image border conditions
        if (!contents.ptr->border_expr.defined()) {
            rf.set_clamped_image_border();
        }

        // same pure def as original filter for the first
        // subsequent filters call the result of prev recfilter
        if (i == 0) {
            rf = func().values();
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
            rf = pure_values;
        }

        // extract the scans from the filter and
        // add them to the new filter
        for (int j=0; j<scans[i].size(); j++) {
            int scan_id = scans[i][j];

            // find the split info struct that corresponds
            // to this scan
            int dim = -1;
            int idx = -1;
            for (int u=0; dim<0 && u<contents.ptr->filter_info.size(); u++) {
                for (int v=0; idx<0 && v<contents.ptr->filter_info[u].num_scans; v++) {
                    if (scan_id == contents.ptr->filter_info[u].scan_id[v]) {
                        dim = u;
                        idx = v;
                    }
                }
            }

            if (dim<0 || idx<0) {
                cerr << "Scan " << scan_id << " not found in recursive filter " << endl;
                assert(false);
            }

            RecFilterDim x(contents.ptr->filter_info[dim].var.name(),
                    contents.ptr->filter_info[dim].image_width);
            bool causal = contents.ptr->filter_info[dim].scan_causal[idx];
            int order   = contents.ptr->filter_info[dim].filter_order;

            vector<float> coeff;
            coeff.push_back(contents.ptr->feedfwd_coeff(scan_id));
            for (int u=0; u<order; u++) {
                coeff.push_back(contents.ptr->feedback_coeff(scan_id,u));
            }

            rf.add_filter(x, coeff, causal);
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

vector<RecFilter> RecFilter::cascade(vector<int> a, vector<int> b, vector<int> c) {
    vector<vector<int> > scans;
    scans.push_back(a);
    scans.push_back(b);
    scans.push_back(c);
    return cascade(scans);
};

vector<RecFilter> RecFilter::cascade(vector<int> a, vector<int> b, vector<int> c, vector<int> d) {
    vector<vector<int> > scans;
    scans.push_back(a);
    scans.push_back(b);
    scans.push_back(c);
    scans.push_back(d);
    return cascade(scans);
};
