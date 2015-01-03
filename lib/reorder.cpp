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

void RecFilter::inline_func(string func_name) {
    if (contents.ptr->name == func_name) {
        return;
    }
    Function F = internal_function(func_name).func;
    inline_function(F, funcs());
    contents.ptr->func.erase(func_name);
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
        RecFilter rf(func().name() + "_" + int_to_string(i));

        // set the image border conditions
        if (contents.ptr->clamped_border) {
            rf.set_clamped_image_border();
        }

        // same pure def as original filter for the first
        // subsequent filters call the result of prev recfilter
        if (i == 0) {
            rf(args) = func().values();
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
            rf(args) = pure_values;
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

            vector<double> coeff;
            coeff.push_back(contents.ptr->feedfwd_coeff(scan_id));
            for (int u=0; u<order; u++) {
                coeff.push_back(contents.ptr->feedback_coeff(scan_id,u));
            }

            rf.add_filter((causal ? +x : -x), coeff);
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
