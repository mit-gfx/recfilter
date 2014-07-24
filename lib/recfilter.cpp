#include "recfilter.h"
#include "recfilter_utils.h"

using std::string;
using std::cerr;
using std::endl;
using std::vector;
using std::queue;
using std::map;

namespace Halide {
namespace Internal {
template<>
RefCount &ref_count<RecFilterContents>(const RecFilterContents *f) {
    return f->ref_count;
}
template<>
void destroy<RecFilterContents>(const RecFilterContents *f) {
    delete f;
}
}
}

// -----------------------------------------------------------------------------

using namespace Halide;
using namespace Halide::Internal;

RecFilter::RecFilter(string n) {
    contents = new RecFilterContents;
    contents.ptr->recfilter = Func(n);
}

void RecFilter::setArgs(Var x)              { setArgs(vec(x));      }
void RecFilter::setArgs(Var x, Var y)       { setArgs(vec(x,y));    }
void RecFilter::setArgs(Var x, Var y, Var z){ setArgs(vec(x,y,z));  }

void RecFilter::setArgs(vector<Var> args) {
    for (int i=0; i<args.size(); i++) {
        SplitInfo s;
        s.clamp_border = false;

        s.var          = args[i];
        s.filter_dim   = i;

        // default values for now
        s.num_splits   = 0;
        s.filter_order = 0;
        s.image_width  = 0;
        s.tile_width   = 0;
        s.num_tiles    = 0;

        s.feedfwd_coeff = Image<float>(0);
        s.feedback_coeff = Image<float>(0,0);

        contents.ptr->split_info.push_back(s);
    }
}

void RecFilter::define(Expr pure_def) {
    vector<string> args;
    for (int i=0; i<contents.ptr->split_info.size(); i++) {
        args.push_back(contents.ptr->split_info[i].var.name());
    }
    contents.ptr->recfilter.function().define(args, vec(pure_def));
}

void RecFilter::define(Tuple pure_def) {
    vector<string> args;
    for (int i=0; i<contents.ptr->split_info.size(); i++) {
        args.push_back(contents.ptr->split_info[i].var.name());
    }
    contents.ptr->recfilter.function().define(args, pure_def.as_vector());
}

void RecFilter::addScan(
        Var x,
        RDom rx,
        float feedfwd,
        vector<float> feedback,
        Causality causality)
{
    Function f = contents.ptr->recfilter.function();

    if (!f.has_pure_definition()) {
        cerr << "Cannot add scans to recursive filter " << f.name()
            << " before specifying a pure definition" << endl;
        assert(false);
    }

    if (rx.dimensions() != 1) {
        cerr << "Each scan variable must be a 1D RDom" << endl;
        assert(false);
    }

    // csausality
    bool causal = (causality == CAUSAL);

    // image dimension for the scan
    int dimension = -1;
    Expr width = rx.x.extent();

    for (int i=0; i<f.args().size(); i++) {
        if (f.args()[i] == x.name()) {
            dimension = i;
        }
    }
    if (dimension == -1) {
        cerr << "Variable " << x << " is not one of the dimensions of the "
            << "recursive filter " << f.name() << endl;
        assert(false);
    }

    if (contents.ptr->split_info[dimension].num_splits>0 &&
            !equal(contents.ptr->split_info[dimension].image_width, width)) {
        cerr << "Extent of RDoms of all scans in the same dimension must be same" << endl;
        assert(false);
    }

    // create the LHS args, replace x by rx for causal and
    // x by w-1-rx for anticausal
    vector<Expr> args;
    for (int i=0; i<f.args().size(); i++) {
        if (i == dimension) {
            if (causal) {
                args.push_back(rx);
            } else {
                args.push_back(width-1-rx);
            }
        } else {
            args.push_back(Var(f.args()[i]));
        }
    }

    // RHS scan definition
    vector<Expr> values(f.values().size());
    for (int i=0; i<values.size(); i++) {
        Expr zero = make_zero(f.output_types()[i]);

        if (feedfwd != 0.0f) {
            values[i] = feedfwd * Call::make(f, args, i);
        } else {
            values[i] = zero;
        }

        for (int j=0; j<feedback.size(); j++) {
            if (feedback[j] != 0.0f) {
                vector<Expr> call_args = args;
                if (causal) {
                    call_args[dimension] = max(call_args[dimension]-(j+1),0);
                    values[i] += select(rx>j, feedback[j] * Call::make(f,call_args,i), zero);
                } else {
                    call_args[dimension] = min(call_args[dimension]+(j+1),width-1);
                    values[i] += select(rx>j, feedback[j] * Call::make(f,call_args,i), zero);
                }
            }
        }
        values[i] = simplify(values[i]);
    }
    f.define_reduction(args, values);

    int scan_order = feedback.size();

    // add details to the split info struct
    SplitInfo s = contents.ptr->split_info[dimension];
    s.rdom       .insert(s.rdom.begin(), rx);
    s.scan_id    .insert(s.scan_id.begin(), f.reductions().size()-1);
    s.scan_causal.insert(s.scan_causal.begin(), causal);
    s.num_splits   = s.num_splits+1;
    s.image_width  = width;
    s.filter_order = std::max(s.filter_order, scan_order);
    contents.ptr->split_info[dimension] = s;

    // add the feedforward and feedback coefficients
    int num_scans = s.feedback_coeff.width();
    int max_order = s.feedback_coeff.height();
    Image<float> feedfwd_coeff(num_scans+1);
    Image<float> feedback_coeff(num_scans+1, std::max(max_order,scan_order));

    // copy all the existing coeff to the new arrays
    for (int j=0; j<num_scans; j++) {
        feedfwd_coeff(j) = s.feedfwd_coeff(j);
        for (int i=0; i<scan_order; i++) {
            feedback_coeff(j,i) = s.feedback_coeff(j,i);
        }
    }

    // add the coeff of the newly added scan as the last row of coeff
    feedfwd_coeff(num_scans) = feedfwd;
    for (int i=0; i<scan_order; i++) {
        feedback_coeff(num_scans, i) = feedback[i];
    }

    // update the feedback and feedforward coeff matrices in all split info
    // structs for all dimensions (even though updating other dimensions is
    // redundant, but still performed for consistency)
    for (int i=0; i<contents.ptr->split_info.size(); i++) {
        contents.ptr->split_info[i].feedfwd_coeff  = feedfwd_coeff;
        contents.ptr->split_info[i].feedback_coeff = feedback_coeff;
    }
}

void RecFilter::addScan(Var x, RDom rx, Causality causal) {
    addScan(x, rx, 1.0f, vec(1.0f), causal);
}

void RecFilter::addScan(Var x, RDom rx, vector<float> feedback, Causality causal) {
    addScan(x, rx, 1.0f, feedback, causal);
}


Func RecFilter::func(void) {
    return contents.ptr->recfilter;
}

Func RecFilter::func(string func_name) {
    // build the dependency graph of the recursive filter
    if (contents.ptr->func_map.empty() || contents.ptr->func_list.empty()) {
        contents.ptr->func_map = find_transitive_calls(contents.ptr->recfilter.function());
        map<string,Function>::iterator f = contents.ptr->func_map.begin();
        while (f != contents.ptr->func_map.end()) {
            contents.ptr->func_list.push_back(f->second);
            f++;
        }
    }

    map<string,Function>::iterator f = contents.ptr->func_map.find(func_name);
    if (f != contents.ptr->func_map.end()) {
        return Func(f->second);
    }
    cerr << "Function " << func_name << " not found as a dependency of ";
    cerr << "recursive filter " << contents.ptr->recfilter.name() << endl;
    assert(false);
}

map<string,Func> RecFilter::funcs(void) {
    // build the dependency graph of the recursive filter
    if (contents.ptr->func_map.empty() || contents.ptr->func_list.empty()) {
        contents.ptr->func_map = find_transitive_calls(contents.ptr->recfilter.function());
        map<string,Function>::iterator f = contents.ptr->func_map.begin();
        while (f != contents.ptr->func_map.end()) {
            contents.ptr->func_list.push_back(f->second);
            f++;
        }
    }

    map<string,Func> func_map;
    map<string,Function>::iterator f = contents.ptr->func_map.begin();
    while (f != contents.ptr->func_map.end()) {
        func_map[f->first] = Func(f->second);
        f++;
    }
    return func_map;
}
