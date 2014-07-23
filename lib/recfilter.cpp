#include "recfilter.h"
#include "recfilter_utils.h"

using namespace Halide;
using namespace Halide::Internal;

using std::string;
using std::cerr;
using std::endl;
using std::vector;
using std::queue;
using std::map;

RecFilter::RecFilter(string n) {
    contents = new RecFilterContents;
    contents.ptr->recfilter = Func(n);
}

void RecFilter::setArgs(vector<Var> args, vector<Expr> width) {
    if (args.size() != width.size()) {
        cerr << "Each dimension of recursive filter " << contents.ptr->recfilter.name()
            << " must have a mapped width" << endl;
        assert(false);
    }

    for (int i=0; i<args.size(); i++) {
        SplitInfo s;
        s.clamp_border = false;

        s.var          = args[i];
        s.image_width  = width[i];
        s.filter_dim   = i;

        // default values for now
        s.num_splits   = 0;
        s.filter_order = 0;
        s.tile_width   = width[i];
        s.num_tiles    = 1;

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

void RecFilter::addScan(bool causal, Var x, RDom rx, float feedfwd, vector<float> feedback) {
    Function f = contents.ptr->recfilter.function();

    if (f.has_pure_definition()) {
        cerr << "Cannot add scans to recursive filter " << f.name()
            << " before specifying a pure definition" << endl;
        assert(false);
    }

    if (rx.dimensions() != 1) {
        cerr << "Each scan variable must be a 1D RDom" << endl;
        assert(false);
    }

    // image dimension for the scan
    int dimension = -1;
    Expr extent = rx.x.extent();
    Expr width;

    for (int i=0; i<f.args().size(); i++) {
        if (f.args()[i] == x.name()) {
            dimension = i;
            width = contents.ptr->split_info[i].image_width;
        }
    }
    if (dimension == -1) {
        cerr << "Variable " << x << " is not one of the dimensions of the "
            << "recursive filter " << f.name() << endl;
        assert(false);
    }

    if (equal(width,extent)) {
        cerr << "Extent of RDom must equal image width in the given dimension" << endl;
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
                    call_args[dimension] = max(call_args[dimension]-j-1,0);
                    values[i] += select(rx>j, feedback[j] * Call::make(f,call_args,i), zero);
                } else {
                    call_args[dimension] = min(call_args[dimension]+j+1,width-1);
                    values[i] += select(rx>j, feedback[j] * Call::make(f,call_args,i), zero);
                }
            }
        }
        values[i] = simplify(values[i]);
    }
    f.define_reduction(args, values);

    // add details to the split info struct
    SplitInfo s = contents.ptr->split_info[dimension];
    s.rdom       .insert(s.rdom.begin(), rx);
    s.scan_id    .insert(s.scan_id.begin(), f.reductions().size()-1);
    s.scan_causal.insert(s.scan_causal.begin(), causal);
    s.num_splits   = s.num_splits+1;
    s.filter_order = std::max(s.filter_order, int(feedback.size()));
    contents.ptr->split_info[dimension] = s;
}

Func RecFilter::func(string func_name) {
    map<string,Function>::iterator f = contents.ptr->func_map.find(func_name);
    if (f != contents.ptr->func_map.end()) {
        return Func(f->second);
    }
    cerr << "Function " << func_name << " not found as a dependency of ";
    cerr << "recursive filter " << contents.ptr->recfilter.name() << endl;
    assert(false);
}
