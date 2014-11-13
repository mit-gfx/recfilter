#include "recfilter.h"
#include "recfilter_internals.h"
#include "modifiers.h"

using std::string;
using std::cerr;
using std::endl;
using std::vector;
using std::map;
using std::ostream;

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

    RecFilterFunc f;
    f.func = contents.ptr->recfilter.function();
    f.func_category = FULL_RESULT;
    contents.ptr->func.insert(make_pair(f.func.name(), f));
}

void RecFilter::setArgs(Var x)              { setArgs(vec(x));     }
void RecFilter::setArgs(Var x, Var y)       { setArgs(vec(x,y));   }
void RecFilter::setArgs(Var x, Var y, Var z){ setArgs(vec(x,y,z)); }

void RecFilter::setArgs(vector<Var> args) {
    RecFilterFunc& f = internal_function( contents.ptr->recfilter.name() );

    for (int i=0; i<args.size(); i++) {
        SplitInfo s;

        // set the variable and filter dimension
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

        // add tag the dimension as pure
        f.pure_var_category.insert(make_pair(args[i].name(), PURE_DIMENSION));
    }
}

// -----------------------------------------------------------------------------

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

void RecFilter::define(vector<Expr> pure_def) {
    vector<string> args;
    for (int i=0; i<contents.ptr->split_info.size(); i++) {
        args.push_back(contents.ptr->split_info[i].var.name());
    }
    contents.ptr->recfilter.function().define(args, pure_def);
}

// -----------------------------------------------------------------------------

void RecFilter::addScan(
        Var x,
        RDom rx,
        float feedfwd,
        vector<float> feedback,
        Causality causality,
        Border border_mode,
        Expr bexpr)
{
    RecFilterFunc& rf = internal_function( contents.ptr->recfilter.name() );
    Function        f = contents.ptr->recfilter.function();

    if (!f.has_pure_definition()) {
        cerr << "Cannot add scans to recursive filter " << f.name()
            << " before specifying a pure definition" << endl;
        assert(false);
    }

    if (rx.dimensions() != 1) {
        cerr << "Each scan variable must be a 1D RDom" << endl;
        assert(false);
    }

    if (!equal(rx.x.min(), INT_ZERO)) {
        cerr << "Scan variables must have 0 as lower limit" << endl;
        assert(false);
    }

    // copy the dimension tags from pure def replacing x by rx
    // change the function tag from pure to scan
    map<string, VarTag> update_var_category = rf.pure_var_category;
    update_var_category.erase(x.name());
    update_var_category.insert(make_pair(rx.x.name(), SCAN_DIMENSION));
    rf.update_var_category.push_back(update_var_category);

    // csausality
    bool causal = (causality == CAUSAL);

    // border pixels
    Expr border_expr;
    switch (border_mode) {
        case CLAMP_TO_ZERO:
            border_expr = FLOAT_ZERO;
            break;

        case CLAMP_TO_EXPR:
            border_expr = bexpr;
            if (expr_depends_on_var(border_expr, x.name())) {
                cerr << "Image border expression for scan along " << x.name()
                    << " must not depend upon " << x.name() << endl;
            }
            if (expr_depends_on_var(border_expr, rx.x.name())) {
                cerr << "Image border expression for scan along " << rx.x.name()
                    << " must not depend upon " << rx.x.name() << endl;
            }
            break;

        case CLAMP_TO_SELF:
        default:
            border_expr = Expr();
            break;
    }

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

    if (contents.ptr->split_info[dimension].rdom.defined() &&
            !rx.same_as(contents.ptr->split_info[dimension].rdom)) {
        cerr << "All scans in the same dimension must use the same RDom" << endl;
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
        values[i] = feedfwd * Call::make(f, args, i);

        for (int j=0; j<feedback.size(); j++) {
            if (feedback[j] != 0.0f) {
                vector<Expr> call_args = args;
                if (causal) {
                    call_args[dimension] = max(call_args[dimension]-(j+1),0);
                } else {
                    call_args[dimension] = min(call_args[dimension]+(j+1),width-1);
                }
                if (border_expr.defined()) {
                    values[i] += feedback[j] * select(rx>j, Call::make(f,call_args,i), border_expr);
                } else {
                    values[i] += feedback[j] * Call::make(f,call_args,i);
                }
            }
        }
        values[i] = simplify(values[i]);
    }
    f.define_update(args, values);

    int scan_order = feedback.size();

    // add details to the split info struct
    SplitInfo s = contents.ptr->split_info[dimension];
    s.scan_id    .insert(s.scan_id.begin(), f.updates().size()-1);
    s.scan_causal.insert(s.scan_causal.begin(), causal);
    s.border_expr.insert(s.border_expr.begin(), border_expr);
    s.rdom         = rx;
    s.num_splits   = s.num_splits+1;
    s.image_width  = width;
    s.filter_order = std::max(s.filter_order, scan_order);
    contents.ptr->split_info[dimension] = s;


    // copy all the existing feedback/feedfwd coeff to the new arrays
    // add the coeff of the newly added scan as the last row of coeff
    int num_scans = f.updates().size();
    int max_order = s.feedback_coeff.height();
    Image<float> feedfwd_coeff(num_scans);
    Image<float> feedback_coeff(num_scans, std::max(max_order,scan_order));
    for (int j=0; j<num_scans-1; j++) {
        feedfwd_coeff(j) = s.feedfwd_coeff(j);
        for (int i=0; i<s.feedback_coeff.height(); i++) {
            feedback_coeff(j,i) = s.feedback_coeff(j,i);
        }
    }
    feedfwd_coeff(num_scans-1) = feedfwd;
    for (int i=0; i<scan_order; i++) {
        feedback_coeff(num_scans-1, i) = feedback[i];
    }

    // update the feedback and feedforward coeff matrices in all split info
    // structs for all dimensions (even though updating other dimensions is
    // redundant, but still performed for consistency)
    for (int i=0; i<contents.ptr->split_info.size(); i++) {
        contents.ptr->split_info[i].feedfwd_coeff  = feedfwd_coeff;
        contents.ptr->split_info[i].feedback_coeff = feedback_coeff;
    }
}

void RecFilter::addScan(Var x, RDom rx, Causality c, Border b, Expr bexpr) {
    addScan(x, rx, 1.0f, vec(1.0f), c, b, bexpr);
}

void RecFilter::addScan(Var x, RDom rx, vector<float> fb, Causality c, Border b, Expr bexpr) {
    addScan(x, rx, 1.0f, fb, c, b, bexpr);
}

// -----------------------------------------------------------------------------

Func RecFilter::func(void) {
    return contents.ptr->recfilter;
}

Func RecFilter::func(string func_name) {
    map<string,RecFilterFunc>::iterator f = contents.ptr->func.find(func_name);
    if (f != contents.ptr->func.end()) {
        return Func(f->second.func);
    } else {
        cerr << "Function " << func_name << " not found as a dependency of ";
        cerr << "recursive filter " << contents.ptr->recfilter.name() << endl;
        assert(false);
    }
}

vector<Func> RecFilter::funcs(void) {
    vector<Func> func_list;
    map<string,RecFilterFunc>::iterator f = contents.ptr->func.begin();
    map<string,RecFilterFunc>::iterator fe= contents.ptr->func.end();
    while (f!=fe) {
        func_list.push_back(Func(f->second.func));
        f++;
    }
    return func_list;
}

RecFilterFunc& RecFilter::internal_function(string func_name) {
    map<string,RecFilterFunc>::iterator f = contents.ptr->func.find(func_name);
    if (f != contents.ptr->func.end()) {
        return f->second;
    } else {
        cerr << "Function " << func_name << " not found as a dependency of ";
        cerr << "recursive filter " << contents.ptr->recfilter.name() << endl;
        assert(false);
    }
}

// -----------------------------------------------------------------------------

void RecFilter::compile_jit(Target target, string filename) {
    if (!filename.empty()) {
        contents.ptr->recfilter.compile_to_lowered_stmt(filename, HTML, target);
    }
    contents.ptr->recfilter.compile_jit(target);
}

void RecFilter::realize(Buffer out, int iterations) {
    // upload all buffers to device
    map<string,Buffer> buff = extract_buffer_calls(contents.ptr->recfilter);
    for (map<string,Buffer>::iterator b=buff.begin(); b!=buff.end(); b++) {
        b->second.copy_to_dev();
    }

    // profiling realizations without copying result back to host
    for (int i=0; i<iterations-1; i++) {
        contents.ptr->recfilter.realize(out);
    }

    // last realization copies result back to host
    contents.ptr->recfilter.realize(out);
    out.copy_to_host();
    out.free_dev_buffer();
}

// -----------------------------------------------------------------------------

void RecFilter::remove_pure_def(string func_name) {
    Function f = func(func_name).function();

    vector<string> args   = f.args();
    vector<Expr>   values = f.values();
    vector<UpdateDefinition> updates = f.updates();

    // nothing to do if function has not update defs
    if (updates.empty()) {
        return;
    }

    // add pure def to the first update def
    {
        for (int j=0; j<updates[0].values.size(); j++) {
            // replace pure args by update def args in the pure value
            Expr val = values[j];
            for (int k=0; k<args.size(); k++) {
                val = substitute(args[k], updates[0].args[k], val);
            }

            // remove let statements in the expression because we need to
            // compare calling args
            updates[0].values[j] = remove_lets(updates[0].values[j]);

            // remove call to current pixel of the function
            updates[0].values[j] = substitute_func_call_with_args(f.name(),
                    updates[0].args, val, updates[0].values[j]);
        }
    }

    // set all pure defs to zero or undef
    for (int i=0; i<values.size(); i++) {
        values[i] = FLOAT_UNDEF;
    }

    f.clear_all_definitions();
    f.define(args, values);
    for (int i=0; i<updates.size(); i++) {
        f.define_update(updates[i].args, updates[i].values);
    }
}

// -----------------------------------------------------------------------------

void RecFilter::generate_hl_code(ostream &s) const {
    map<string,RecFilterFunc>::iterator f;
    for (f=contents.ptr->func.begin(); f!=contents.ptr->func.end(); f++) {
        s << f->second << "\n";
    }
    s << "\n";
    for (f=contents.ptr->func.begin(); f!=contents.ptr->func.end(); f++) {
        s << f->second.func << "\n";
    }
    s << "\n";
    for (f=contents.ptr->func.begin(); f!=contents.ptr->func.end(); f++) {
        map<int,vector<string> >::iterator sit = f->second.schedule.begin();
        map<int,vector<string> >::iterator se  = f->second.schedule.end();
        while (sit!=se) {
            int def = sit->first;
            vector<string> str = sit->second;
            if (!str.empty()) {
                s << f->second.func.name();
                if (def>=0) {
                    s << ".update(" << def << ")";
                }
            }
            for (int i=0; i<str.size(); i++) {
                s << "." << str[i];
            }
            s << ";\n";
            sit++;
        }
    }
    s << "\n";
}
