#include "recfilter.h"
#include "recfilter_internals.h"
#include "modifiers.h"

using std::string;
using std::cerr;
using std::endl;
using std::vector;
using std::map;
using std::ostream;
using std::stringstream;

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
    contents.ptr->name = n;

    RecFilterFunc f;
    f.func = Function(n);
    f.func_category = FULL_RESULT;
    contents.ptr->func.insert(make_pair(f.func.name(), f));
}

void RecFilter::set_args(Var x, Expr wx) {
    set_args(vec(x), vec(wx));
}

void RecFilter::set_args(Var x, Var y, Expr wx, Expr wy) {
    set_args(vec(x,y), vec(wx,wy));
}

void RecFilter::set_args(Var x, Var y, Var z, Expr wx, Expr wy, Expr wz) {
    set_args(vec(x,y,z), vec(wx,wy,wz));
}

void RecFilter::set_args(vector<Var> args, vector<Expr> widths) {
    RecFilterFunc& f = internal_function(contents.ptr->name);

    if (contents.ptr->split_info.empty()) {
        cerr << "Recursive filter dimensions already set" << endl;
        assert(false);
    }

    for (int i=0; i<args.size(); i++) {
        SplitInfo s;

        // set the variable and filter dimension
        s.var          = args[i];
        s.filter_dim   = i;

        // extent and domain of all scans in this dimension
        s.image_width  = widths[i];
        s.rdom         = RDom(0, s.image_width, unique_name("r"+s.var.name()));
        s.tile_width   = s.image_width;
        s.num_tiles    = 1;

        // default values for now
        s.num_splits   = 0;
        s.filter_order = 0;

        s.feedfwd_coeff  = Image<float>(0);
        s.feedback_coeff = Image<float>(0,0);

        contents.ptr->split_info.push_back(s);

        // bound the output buffer for each dimension
        Func(f.func).bound(s.var, 0, s.image_width);

        // add tag the dimension as pure
        f.pure_var_category.insert(make_pair(args[i].name(), PURE_DIMENSION));
    }
}

// -----------------------------------------------------------------------------

void RecFilter::define(Expr pure_def) {
    Function f = internal_function(contents.ptr->name).func;
    vector<string> args;
    for (int i=0; i<contents.ptr->split_info.size(); i++) {
        args.push_back(contents.ptr->split_info[i].var.name());
    }
    f.define(args, vec(pure_def));
}

void RecFilter::define(Tuple pure_def) {
    Function f = internal_function(contents.ptr->name).func;
    vector<string> args;
    for (int i=0; i<contents.ptr->split_info.size(); i++) {
        args.push_back(contents.ptr->split_info[i].var.name());
    }
    f.define(args, pure_def.as_vector());
}

void RecFilter::define(vector<Expr> pure_def) {
    Function f = internal_function(contents.ptr->name).func;
    vector<string> args;
    for (int i=0; i<contents.ptr->split_info.size(); i++) {
        args.push_back(contents.ptr->split_info[i].var.name());
    }
    f.define(args, pure_def);
}

// -----------------------------------------------------------------------------

void RecFilter::set_image_border(Border b, Halide::Expr expr) {
//    Expr border_expr;
//    switch (border_mode) {
//        case CLAMP_TO_ZERO:
//            border_expr = FLOAT_ZERO;
//            break;
//
//        case CLAMP_TO_EXPR:
//            border_expr = bexpr;
//            if (expr_depends_on_var(border_expr, x.name())) {
//                cerr << "Image border expression for scan along " << x.name()
//                    << " must not depend upon " << x.name() << endl;
//            }
//            if (expr_depends_on_var(border_expr, rx.x.name())) {
//                cerr << "Image border expression for scan along " << rx.x.name()
//                    << " must not depend upon " << rx.x.name() << endl;
//            }
//            break;
//
//        case CLAMP_TO_SELF:
//        default:
//            border_expr = Expr();
//            break;
//    }
//
//    for (int i=0; i<contents.ptr->split_info.size(); i++) {
//        SplitInfo s = contents.ptr->split_info[i];
//        s.border_expr.insert(s.border_expr.begin(), border_expr);
//        contents.ptr->split_info[i] = s;
//    }
}

void RecFilter::add_filter(
        Var x,
        float feedfwd,
        vector<float> feedback,
        Causality causality,
        Border border_mode,
        Halide::Expr bexpr)
{
    RecFilterFunc& rf = internal_function(contents.ptr->name);
    Function        f = rf.func;

    if (!f.has_pure_definition()) {
        cerr << "Cannot add scans to recursive filter " << f.name()
            << " before specifying a pure definition" << endl;
        assert(false);
    }

    // filter order and csausality
    int scan_order = feedback.size();
    bool causal = (causality == CAUSAL);

    // image dimension for the scan
    int dimension = -1;
    for (int i=0; dimension<0 && i<f.args().size(); i++) {
        if (f.args()[i] == x.name()) {
            dimension = i;
        }
    }
    if (dimension == -1) {
        cerr << "Variable " << x << " is not one of the dimensions of the "
            << "recursive filter " << f.name() << endl;
        assert(false);
    }

    // check the border expression
    Expr border_expr;
    {
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
                break;

            case CLAMP_TO_SELF:
            default:
                border_expr = Expr();
                break;
        }
    }

    // add details to the split info struct
    SplitInfo s = contents.ptr->split_info[dimension];
    s.scan_id    .insert(s.scan_id.begin(), f.updates().size()-1);
    s.scan_causal.insert(s.scan_causal.begin(), causal);
    s.border_expr.insert(s.border_expr.begin(), border_expr);
    s.num_splits   = s.num_splits+1;
    s.filter_order = std::max(s.filter_order, scan_order);
    contents.ptr->split_info[dimension] = s;

    // reduction domain for the scan
    RDom rx    = contents.ptr->split_info[dimension].rdom;
    Expr width = contents.ptr->split_info[dimension].image_width;

    // copy the dimension tags from pure def replacing x by rx
    // change the function tag from pure to scan
    map<string, VarTag> update_var_category = rf.pure_var_category;
    update_var_category.erase(x.name());
    update_var_category.insert(make_pair(rx.x.name(), SCAN_DIMENSION));
    rf.update_var_category.push_back(update_var_category);

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

// -----------------------------------------------------------------------------

RecFilterSchedule RecFilter::intra_schedule(void) {
    FuncTag ftag = INTRA_TILE_SCAN;
    vector<string> func_list;
    map<string,RecFilterFunc>::iterator f_it  = contents.ptr->func.begin();
    map<string,RecFilterFunc>::iterator f_end = contents.ptr->func.end();
    while (f_it != f_end) {
        if (f_it->second.func_category & ftag) {
            func_list.push_back(f_it->second.func.name());
        }
        f_it++;
    }
    if (func_list.empty()) {
        cerr << "No function has the scheduling tag " << ftag << endl;
        assert(false);
    }
    return RecFilterSchedule(*this, func_list);
}

RecFilterSchedule RecFilter::inter_schedule(void) {
    FuncTag ftag = INTER_TILE_SCAN;
    vector<string> func_list;
    map<string,RecFilterFunc>::iterator f_it  = contents.ptr->func.begin();
    map<string,RecFilterFunc>::iterator f_end = contents.ptr->func.end();
    while (f_it != f_end) {
        if (f_it->second.func_category & ftag) {
            func_list.push_back(f_it->second.func.name());
        }
        f_it++;
    }
    if (func_list.empty()) {
        cerr << "No function has the scheduling tag " << ftag << endl;
        assert(false);
    }
    return RecFilterSchedule(*this, func_list);
}

// -----------------------------------------------------------------------------

Func RecFilter::func(void) {
    return Func(internal_function(contents.ptr->name).func);
}

Func RecFilter::func(string func_name) {
    map<string,RecFilterFunc>::iterator f = contents.ptr->func.find(func_name);
    if (f != contents.ptr->func.end()) {
        return Func(f->second.func);
    } else {
        cerr << "Function " << func_name << " not found as a dependency of ";
        cerr << "recursive filter " << contents.ptr->name << endl;
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
        cerr << "recursive filter " << contents.ptr->name << endl;
        assert(false);
    }
}

// -----------------------------------------------------------------------------

void RecFilter::compile_jit(Target target, string filename) {
    Func F(internal_function(contents.ptr->name).func);
    if (!filename.empty()) {
        F.compile_to_lowered_stmt(filename, HTML, target);
    }
    F.compile_jit(target);
}

void RecFilter::realize(Buffer out, int iterations) {
    Func F(internal_function(contents.ptr->name).func);

    // upload all buffers to device
    map<string,Buffer> buff = extract_buffer_calls(F);
    for (map<string,Buffer>::iterator b=buff.begin(); b!=buff.end(); b++) {
        b->second.copy_to_dev();
    }

    // profiling realizations without copying result back to host
    for (int i=0; i<iterations-1; i++) {
        F.realize(out);
    }

    // last realization copies result back to host
    F.realize(out);
    out.copy_to_host();
    out.free_dev_buffer();
}

// -----------------------------------------------------------------------------

string RecFilter::print_synopsis(void) const {
    stringstream s;
    map<string,RecFilterFunc>::iterator f;
    for (f=contents.ptr->func.begin(); f!=contents.ptr->func.end(); f++) {
        s << f->second << "\n";
    }
    s << "\n";
    return s.str();
}

string RecFilter::print_schedule(void) const {
    stringstream s;
    map<string,RecFilterFunc>::iterator f;
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
    return s.str();
}

string RecFilter::print_functions(void) const {
    stringstream s;
    map<string,RecFilterFunc>::iterator f;
    for (f=contents.ptr->func.begin(); f!=contents.ptr->func.end(); f++) {
        s << f->second.func << "\n";
    }
    s << "\n";
    return s.str();
}

string RecFilter::print_hl_code(void) const {
    string a = print_synopsis();
    string b = print_functions();
    string c = print_schedule();
    return a+b+c;
}
