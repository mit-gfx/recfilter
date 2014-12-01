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

RecFilter::RecFilter(void) {}

RecFilter::RecFilter(Var x, Expr wx, string n)
    : RecFilter(vec(x), vec(wx), n) {}

RecFilter::RecFilter(Var x, Expr wx, Var y, Expr wy, string n)
    : RecFilter(vec(x,y), vec(wx,wy), n) {}

RecFilter::RecFilter(Var x, Expr wx, Var y, Expr wy, Var z, Expr wz, string n)
    : RecFilter(vec(x,y,z), vec(wx,wy,wz), n) {}

RecFilter::RecFilter(vector<Var> args, vector<Expr> widths, string n) {
    contents = new RecFilterContents;

    contents.ptr->name = n;
    contents.ptr->tiled          = false;
    contents.ptr->finalized      = false;
    contents.ptr->border_expr    = FLOAT_ZERO;
    contents.ptr->feedfwd_coeff  = Image<float>(0);
    contents.ptr->feedback_coeff = Image<float>(0,0);

    RecFilterFunc f;
    f.func = Function(n);
    f.func_category = INTRA_N;

    if (!contents.ptr->filter_info.empty()) {
        cerr << "Recursive filter dimensions already set" << endl;
        assert(false);
    }

    for (int i=0; i<args.size(); i++) {
        FilterInfo s;

        // set the variable and filter dimension
        s.var          = args[i];
        s.filter_dim   = i;

        // extent and domain of all scans in this dimension
        s.image_width  = widths[i];
        s.rdom         = RDom(0, s.image_width, unique_name("r"+s.var.name()));

        // default values for now
        s.num_scans      = 0;
        s.filter_order   = 0;

        contents.ptr->filter_info.push_back(s);

        // add tag the dimension as pure
        f.pure_var_category.insert(make_pair(args[i].name(), FULL));
    }

    contents.ptr->func.insert(make_pair(f.func.name(), f));
}

// -----------------------------------------------------------------------------

void RecFilter::define(Expr pure_def) {
    Function f = internal_function(contents.ptr->name).func;
    vector<string> args;
    for (int i=0; i<contents.ptr->filter_info.size(); i++) {
        args.push_back(contents.ptr->filter_info[i].var.name());
    }
    f.define(args, vec(pure_def));

    // bound the output buffer for each dimension
    for (int i=0; i<contents.ptr->filter_info.size(); i++) {
        Var  v      = contents.ptr->filter_info[i].var;
        Expr extent = contents.ptr->filter_info[i].image_width;
        Func(f).bound(v, 0, extent);
    }
}

void RecFilter::define(Tuple pure_def) {
    Function f = internal_function(contents.ptr->name).func;
    vector<string> args;
    for (int i=0; i<contents.ptr->filter_info.size(); i++) {
        args.push_back(contents.ptr->filter_info[i].var.name());
    }
    f.define(args, pure_def.as_vector());

    // bound the output buffer for each dimension
    for (int i=0; i<contents.ptr->filter_info.size(); i++) {
        Var  v      = contents.ptr->filter_info[i].var;
        Expr extent = contents.ptr->filter_info[i].image_width;
        Func(f).bound(v, 0, extent);
    }
}

void RecFilter::define(vector<Expr> pure_def) {
    Function f = internal_function(contents.ptr->name).func;
    vector<string> args;
    for (int i=0; i<contents.ptr->filter_info.size(); i++) {
        args.push_back(contents.ptr->filter_info[i].var.name());
    }
    f.define(args, pure_def);

    // bound the output buffer for each dimension
    for (int i=0; i<contents.ptr->filter_info.size(); i++) {
        Var  v      = contents.ptr->filter_info[i].var;
        Expr extent = contents.ptr->filter_info[i].image_width;
        Func(f).bound(v, 0, extent);
    }
}

// -----------------------------------------------------------------------------

void RecFilter::set_clamped_image_border(void) {
    Function f = internal_function(contents.ptr->name).func;
    if (f.has_pure_definition()) {
        cerr << "Border clamping must be set before calling RecFilter::define()" << endl;
        assert(false);
    }
    contents.ptr->border_expr = Expr();
}

void RecFilter::add_causal_filter(Var x, vector<float> coeff) {
    add_filter(x, coeff, true);
}

void RecFilter::add_anticausal_filter(Var x, vector<float> coeff) {
    add_filter(x, coeff, false);
}

void RecFilter::add_filter(Var x, vector<float> coeff, bool causal) {
    RecFilterFunc& rf = internal_function(contents.ptr->name);
    Function        f = rf.func;

    if (!f.has_pure_definition()) {
        cerr << "Cannot add scans to recursive filter " << f.name()
            << " before specifying an initial definition using RecFilter::define()" << endl;
        assert(false);
    }

    if (coeff.size()<2) {
        cerr << "Cannot add scan to recursive filter " << f.name()
            << " without feed forward and feedback coefficients" << endl;
        assert(false);
    }

    float feedfwd = coeff[0];
    vector<float> feedback;
    feedback.insert(feedback.begin(), coeff.begin()+1, coeff.end());

    // filter order and csausality
    int scan_order = feedback.size();

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

    // reduction domain for the scan
    RDom rx    = contents.ptr->filter_info[dimension].rdom;
    Expr width = contents.ptr->filter_info[dimension].image_width;

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
                if (contents.ptr->border_expr.defined()) {
                    values[i] += feedback[j] * select(rx>j, Call::make(f,call_args,i), contents.ptr->border_expr);
                } else {
                    values[i] += feedback[j] * Call::make(f,call_args,i);
                }
            }
        }
        values[i] = simplify(values[i]);
    }
    f.define_update(args, values);

    // add details to the split info struct
    FilterInfo s = contents.ptr->filter_info[dimension];
    s.scan_id    .insert(s.scan_id.begin(), f.updates().size()-1);
    s.scan_causal.insert(s.scan_causal.begin(), causal);
    s.num_scans   = s.num_scans+1;
    s.filter_order = std::max(s.filter_order, scan_order);
    contents.ptr->filter_info[dimension] = s;

    // copy all the existing feedback/feedfwd coeff to the new arrays
    // add the coeff of the newly added scan as the last row of coeff
    int num_scans = f.updates().size();
    int max_order = contents.ptr->feedback_coeff.height();
    Image<float> feedfwd_coeff(num_scans);
    Image<float> feedback_coeff(num_scans, std::max(max_order,scan_order));
    for (int j=0; j<num_scans-1; j++) {
        feedfwd_coeff(j) = contents.ptr->feedfwd_coeff(j);
        for (int i=0; i<contents.ptr->feedback_coeff.height(); i++) {
            feedback_coeff(j,i) = contents.ptr->feedback_coeff(j,i);
        }
    }
    feedfwd_coeff(num_scans-1) = feedfwd;
    for (int i=0; i<scan_order; i++) {
        feedback_coeff(num_scans-1, i) = feedback[i];
    }

    // update the feedback and feedforward coeff matrices in all filter info
    contents.ptr->feedfwd_coeff  = feedfwd_coeff;
    contents.ptr->feedback_coeff = feedback_coeff;

    // copy the dimension tags from pure def replacing x by rx
    // change the function tag from pure to scan
    map<string, VarTag> update_var_category = rf.pure_var_category;
    update_var_category.erase(x.name());
    update_var_category.insert(make_pair(rx.x.name(), FULL|SCAN));
    rf.update_var_category.push_back(update_var_category);
}

// -----------------------------------------------------------------------------

RecFilterSchedule RecFilter::intra_schedule(int id) {
    vector<string> func_list;

    map<string,RecFilterFunc>::iterator f_it = contents.ptr->func.begin();
    for (; f_it!=contents.ptr->func.end(); f_it++) {
        bool function_condition = false;
        FuncTag ftag = f_it->second.func_category;

        switch(id) {
            case 0: function_condition |= (ftag==FuncTag(INTRA_1) | ftag==FuncTag(INTRA_N)); break;
            case 1: function_condition |= (ftag==FuncTag(INTRA_N)); break;
            default:function_condition |= (ftag==FuncTag(INTRA_1)); break;
        }

        if (function_condition) {
            string func_name = f_it->second.func.name();

            // all functions which are REINDEX and call/called by this function
            map<string,RecFilterFunc>::iterator g_it = contents.ptr->func.begin();
            for (; g_it!=contents.ptr->func.end(); g_it++) {
                RecFilterFunc rf = g_it->second;
                if (rf.func_category==REINDEX) {
                    if (rf.callee_func==func_name || rf.caller_func==func_name) {
                        func_list.push_back(g_it->first);
                    }
                }
            }
            func_list.push_back(func_name);
        }
    }

    if (func_list.empty()) {
        cerr << "No " << (id==0 ? " " : (id==1 ? "1D " : "nD "));
        cerr << "intra tile functions to schedule" << endl;
        assert(false);
    }
    return RecFilterSchedule(*this, func_list);
}

RecFilterSchedule RecFilter::inter_schedule(void) {
    vector<string> func_list;

    map<string,RecFilterFunc>::iterator f_it = contents.ptr->func.begin();
    for (; f_it!=contents.ptr->func.end(); f_it++) {
        if (f_it->second.func_category==INTER) {
            string func_name = f_it->second.func.name();
            func_list.push_back(func_name);
        }
    }

    if (func_list.empty()) {
        cerr << "No inter tile functions to schedule" << endl;
        assert(false);
    }
    return RecFilterSchedule(*this, func_list);
}

VarTag RecFilter::full_parallel (int id) { return VarTag(FULL, id); }
VarTag RecFilter::full_scan     (int id) { return VarTag(FULL|SCAN, id); }
VarTag RecFilter::inner_parallel(int id) { return VarTag(INNER, id); }
VarTag RecFilter::outer_parallel(int id) { return VarTag(OUTER, id); }

VarTag RecFilter::inner_scan(void) { return (INNER|SCAN); }
VarTag RecFilter::outer_scan(void) { return (OUTER|SCAN); }
VarTag RecFilter::inner_tail(void) { return (INNER|TAIL); }

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
    if (!contents.ptr->finalized) {
        finalize(target);
    }

    Func F(internal_function(contents.ptr->name).func);
    if (!filename.empty()) {
        F.compile_to_lowered_stmt(filename, HTML, target);
    }
    F.compile_jit(target);
}

void RecFilter::realize(Buffer out, int iterations) {
    if (!contents.ptr->finalized) {
        finalize(get_jit_target_from_environment());
    }

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
