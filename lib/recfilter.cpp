#include "recfilter.h"
#include "recfilter_internals.h"
#include "modifiers.h"
#include "timing.h"

#define AUTO_SCHEDULE_MAX_DIMENSIONS 3

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

// -----------------------------------------------------------------------------

RecFilterRefVar::RecFilterRefVar(RecFilter r, std::vector<RecFilterDim> a) :
    rf(r), args(a) {}

void RecFilterRefVar::operator=(Expr pure_def) {
    rf.define(args, vec(pure_def));
}

void RecFilterRefVar::operator=(const Tuple &pure_def) {
    rf.define(args, pure_def.as_vector());
}

void RecFilterRefVar::operator=(vector<Expr> pure_def) {
    rf.define(args, pure_def);
}

void RecFilterRefVar::operator=(FuncRefVar pure_def) {
    rf.define(args, {Expr(pure_def)});
}

void RecFilterRefVar::operator=(FuncRefExpr pure_def) {
    rf.define(args, {Expr(pure_def)});
}

RecFilterRefVar::operator Expr(void) {
    return this->operator[](0);
}

Expr RecFilterRefVar::operator[](int i) {
    Function main_func = rf.as_func().function();
    vector<Expr> expr_args;
    for (int j=0; j<args.size(); j++) {
        expr_args.push_back(args[j]);
    }
    if (i>=main_func.outputs()) {
        cerr << "Could not find output buffer " << i
             << " in recursive filter " << rf.name();
        assert(false);
    }
    return Call::make(main_func, expr_args, i);
}

RecFilterRefExpr::RecFilterRefExpr(RecFilter r, std::vector<Expr> a) :
    rf(r), args(a) {}

RecFilterRefExpr::operator Expr(void) {
    return this->operator[](0);
}

Expr RecFilterRefExpr::operator[](int i) {
    Function main_func = rf.as_func().function();
    if (i>=main_func.outputs()) {
        cerr << "Could not find output buffer " << i
             << " in recursive filter " << rf.name();
        assert(false);
    }
    return Call::make(main_func, args, i);
}

// -----------------------------------------------------------------------------

RecFilter::RecFilter(string name) {
    contents = new RecFilterContents;
    if (name.empty()) {
        contents.ptr->name = unique_name("R");
    } else {
        contents.ptr->name = unique_name(name);
    }
    contents.ptr->tiled          = false;
    contents.ptr->finalized      = false;
    contents.ptr->compiled       = false;
    contents.ptr->clamped_border = false;
    contents.ptr->feedfwd_coeff  = Image<float>(0);
    contents.ptr->feedback_coeff = Image<float>(0,0);

    contents.ptr->target = get_jit_target_from_environment();
    if (contents.ptr->target.to_string().empty()) {
        cerr << "Warning: HL_JIT_TARGET not set, using default" << endl;
    }
}

RecFilter& RecFilter::operator=(const RecFilter &f) {
    contents = f.contents;
    return *this;
}

string RecFilter::name(void) const {
    return contents.ptr->name;
}

RecFilterRefVar RecFilter::operator()(RecFilterDim x) {
    return RecFilterRefVar(*this,vec(x));
}
RecFilterRefVar RecFilter::operator()(RecFilterDim x, RecFilterDim y) {
    return RecFilterRefVar(*this,vec(x,y));
}
RecFilterRefVar RecFilter::operator()(RecFilterDim x, RecFilterDim y, RecFilterDim z){
    return RecFilterRefVar(*this,vec(x,y,z));
}
RecFilterRefVar RecFilter::operator()(vector<RecFilterDim> x) {
    return RecFilterRefVar(*this, x);
}

RecFilterRefExpr RecFilter::operator()(Expr x) {
    return RecFilterRefExpr(*this,vec(x));
}
RecFilterRefExpr RecFilter::operator()(Expr x, Expr y) {
    return RecFilterRefExpr(*this,vec(x,y));
}
RecFilterRefExpr RecFilter::operator()(Expr x, Expr y, Expr z) {
    return RecFilterRefExpr(*this,vec(x,y,z));
}
RecFilterRefExpr RecFilter::operator()(vector<Expr> x) {
    return RecFilterRefExpr(*this, x);
}

void RecFilter::define(vector<RecFilterDim> pure_args, vector<Expr> pure_def) {
    assert(contents.ptr);
    assert(!pure_args.empty());
    assert(!pure_def.empty());

    contents.ptr->type = pure_def[0].type();
    for (int i=1; i<pure_def.size(); i++) {
        if (contents.ptr->type != pure_def[i].type()) {
            cerr << "Type of all Tuple elements in filter definition must be same" << endl;
            assert(false);
        }
    }

    if (!contents.ptr->filter_info.empty()) {
        cerr << "Recursive filter " << contents.ptr->name << " already defined" << endl;
        assert(false);
    }

    RecFilterFunc rf;
    rf.func = Function(contents.ptr->name);
    rf.func_category = INTRA_N;

    // add the arguments

    for (int i=0; i<pure_args.size(); i++) {
        FilterInfo s;

        // set the variable and filter dimension
        s.filter_dim   = i;
        s.var          = pure_args[i].var();

        // extent and domain of all scans in this dimension
        s.image_width = pure_args[i].extent();
        s.tile_width  = s.image_width;
        s.rdom        = RDom(0, s.image_width, unique_name("r"+s.var.name()));

        // default values for now
        s.num_scans      = 0;
        s.filter_order   = 0;

        contents.ptr->filter_info.push_back(s);

        // add tag the dimension as pure
        rf.pure_var_category.insert(make_pair(pure_args[i].var().name(), VarTag(FULL,i)));
    }

    contents.ptr->func.insert(make_pair(rf.func.name(), rf));

    // add the right hand side definition
    Function f = rf.func;

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
    if (!contents.ptr->filter_info.empty()) {
        cerr << "Recursive filter " << contents.ptr->name << " already defined" << endl;
        assert(false);
    }
    contents.ptr->clamped_border = true;
}

void RecFilter::add_filter(RecFilterDim x, vector<float> coeff) {
    add_filter(RecFilterDimAndCausality(x,true), coeff);
}

void RecFilter::add_filter(RecFilterDimAndCausality x, vector<float> coeff) {
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

    bool causal = x.causal();

    float feedfwd = coeff[0];
    vector<float> feedback;
    feedback.insert(feedback.begin(), coeff.begin()+1, coeff.end());

    // filter order and csausality
    int scan_order = feedback.size();

    // image dimension for the scan
    int dimension = -1;
    for (int i=0; dimension<0 && i<f.args().size(); i++) {
        if (f.args()[i] == x.var().name()) {
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
        values[i] = Cast::make(contents.ptr->type,feedfwd) *
            Call::make(f, args, i);

        for (int j=0; j<feedback.size(); j++) {
            vector<Expr> call_args = args;
            if (causal) {
                call_args[dimension] = max(call_args[dimension]-(j+1),0);
            } else {
                call_args[dimension] = min(call_args[dimension]+(j+1),width-1);
            }
            if (contents.ptr->clamped_border) {
                values[i] += Cast::make(contents.ptr->type,feedback[j]) *
                    Call::make(f,call_args,i);
            } else {
                values[i] += Cast::make(contents.ptr->type,feedback[j]) *
                    select(rx>j, Call::make(f,call_args,i), make_zero(contents.ptr->type));
            }
        }
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

    // decrement the full count of all vars whose count was more than count of x
    int count_x = update_var_category[x.var().name()].count();
    update_var_category.erase(x.var().name());
    map<string,VarTag>::iterator vit;
    for (vit=update_var_category.begin(); vit!=update_var_category.end(); vit++) {
        if (vit->second.check(FULL) && !vit->second.check(SCAN)) {
            int count = vit->second.count();
            if (count > count_x) {
                update_var_category[vit->first] = VarTag(FULL,count-1);
            }
        }
    }
    update_var_category.insert(make_pair(rx.x.name(), FULL|SCAN));
    rf.update_var_category.push_back(update_var_category);
}

// -----------------------------------------------------------------------------

RecFilterSchedule RecFilter::full_schedule(void) {
    if (contents.ptr->tiled) {
        cerr << "Filter is tiled, use RecFilter::intra_schedule() "
             << "and RecFilter::inter_schedule()\n" << endl;
        assert(false);
    }
    return RecFilterSchedule(*this, { name() });
}

RecFilterSchedule RecFilter::intra_schedule(int id) {
    if (!contents.ptr->tiled) {
        cerr << "\nNo intra-tile terms to schedule in a non-tiled filter" << endl;
        cerr << "Use RecFilter::schedule() and Halide scheduling API\n" << endl;
        assert(false);
    }

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
                    if (rf.producer_func==func_name || rf.consumer_func==func_name) {
                        func_list.push_back(g_it->first);
                    }
                }
            }
            func_list.push_back(func_name);
        }
    }

    if (func_list.empty()) {
        cerr << "Warning: No " << (id==0 ? " " : (id==1 ? "1D " : "nD "));
        cerr << "intra tile functions to schedule" << endl;
    }
    return RecFilterSchedule(*this, func_list);
}

RecFilterSchedule RecFilter::inter_schedule(void) {
    if (!contents.ptr->tiled) {
        cerr << "\nNo inter-tile terms to schedule in a non-tiled filter" << endl;
        cerr << "Use RecFilter::schedule() and Halide scheduling API\n" << endl;
        assert(false);
    }

    vector<string> func_list;

    map<string,RecFilterFunc>::iterator f_it = contents.ptr->func.begin();
    for (; f_it!=contents.ptr->func.end(); f_it++) {
        if (f_it->second.func_category==INTER) {
            string func_name = f_it->second.func.name();
            func_list.push_back(func_name);
        }
    }

    if (func_list.empty()) {
        cerr << "Warning: No inter tile functions to schedule" << endl;
    }

    return RecFilterSchedule(*this, func_list);
}

void RecFilter::compute_at(Func external, Var looplevel) {
    // Func representing the final result
    RecFilterFunc& rF = internal_function(name());
    Function f        = rF.func;

    // check that the filter does not depend upon F
    if (contents.ptr->func.find(external.name()) != contents.ptr->func.end()) {
        cerr << "Cannot compute " << name() << " at " << external.name()
             << " because it is a consumer of " << external.name() << endl;
        assert(false);
    }

    // this function must not have a consumer because this is now being set to
    // be computed at something else
    if (!rF.consumer_func.empty() || rF.external_consumer_func.defined()) {
        cerr << "Cannot compute " << name() << " at " << external.name()
             << " because it already has a consumer " << endl;
        assert(false);
    }

    // check that the compute looplevel of the final result is not already set
    if (!f.schedule().compute_level().is_inline() ||
        !f.schedule().store_level().is_inline())
    {
        cerr << "Cannot compute " << name() << " inside " << external.name()
             << " because it is set to be computed at "
             << f.schedule().compute_level().func << " "
             << f.schedule().compute_level().var << endl;
        assert(false);
    }

    // new compute at level
    string compute_level_str = "compute_at(" + external.name() +
        ", Var(\"" + looplevel.name() + "\"))";

    // update the store and compute level of the final result
    Func(f).compute_at(external, looplevel);
    rF.pure_schedule.push_back(compute_level_str);

    // set the producer of this function to be computed at the same looplevel
    if (!rF.producer_func.empty()) {
        RecFilterFunc& producer = internal_function(rF.producer_func);
        if (!producer.func.schedule().compute_level().is_inline() ||
            !producer.func.schedule().store_level().is_inline())
        {
            Func(producer.func).compute_at(external, looplevel);
            producer.pure_schedule.push_back(compute_level_str);
        }

        // find all Functions whose consumer is this the above producer and set them
        // to be computed inside the external function. This is because the Func
        // computing the final result is now being computed inside some loop of an
        // external function. So all upstream functions should be same, or else they
        // will trigger a write to global memory
        map<string, RecFilterFunc>::iterator fit;
        for (fit=contents.ptr->func.begin(); fit!=contents.ptr->func.end(); fit++) {
            RecFilterFunc& rG = fit->second;
            if (rG.consumer_func == producer.func.name()) {
                rG.external_consumer_func = external;
                rG.external_consumer_var  = looplevel;
            }
        }
    }
}

// -----------------------------------------------------------------------------

void RecFilter::cpu_auto_schedule(int vector_width) {
    if (contents.ptr->tiled) {
        cpu_auto_intra_schedule(vector_width);
        cpu_auto_inter_schedule(vector_width);
    } else {
        cpu_auto_full_schedule(vector_width);
    }
}

void RecFilter::cpu_auto_full_schedule(int vector_width) {
    if (contents.ptr->tiled) {
        cerr << "Filter is tiled, use RecFilter::cpu_auto_intra_schedule() "
             << "and RecFilter::cpu_auto_inter_schedule()\n" << endl;
        assert(false);
    }

    // scan dimension can be unrolled
    // inner most dimension must be vectorized
    // all full dimensions must be parallelized

    full_schedule().compute_globally()
        .reorder(full_scan(), full())
        .vectorize(full(0), vector_width)
        .parallel(full());                  // TODO: only parallelize outermost, not all
}

void RecFilter::cpu_auto_intra_schedule(int vector_width) {
    if (!contents.ptr->tiled) {
        cerr << "Filter is not tiled, use RecFilter::cpu_auto_full_schedule()\n" << endl;
        assert(false);
    }

    RecFilterSchedule R = intra_schedule(0);

    if (R.empty()) {
        return;
    }

    int max_tile = 0;
    for (int i=0; i<contents.ptr->filter_info.size(); i++) {
        if (contents.ptr->filter_info[i].tile_width<contents.ptr->filter_info[i].image_width) {
            max_tile = std::max(max_tile, contents.ptr->filter_info[i].tile_width);
        }
    }

    R.compute_locally()
        .split(full(0), max_tile, inner(), outer())  // convert upto 3 full dimensions
        .split(full(0), max_tile, inner(), outer())  // into tiles
        .split(full(0), max_tile, inner(), outer())
        .reorder({inner_scan(), inner(), outer()})   // scan dimension is innermost
        .vectorize(inner(0), vector_width)           // vectorize innermost non-scan dimension
        .parallel(outer());                          // TODO: only parallelize outermost
}

void RecFilter::cpu_auto_inter_schedule(int vector_width) {
    if (!contents.ptr->tiled) {
        cerr << "Filter is not tiled, use RecFilter::cpu_auto_full_schedule()\n" << endl;
        assert(false);
    }

    RecFilterSchedule R = inter_schedule();

    if (R.empty()) {
        return;
    }

    int max_tile = 0;
    for (int i=0; i<contents.ptr->filter_info.size(); i++) {
        if (contents.ptr->filter_info[i].tile_width<contents.ptr->filter_info[i].image_width) {
            max_tile = std::max(max_tile, contents.ptr->filter_info[i].tile_width);
        }
    }

    R.compute_globally()
        .split(full(0), max_tile, inner(), outer())         // convert upto 3 full dimensions
        .split(full(0), max_tile, inner(), outer())         // into tiles
        .split(full(0), max_tile, inner(), outer())
        .reorder({outer_scan(), tail(), inner(), outer()})  // scan dimension is innermost
        .vectorize(inner(0), vector_width)                  // vectorize innermost non-scan dimension
        .parallel(outer());                                 // TODO: only parallelize outermost
}

// -----------------------------------------------------------------------------

void RecFilter::gpu_auto_schedule(int max_threads, int tile_width) {
    if (contents.ptr->tiled) {
        gpu_auto_intra_schedule(1,max_threads);
        gpu_auto_intra_schedule(2,max_threads);
        gpu_auto_inter_schedule(max_threads);
    } else {
        gpu_auto_full_schedule(max_threads, tile_width);
    }
}

void RecFilter::gpu_auto_full_schedule(int max_threads, int tile_width) {
    if (contents.ptr->tiled) {
        cerr << "Filter is tiled, use RecFilter::gpu_auto_intra_schedule() "
             << "and RecFilter::gpu_auto_inter_schedule()\n" << endl;
        assert(false);
    }

    RecFilterSchedule R = full_schedule();

    R.compute_globally()
        .unroll(full_scan())
        .split(full(0), tile_width, inner(), outer())  // convert upto three full
        .split(full(0), tile_width, inner(), outer())  // dimensions into tiles
        .split(full(0), tile_width, inner(), outer());

    VarTag tx = inner(0);
    VarTag ty = inner(1);
    VarTag tz = inner(2);

    if (R.contains_vars_with_tag(ty) && tile_width*tile_width>max_threads) {
        int factor = tile_width*tile_width/max_threads;
        R.split(ty, factor).unroll(ty.split_var());
    }

    R.reorder({full_scan(), ty.split_var(), tz, tx, ty, outer()})
        .gpu_threads(tx, ty)
        .gpu_blocks(outer(0), outer(1), outer(2));
}

void RecFilter::gpu_auto_inter_schedule(int max_threads) {
    if (!contents.ptr->tiled) {
        cerr << "Filter is not tiled, use RecFilter::gpu_auto_full_schedule()\n" << endl;
        assert(false);
    }

    if (contents.ptr->filter_info.size()>AUTO_SCHEDULE_MAX_DIMENSIONS) {
        cerr << "Auto schedules are not supported for more than " << AUTO_SCHEDULE_MAX_DIMENSIONS << " filter dimensions" << endl;
        assert(false);
    }

    RecFilterSchedule R = inter_schedule();

    if (R.empty()) {
        return;
    }

    // max tile is the maximum number of threads that will be launched
    // by specifying either of tx, ty, or tz as parallel
    int max_tile = 0;
    for (int i=0; i<contents.ptr->filter_info.size(); i++) {
        if (contents.ptr->filter_info[i].tile_width<contents.ptr->filter_info[i].image_width) {
            max_tile = std::max(max_tile, contents.ptr->filter_info[i].tile_width);
        }
    }

    // store inner dimensions innermost because threads are operating
    // on these dimensions - memory coalescing
    R.compute_globally()
     .reorder_storage({full(), inner(), tail(), outer()});

    R.split(full(0), max_tile, inner(), outer())  // upto two full dimensions
     .split(full(0), max_tile, inner(), outer());

    VarTag sc = outer_scan();       // exactly one scan dimension

    VarTag tx = inner(0);           // at most 3 inner dimensions
    VarTag ty = inner(1);           // because 1 inner dimension is a tail

    VarTag bx = outer(0);           // at most 2 outer dimensions
    VarTag by = outer(1);           // as 1 outer dim is being scanned

    // split an outer dimensions to generate extra threads to fill CUDA warps;
    // any parallelism ty for 3D filters is not exploited
    int factor = max_threads/max_tile;

    R.unroll(sc)
        .split(bx, factor)
        .reorder({sc, tail(), ty, bx.split_var(), tx, bx, by})
        .gpu_threads(tx, bx.split_var())
        .gpu_blocks (bx, by);
}

void RecFilter::gpu_auto_intra_schedule(int id, int max_threads) {
    if (!contents.ptr->tiled) {
        cerr << "Filter is not tiled, use RecFilter::gpu_auto_full_schedule()\n" << endl;
        assert(false);
    }

    if (contents.ptr->filter_info.size()>AUTO_SCHEDULE_MAX_DIMENSIONS) {
        cerr << "Auto schedules are not supported for more than " << AUTO_SCHEDULE_MAX_DIMENSIONS << " filter dimensions" << endl;
        assert(false);
    }

    RecFilterSchedule R = intra_schedule(id);

    if (R.empty()) {
        return;
    }

    // max tile is the maximum number of threads that will be launched
    // by specifying either of tx, ty, or tz as parallel
    int max_tile  = 0;
    int max_order = 0;
    int num_scans = 0;
    for (int i=0; i<contents.ptr->filter_info.size(); i++) {
        if (contents.ptr->filter_info[i].tile_width<contents.ptr->filter_info[i].image_width) {
            max_tile = std::max(max_tile, contents.ptr->filter_info[i].tile_width);
        }
        max_order  = std::max(max_order, contents.ptr->filter_info[i].filter_order);
        num_scans += contents.ptr->filter_info[i].num_scans;
    }

    R.split(full(0), max_tile, inner(), outer())  // upto two full dimensions
     .split(full(0), max_tile, inner(), outer());

    VarTag sc = inner_scan();       // exactly one scan dimension

    VarTag tx = inner(0);           // at most 3 inner dimensions
    VarTag ty = inner(1);
    VarTag tz = inner(2);

    VarTag bx = outer(0);           // at most 3 outer dimensions
    VarTag by = outer(1);
    VarTag bz = outer(2);

    cerr << print_schedule() << "HAHAHAHA" << endl;

    R.compute_locally().storage_layout(INVALID, outer());
    cerr << print_schedule() << "HOHOHOHOH" << endl;


    switch (id) {
        case 1:
            // for nD intra tile terms: each inner var is of size max_tile if ty is not
            // empty then this will give too many threads, reduce by splitting ty; any
            // parallelism in tz in case of 3D filters is not exploited
            if (R.contains_vars_with_tag(ty) && max_tile*max_tile>max_threads) {
                int factor = max_tile*max_tile/max_threads;
                R.split(ty, factor).unroll(ty.split_var());
            }
            R.unroll(sc)
                .reorder({sc, ty.split_var(), tz, tx, ty, outer()})
                .gpu_threads(tx, ty)
                .gpu_blocks (bx, by, bz);
            break;


        case 2:
            // for intra tile terms that compute cross dimensional residuals, generate more
            // threads by splitting an outer dimension and using it as threads; any parallelism
            // in tz in case of 3D filters is not exploited
            R.unroll(sc)
                .split(bx, max_tile/(num_scans*max_order))
                .reorder({tx, ty, tz, sc, tail(), bx.split_var(), outer()})
                .fuse(tail(), tx)
                .gpu_threads(tail(), bx.split_var())
                .gpu_blocks (bx, by, bz);
            break;

        default: break;
    }
}

// -----------------------------------------------------------------------------

VarTag RecFilter::full          (int i) { return VarTag(FULL,  i);     }
VarTag RecFilter::inner         (int i) { return VarTag(INNER, i);     }
VarTag RecFilter::outer         (int i) { return VarTag(OUTER, i);     }
VarTag RecFilter::tail          (void)  { return VarTag(TAIL);         }
VarTag RecFilter::full_scan     (void)  { return VarTag(FULL|SCAN);    }
VarTag RecFilter::inner_scan    (void)  { return VarTag(INNER|SCAN);   }
VarTag RecFilter::outer_scan    (void)  { return VarTag(OUTER|SCAN);   }
VarTag RecFilter::inner_channels(void)  { return VarTag(INNER|CHANNEL);}
VarTag RecFilter::outer_channels(void)  { return VarTag(OUTER|CHANNEL);}

// -----------------------------------------------------------------------------

Func RecFilter::as_func(void) {
    if (contents.ptr->func.empty()) {
        cerr << "Filter " << contents.ptr->name << " not defined" << endl;
        assert(false);
    }
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

void RecFilter::compile_jit(string filename) {
    if (!contents.ptr->finalized) {
        finalize();
    }

    Func F = as_func();
    if (!filename.empty()) {
        F.compile_to_lowered_stmt(filename, HTML, contents.ptr->target);
    }
    F.compile_jit(contents.ptr->target);

    contents.ptr->compiled = true;
}

Realization RecFilter::create_realization(void) {
    // check if any of the functions have a schedule
    // true if all functions have default inline schedule
    bool no_schedule_applied = true;
    map<string,RecFilterFunc>::iterator fit;
    for (fit=contents.ptr->func.begin(); fit!=contents.ptr->func.end(); fit++) {
        Function f = fit->second.func;
        no_schedule_applied &= f.schedule().compute_level().is_inline();
    }

    // apply a default schedule to compute everything
    // in global memory if no schedule has been used
    if (no_schedule_applied) {
        if (contents.ptr->tiled) {
            inter_schedule().compute_globally();
            intra_schedule().compute_globally();
        } else {
            full_schedule().compute_globally();
        }
        cerr << "Warning: Applied default schedule to filter "
            << contents.ptr->name << endl;
    }

    // recompile the filter
    contents.ptr->compiled = false;
    compile_jit();

    // upload all buffers to device if computed on GPU
    Func F(internal_function(contents.ptr->name).func);
    if (contents.ptr->target.has_gpu_feature()) {
        map<string,Buffer> buff = extract_buffer_calls(F);
        for (map<string,Buffer>::iterator b=buff.begin(); b!=buff.end(); b++) {
            b->second.copy_to_dev();
            //b->second.copy_to_device();
        }
    }

    // allocate the buffer
    vector<int> buffer_size;
    for (int i=0; i<contents.ptr->filter_info.size(); i++) {
        buffer_size.push_back(contents.ptr->filter_info[i].image_width);
    }

    // create a realization object
    vector<Buffer> buffers;
    for (int i=0; i<F.outputs(); i++) {
        buffers.push_back(Buffer(contents.ptr->type, buffer_size));
    }

    return Realization(buffers);
}

Realization RecFilter::realize(void) {
    Func F(internal_function(contents.ptr->name).func);
    Realization R = create_realization();
    F.realize(R, contents.ptr->target);
    return R;
}

float RecFilter::profile(int iterations) {
    Func F(internal_function(contents.ptr->name).func);
    Realization R = create_realization();

    double total_time = 0;
    unsigned long time_start, time_end;

    if (contents.ptr->target.has_gpu_feature()) {
        F.realize(R, contents.ptr->target); // warmup run

        time_start = millisecond_timer();
        for (int i=0; i<iterations; i++) {
            F.realize(R, contents.ptr->target);
        }
        time_end = millisecond_timer();
    } else {
        time_start = millisecond_timer();
        for (int i=0; i<iterations; i++) {
            F.realize(R, contents.ptr->target);
        }
        time_end = millisecond_timer();
    }
    total_time = (time_end-time_start);

    return total_time/iterations;
}

Target RecFilter::target(void) {
    return contents.ptr->target;
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
        map<int,vector<string> >::iterator sit;

        // dump the pure def schedule
        if (!f->second.pure_schedule.empty()) {
            vector<string> str = f->second.pure_schedule;
            s << f->second.func.name();
            // first print any compute at rules
            bool compute_def_found = false;
            for (int i=0; !compute_def_found && i<str.size(); i++) {
                if (str[i].find("compute_root")!= string::npos ||
                        str[i].find("compute_at")  != string::npos) {
                    s << "." << str[i];
                    str.erase(str.begin()+i);
                    compute_def_found = true;
                }
            }
            for (int i=0; i<str.size(); i++) {
                if (str.size()<2) {
                    s << "." << str[i];
                } else {
                    s << "\n    ." << str[i];
                }
            }
            s << ";\n";
        }

        // dump the update def schedules
        for (sit=f->second.update_schedule.begin(); sit!=f->second.update_schedule.end(); sit++) {
            int def = sit->first;
            vector<string> str = sit->second;
            if (!str.empty()) {
                s << f->second.func.name() << ".update(" << def << ")";
            }
            for (int i=0; i<str.size(); i++) {
                s << "\n    ." << str[i];
            }
            s << ";\n";
        }
    }
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
