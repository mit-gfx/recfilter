#include "split.h"

#define SPLIT_HELPER_NAME '-'

using namespace Halide;
using namespace Halide::Internal;

using std::string;
using std::cerr;
using std::endl;
using std::vector;
using std::queue;
using std::map;

// -----------------------------------------------------------------------------

struct SplitInfo {
    bool scan_causal;
    int scan_stage;
    int filter_order;
    int filter_dim;
    int reduction_def;
    Var var;
    Var inner_var;
    Var outer_var;
    RDom rdom;
    RDom split_rdom;
    RDom outer_rdom;
    RDom inner_rdom;
    Expr tile_width;
    Expr image_width;
    Func filter_weights;
    vector<Function> func_list;
};

// -----------------------------------------------------------------------------

static Func tail_weights(Func filter_weights, Expr tile_width, int filter_dim, int filter_order) {
    assert(filter_weights.defined());

    Type type = filter_weights.values()[0].type();
    Expr zero = Cast::make(type, 0);
    Expr one  = Cast::make(type, 1);

    Var i("i"), j("j"), r("r");
    RDom c(0, filter_order, "c");
    RDom m(0, filter_order, 0, filter_order, 0, filter_order, 1, tile_width-1, "m");

    Func A("A");
    Func Ar("Ar");

    // filter matrix to power r= A^r_F [Nehab et al 2011, appendix A]
    A(i,j) = select(j==filter_order-1,
            filter_weights(filter_dim, filter_order-1-i),
            select(i==j+1, one, zero));


    Ar(i, j, r) = select(r==0, A(i,j), 0);
    Ar(m.y, m.z, m.w) += Ar(m.y, m.x, m.w-1) * A(m.x, m.z);

    Ar.compute_root();
    Ar.bound(i,0,filter_order).bound(j,0,filter_order).bound(r,0,tile_width);

    // last row of matrix
    Func W("Weight_NO_REVEAL_" + int_to_string(filter_dim));
    W(r,j) = Ar(filter_order-1-j, filter_order-1, r);

    W.compute_root();
    W.bound(r,0,tile_width).bound(j,0,filter_order);

    return W;
}

// -----------------------------------------------------------------------------

static bool check_causal_scan(Function f, RVar rx, size_t reduction_def, size_t dimension) {
    assert(reduction_def < f.reductions().size());

    ReductionDefinition reduction = f.reductions()[reduction_def];
    Expr               arg       = reduction.args[dimension];

    // check if reduction arg increases on increasing the RVar
    // causal scan if yes, else anticausal
    Expr a = substitute(rx.name(), 0, arg);
    Expr b = substitute(rx.name(), 1, arg);
    Expr c = simplify(a<b);

    if (equal(c, Cast::make(c.type(),1))) {
        return true;
    } else if (equal(c, Cast::make(c.type(),0))) {
        return false;
    } else {
        cerr << "Could not deduce causal or anticausal scan for reduction definition "
            << reduction_def << " of " << f.name() << endl;
        assert(false);
    }
}

// -----------------------------------------------------------------------------

static void check_split_feasible(
        Func& func,
        vector<int>  dimension,
        vector<Var>  var,
        vector<Var>  inner_var,
        vector<Var>  outer_var,
        vector<RDom> rdom,
        vector<RDom> inner_rdom,
        vector<int>  order)
{
    size_t num_splits = var.size();

    assert(num_splits == dimension.size()  && "Each split must have a mapped function dimension");
    assert(num_splits == rdom.size()       && "Each split must have a mapped RDom");
    assert(num_splits == inner_var.size()  && "Each split must have a mapped inner Var");
    assert(num_splits == outer_var.size()  && "Each split must have a mapped outer Var");
    assert(num_splits == inner_rdom.size() && "Each split must have a mapped inner RDom");
    assert(num_splits == order.size()      && "Each split must have a mapped filter order");

    Function F = func.function();

    assert(F.has_pure_definition() &&  "Func to be split must be defined");
    assert(!F.is_pure() && "Use Halide::Func::split for pure Funcs");


    // check variables
    for (size_t k=0; k<num_splits; k++) {
        int dim = dimension[k];

        // RDom to be split must be 1D, each reduction definition should be 1D reduction
        if (rdom[k].dimensions() != 1) {
            cerr << "RDom to split must be 1D, each reduction "
                << "definition should use a unique be 1D RDom";
            assert(false);
        }

        // given inner RDom must be 1D, intra tile scans are 1D as full scan is 1D
        if (inner_rdom[k].dimensions() != 1) {
            cerr << "Inner RDom must be 1D, as splitting a 1D reduction"
                << "definition produces 1D intra-tile reductions";
            assert(false);
        }

        // variable at given dimension must match the one to be split
        if (F.args()[dim] != var[k].name()) {
            cerr << "Variable at dimension " << dim << " must match the one "
                << "specified for splitting"   << endl;
            assert(false);
        }

        // RDom to be split must appear in exactly one reduction definition
        size_t num_reductions_involving_rdom = 0;
        for (size_t i=0; i<F.reductions().size(); i++) {
            string rdom_name = rdom[k].x.name();
            bool reduction_involves_rdom = false;
            for (size_t j=0; j<F.reductions()[i].args.size(); j++) {
                reduction_involves_rdom |= expr_depends_on_var(F.reductions()[i].args[j], rdom_name);
            }
            for (size_t j=0; j<F.reductions()[i].values.size(); j++) {
                reduction_involves_rdom |= expr_depends_on_var(F.reductions()[i].values[j], rdom_name);
            }
            if (reduction_involves_rdom) {
                num_reductions_involving_rdom++;
            }
        }
        if (num_reductions_involving_rdom < 1) {
            cerr << "RDom to be split must appear in one reduction definition, found in none";
            assert(false);
        }
        if (num_reductions_involving_rdom > 1) {
            cerr << "RDom to be split must appear in only one reduction definition, found in multiple";
            assert(false);
        }

        // RDom to be split must not appear at any dimension other than the one specified
        for (size_t i=0; i<F.reductions().size(); i++) {
            string rdom_name = rdom[k].x.name();
            bool reduction_involves_rdom = false;
            for (size_t j=0; j<F.reductions()[i].args.size(); j++) {
                if (j!=dim && expr_depends_on_var(F.reductions()[i].args[j], rdom_name)) {
                    cerr << "RDom to be split must appear only at the specified dimensino, found in others";
                    assert(false);
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------

static bool check_for_pure_split(Function F, SplitInfo split_info) {
    // check if the split is pure, i.e.
    // - if the Function is pure, or
    // - if the RDom to be split does not appear in any reduction definition

    if (F.is_pure()) {
        return true;
    }

    bool rdom_exists_in_reduction_def = false;
    for (size_t i=0; i<F.reductions().size(); i++) {
        string inner_rdom_name = split_info.split_rdom.x.name();
        string outer_rdom_name = split_info.split_rdom.y.name();

        for (size_t j=0; j<F.reductions()[i].args.size(); j++) {
            bool a = expr_depends_on_var(F.reductions()[i].args[j], inner_rdom_name);
            rdom_exists_in_reduction_def |= a;
        }
        for (size_t j=0; j<F.reductions()[i].values.size(); j++) {
            bool a = expr_depends_on_var(F.reductions()[i].values[j], inner_rdom_name);
            rdom_exists_in_reduction_def |= a;
        }
        for (size_t j=0; j<F.reductions()[i].args.size(); j++) {
            bool a = expr_depends_on_var(F.reductions()[i].args[j], outer_rdom_name);
            rdom_exists_in_reduction_def |= a;
        }
        for (size_t j=0; j<F.reductions()[i].values.size(); j++) {
            bool a = expr_depends_on_var(F.reductions()[i].values[j], outer_rdom_name);
            rdom_exists_in_reduction_def |= a;
        }
    }
    return !rdom_exists_in_reduction_def;
}


static Function split_function_dimensions(Function F, vector<SplitInfo> split_info) {
    Function Fsplit(F.name() + SPLIT_HELPER_NAME);

    // set of dimensions to split
    map<int,Expr>    dim_to_tile_width;
    map<string,Var>  var_to_inner_var;
    map<string,Var>  var_to_outer_var;
    map<string,Expr> var_to_image_width;
    map<string,Expr> var_to_inner_expr;
    map<string,Expr> var_to_outer_expr;
    map<string,Expr> var_to_reduction_extent;

    // setup mappings between variables, RDoms and their inner/outer variants
    for (size_t i=0; i<split_info.size(); i++) {
        if (dim_to_tile_width.find(split_info[i].filter_dim) != dim_to_tile_width.end()) {
            if (!equal(dim_to_tile_width[split_info[i].filter_dim], split_info[i].tile_width)) {
                cerr << "Different tile widths specified for splitting same dimension" << endl;
                assert(false);
            }
        } else {
            dim_to_tile_width[split_info[i].filter_dim] = split_info[i].tile_width;
        }

        var_to_inner_var  [split_info[i].var.name()]   = split_info[i].inner_var;
        var_to_outer_var  [split_info[i].var.name()]   = split_info[i].outer_var;
        var_to_inner_expr [split_info[i].var.name()]   = split_info[i].inner_var;
        var_to_outer_expr [split_info[i].var.name()]   = split_info[i].outer_var;
        var_to_image_width[split_info[i].var.name()]   = split_info[i].image_width;
        var_to_inner_expr [split_info[i].rdom.x.name()]= split_info[i].split_rdom.x;
        var_to_outer_expr [split_info[i].rdom.x.name()]= split_info[i].split_rdom.y;
    }

    Var xs(SCAN_STAGE_ARG);

    vector<string> pure_args = F.args();
    vector<Expr>   pure_values = F.values();
    vector<ReductionDefinition> reductions = F.reductions();

    // add the scan stage arg to pure args
    // set pure values to zero for all scan stages other than 0
    // add scan stage value as arg to reductions, so that each
    // reduction computes at its own stage
    pure_args.push_back(xs.name());
    for (size_t i=0; i<pure_values.size(); i++) {
        pure_values[i] = select(xs==0, pure_values[i], 0);
    }
    for (size_t i=0; i<reductions.size(); i++) {
        int scan_stage = -1;
        for (size_t j=0; j<split_info.size(); j++) {
            if (split_info[j].reduction_def == i) {
                scan_stage = split_info[j].scan_stage;
            }
        }
        if (scan_stage < 0) {
            cerr << "Split operation not defined for all scans" << endl;
            assert(false);
        }
        reductions[i].args.push_back(scan_stage);
    }

    // go to each function dimension and replace the Var with inner
    // and outer Var, replace RDom with inner and outer RDoms
    map<int,Expr>::iterator it = dim_to_tile_width.begin();
    map<int,Expr>::iterator end= dim_to_tile_width.end();
    for (; it != end; it++) {
        int dim = it->first;
        string x = F.args()[dim];
        Expr tile_width = it->second;

        // redefine Fsplit after splitting each dimension
        Fsplit.clear_all_definitions();

        int var_index = -1;
        for (size_t i=0; var_index<0 && i<pure_args.size(); i++) {
            if (x == pure_args[i]) {
                var_index = i;
            }
        }
        assert(var_index >= 0);


        // pure definition
        {
            if (var_to_inner_var.find(x) == var_to_inner_var.end()) {
                cerr << "No inner variable specified for variable " << x << endl;
                assert(false);
            }
            if (var_to_outer_var.find(x) == var_to_outer_var.end()) {
                cerr << "No outer variable specified for variable " << x << endl;
                assert(false);
            }

            Var xi = var_to_inner_var[x];
            Var xo = var_to_outer_var[x];

            // pure definition args: remove x and replace by xi, xo
            pure_args[var_index] = xi.name();
            pure_args.insert(pure_args.begin()+var_index+1, xo.name());

            // pure definition values: replace x by tile*xo+xi in RHS values
            for (size_t i=0; i<pure_values.size(); i++) {
                pure_values[i] = substitute(x, tile_width*xo+xi, pure_values[i]);
            }

            Fsplit.define(pure_args, pure_values);
        }

        // change the reduction definitions that involve rx
        for (size_t i=0; i<reductions.size(); i++) {
            vector<string> expr_vars = extract_vars_or_rvars_in_expr(
                    F.reductions()[i].args[dim]);

            vector<string> expr_params = extract_params_in_expr(
                    var_to_image_width[x]);

            if (expr_vars.size() != 1) {
                cerr << "Only one Var or RVar can be referenced in reduction definition arg "
                    << F.reductions()[i].args[dim] << endl;
                assert(false);
            }

            if (expr_params.size() != 1) {
                cerr << "Only one Param must be referenced in reduction extent "
                    << var_to_image_width[x] << endl;
                assert(false);
            }

            string rx = expr_vars[0];

            if (var_to_inner_expr.find(rx) == var_to_inner_expr.end()) {
                cerr << "No inner expr specified for variable " << rx << endl;
                assert(false);
            }
            if (var_to_outer_expr.find(rx) == var_to_outer_expr.end()) {
                cerr << "No outer expr specified for variable " << rx << endl;
                assert(false);
            }

            Expr rxi = var_to_inner_expr[rx];
            Expr rxo = var_to_outer_expr[rx];
            string image_width = expr_params[0];

            // reduction definition args: replace rx by rxi,rxo and image_width by tile_width (for anticausal filter)
            reductions[i].args[var_index] = substitute(rx, rxi, reductions[i].args[var_index]);
            reductions[i].args[var_index] = substitute(image_width, tile_width, reductions[i].args[var_index]);
            reductions[i].args.insert(reductions[i].args.begin()+var_index+1, rxo);

            // change reduction definition RHS values
            for (size_t j=0; j<reductions[i].values.size(); j++) {
                Expr value = reductions[i].values[j];

                // change calls to original Func by split Func
                // add rxo as calling arg and replace rx by rxi to this function
                value = substitute_func_call(F.name(), Fsplit, value);
                value = insert_arg_to_func_call(Fsplit.name(), var_index+1, rxo, value);
                value = substitute(rx, rxi, value);
                value = substitute(image_width, tile_width, value);
                reductions[i].values[j] = value;
            }

            Fsplit.define_reduction(reductions[i].args, reductions[i].values);
        }
    }

    //vector<string> pure_args = Fsplit.args();
    //vector<Expr>   pure_values = Fsplit.values();
    //vector<ReductionDefinition> reductions = Fsplit.reductions();

    //for (size_t i=0; i<reductions(); i++) {
    //}



    return Fsplit;
}

// -----------------------------------------------------------------------------

static Function create_intra_tile_term(Function F, SplitInfo split_info, string func_name) {
    Function function(func_name);

    Var  xo   = split_info.outer_var;
    RDom r2D  = split_info.split_rdom; // RDom(inner RVar, outer RVar)
    RDom rxi  = split_info.inner_rdom; // RDom(inner RVar)

    // pure definition remains same as the split function
    function.define(F.args(), F.values());

    // reduction definitions: remove the inter-tile reduction
    // i.e. replace outer RDom placeholder with outer_var
    // and replace inner_rdom with a 1D version of the inner_rdom
    for (size_t i=0; i<F.reductions().size(); i++) {
        vector<Expr> args = F.reductions()[i].args;
        vector<Expr> values = F.reductions()[i].values;

        // replace r2D.outer_rvar by xo in args
        // replace r2D.inner_rvar by rxi in args
        for (size_t j=0; j<args.size(); j++) {
            args[j] = substitute(r2D.x.name(), rxi, args[j]);
            args[j] = substitute(r2D.y.name(), xo, args[j]);
        }

        // change calls to original Func by intra tile Func
        // replace outer RDom by xo in args
        for (size_t j=0; j<values.size(); j++) {
            values[j] = substitute_func_call(F.name(), function, values[j]);
            values[j] = substitute(r2D.x.name(), rxi, values[j]);
            values[j] = substitute(r2D.y.name(),  xo, values[j]);
        }

        function.define_reduction(args, values);
    }

    return function;
}

static Function create_intra_tail_term(Function F_intra, SplitInfo split_info, string func_name) {
    Function function(func_name);

    Expr tile = split_info.tile_width;
    Var  xi   = split_info.inner_var;

    vector<string> args;
    vector<Expr> call_args;
    vector<Expr> values;

    // pure args same as F_intra except for scan stage arg
    // args for calling F_intra
    // - xi replaced by tail indices
    // - scan stage arg set to actual scan stage
    for (size_t i=0; i<F_intra.args().size(); i++) {
        if (SCAN_STAGE_ARG == F_intra.args()[i]) {
            call_args.push_back(split_info.scan_stage);
        }
        else if (xi.name() == F_intra.args()[i]) {
            args.push_back(xi.name());
            call_args.push_back(tile-xi-1);
        }
        else {
            args.push_back(F_intra.args()[i]);
            call_args.push_back(Var(F_intra.args()[i]));
        }
    }

    // call intra tile term at elements
    for (int j=0; j<F_intra.outputs(); j++) {
        Expr value = Call::make(F_intra, call_args, j);
        values.push_back(value);
    }

    function.define(args, values);

    return function;
}

static Function create_complete_tail_term(Function F_tail, SplitInfo split_info, string func_name) {
    Function function(func_name);

    Var  x    = split_info.var;
    Var  xi   = split_info.inner_var;
    Var  xo   = split_info.outer_var;
    int  dim  = split_info.filter_dim;
    int  order= split_info.filter_order;
    Expr tile = split_info.tile_width;
    RDom rxo  = split_info.outer_rdom;

    Func weight = split_info.filter_weights;

    // pure definition
    {
        // pure definition simply initializes with the intra tile tail
        vector<Expr> call_args;
        vector<Expr> values;
        for (size_t i=0; i<F_tail.args().size(); i++) {
            call_args.push_back(Var(F_tail.args()[i]));
        }
        for (size_t i=0; i<F_tail.outputs(); i++) {
            values.push_back(Call::make(F_tail, call_args, i));
        }
        function.define(F_tail.args(), values);
    }

    // reduction definition
    {
        vector<Expr> args;
        vector<Expr> values;
        vector<Expr> call_args_curr_tile;
        vector<Expr> call_args_prev_tile;

        for (size_t i=0; i<F_tail.args().size(); i++) {
            string arg = F_tail.args()[i];
            if (arg == xo.name()) {
                // replace xo by rxo.z or rxo.z-1 as tile idx,
                args.push_back(rxo.z);
                call_args_curr_tile.push_back(rxo.z);
                call_args_prev_tile.push_back(max(rxo.z-1,0));
            } else if (arg == xi.name()) {
                // replace xi by rxo.y as tail element index in args and current tile term
                // replace xi by rxo.x as tail element in prev tile term because each rxo.y
                // is computed from all prev tile tail elements
                args.push_back(rxo.y);
                call_args_curr_tile.push_back(rxo.y);
                call_args_prev_tile.push_back(rxo.x);
            } else {
                args.push_back(Var(arg));
                call_args_curr_tile.push_back(Var(arg));
                call_args_prev_tile.push_back(Var(arg));
            }
        }
        for (size_t i=0; i<F_tail.outputs(); i++) {
            // multiply each tail element with its weight before adding
            values.push_back(
                    Call::make(function, call_args_curr_tile, i) +
                    select(rxo.z>0, weight(tile-rxo.y-1,rxo.x) *
                        Call::make(function, call_args_prev_tile, i), 0));
        }

        function.define_reduction(args, values);
    }

    return function;
}

// -----------------------------------------------------------------------------

static void create_recursive_split(Function& F, SplitInfo& split_info) {
    Expr tile = split_info.tile_width;
    int  dim  = split_info.filter_dim;
    int  order= split_info.filter_order;
    Var  x    = split_info.var;
    Var  xi   = split_info.inner_var;
    Var  xo   = split_info.outer_var;

    Func weight = split_info.filter_weights;

    string s1 = F.name() + DELIMITER + INTRA_TILE_RESULT    + "_" + x.name();
    string s2 = F.name() + DELIMITER + INTRA_TILE_TAIL_TERM + "_" + x.name();
    string s3 = F.name() + DELIMITER + INTER_TILE_TAIL_SUM  + "_" + x.name();
    string s4 = F.name() + DELIMITER + INTER_TILE_DEPENDENCY+ "_" + x.name();

    Function F_intra = create_intra_tile_term   (F,       split_info, s1.c_str());
    Function F_tail  = create_intra_tail_term   (F_intra, split_info, s2.c_str());
    Function F_ctail = create_complete_tail_term(F_tail,  split_info, s3.c_str());

    Function F_inter_deps(s4.c_str());

    // tail dependencies to be added to the final term
    {
        vector<string> args = F_intra.args();
        vector<Expr> values(F.outputs());
        for (int k=0; k<order; k++) {
            vector<Expr> call_args;
            for (size_t i=0; i<F_ctail.args().size(); i++) {
                string arg = F_ctail.args()[i];
                if (xi.name() == arg)
                    call_args.push_back(k);
                else
                    call_args.push_back(Var(arg));
            }
            for (int i=0; i<F_ctail.outputs(); i++) {
                Expr val = weight(xi,k) * Call::make(F_ctail, call_args, i);
                if (k==0) values[i] = val;
                else      values[i]+= val;
            }
        }
        F_inter_deps.define(args, values);
    }

    // final term consisting of intra term and dependencies
    {
        vector<string> args = F.args();
        vector<Expr> values;
        vector<Expr> intra_call_args;
        vector<Expr> inter_call_args;
        for (size_t i=0; i<F_intra.args().size(); i++) {
            if (xo.name() == F_intra.args()[i]) {
                intra_call_args.push_back(xo);
                inter_call_args.push_back(max(xo-1,0));  // prev tile
            } else {
                intra_call_args.push_back(Var(F_intra.args()[i]));
                inter_call_args.push_back(Var(F_intra.args()[i]));
            }
        }
        for (int i=0; i<F.outputs(); i++) {
            Expr val = Call::make(F_intra, intra_call_args, i) +
                select(xo>0, Call::make(F_inter_deps, inter_call_args, i), 0);
            values.push_back(val);
        }

        // redefine original Func to point to calls to split Func
        F.clear_all_definitions();
        F.define(args, values);
    }

    // extra functions generated
    split_info.func_list.push_back(F_intra);
    split_info.func_list.push_back(F_tail);
    split_info.func_list.push_back(F_ctail);
    split_info.func_list.push_back(F_inter_deps);
}


// -----------------------------------------------------------------------------

void split(
        Func& func,
        vector<int>  dimension,
        vector<Var>  var,
        vector<Var>  inner_var,
        vector<Var>  outer_var,
        vector<RDom> rdom,
        vector<RDom> inner_rdom)
{
    Var i,j;
    Func filter_weights("default_filter_wt");    // default filter weight = 1
    filter_weights(i,j) = Cast::make(func.values()[0].type(), 1);

    vector<int> order(dimension.size(), 1);     // default first order

    split(func, filter_weights, dimension, var, inner_var, outer_var,
            rdom, inner_rdom, order);
}

void split(
        Func& func,
        vector<int>  dimension,
        vector<Var>  var,
        vector<Var>  inner_var,
        vector<Var>  outer_var,
        vector<RDom> rdom,
        vector<RDom> inner_rdom,
        vector<int>  order)
{
    Var i,j;
    Func filter_weights("default_filter_wt");    // default filter weight = 1
    filter_weights(i,j) = Cast::make(func.values()[0].type(), 1);

    split(func, filter_weights, dimension, var, inner_var, outer_var,
            rdom, inner_rdom, order);
}

void split(
        Func& func,
        Func filter_weights,
        vector<int>  dimension,
        vector<Var>  var,
        vector<Var>  inner_var,
        vector<Var>  outer_var,
        vector<RDom> rdom,
        vector<RDom> inner_rdom,
        vector<int>  order)
{

    check_split_feasible(func,dimension,var,inner_var,outer_var,rdom,inner_rdom,order);

    Function F = func.function();

    size_t num_splits = var.size();

    vector<Expr> tile;
    vector<RDom> split_rdom;
    vector<RDom> outer_rdom;
    vector<Expr> image_width;

    for (size_t i=0; i<num_splits; i++) {
        // individual tile boundaries
        Expr inner_rdom_extent = simplify(inner_rdom[i].x.extent()+1);
        tile.push_back(inner_rdom_extent);

        // extent of reduction along dimensions to be split
        assert(extract_params_in_expr(rdom[i].x.extent()).size()==1 &&
                "RDom extent must have a single image parameter");

        image_width.push_back(rdom[i].x.extent());

        // outer_rdom.x: over tail elems of prev tile to compute tail of current tile
        // outer_rdom.y: over all tail elements of current tile
        // outer_rdom.z: over all tiles
        outer_rdom.push_back(RDom(
                    0,order[i],
                    0,order[i],
                    1, image_width[i]/tile[i]-1,
                    "r"+var[i].name()+"o"));

        // 2D RDom which contains intra tile and inter tile RDom as two dimensions
        split_rdom.push_back(RDom(inner_rdom[i].x.min(), inner_rdom[i].x.extent(),
               outer_rdom[i].z.min(), outer_rdom[i].z.extent(), rdom[i].x.name()));
    }

    // list of structs with all the info about each split
    vector<SplitInfo> split_info;

    // reorder the splits such that the last scan is split first
    // loop over all reduction definitions in reverse order and split
    // them if needed
    for (int u=F.reductions().size()-1; u>=0; u--) {
        assert(F.reductions()[u].domain.defined() &&
                "Reduction definition has no reduction domain");

        // extract the RDom in the reduction definition and
        // compare with the reduction domain to be split
        for (size_t v=0; v<num_splits; v++) {
            if (F.reductions()[u].domain.same_as(rdom[v].domain())) {
                SplitInfo s;
                s.reduction_def  = u;
                s.scan_causal    = true;            // change it in next section
                s.scan_stage     = 0;               // default, change it in next section
                s.var            = var[v];
                s.rdom           = rdom[v];
                s.inner_var      = inner_var[v];
                s.outer_var      = outer_var[v];
                s.inner_rdom     = inner_rdom[v];
                s.outer_rdom     = outer_rdom[v];
                s.split_rdom     = split_rdom[v];
                s.tile_width     = tile[v];
                s.filter_order   = order[v];
                s.filter_dim     = dimension[v];
                s.image_width    = image_width[v];
                s.filter_weights = tail_weights(filter_weights, tile[v], dimension[v], order[v]);
                split_info.push_back(s);
            }
        }

        // go to all scans in each dimension, and assign scan stages
        // all causal scans in same dimension must be computed in different stages so that
        // they don't stomp on each other's tails; same for anticausal scans
        for (size_t d=0; d<F.args().size(); d++) {
            size_t num_causal_scans = 0;
            size_t num_anticausal_scans = 0;
            for (size_t i=0; i<split_info.size(); i++) {
                // scan in same dimension
                if (split_info[i].filter_dim == d) {
                    if (check_causal_scan(F, split_info[i].rdom.x, split_info[i].reduction_def, d)) {
                        split_info[i].scan_causal = true;
                        split_info[i].scan_stage = num_causal_scans;
                        num_causal_scans++;
                    } else {
                        split_info[i].scan_causal = false;
                        split_info[i].scan_stage = num_anticausal_scans;
                        num_anticausal_scans++;
                    }
                }
            }
        }
    }


    // create a function whose dimensions are split without
    // affecting computation, all split operations affect this function
    Function Fsplit = split_function_dimensions(F, split_info);

    // apply the individual split for each reduction variable - separate
    // the intra tile copmutation from inter tile computation
    queue<Function> funcs_to_split;
    funcs_to_split.push(Fsplit);

    int max_scan_stage = -1;

    for (size_t i=0; i<split_info.size() && !funcs_to_split.empty(); i++) {
        max_scan_stage = std::max(max_scan_stage, split_info[i].scan_stage);

        queue<Function> funcs_to_split_in_next_dimension;
        while (!funcs_to_split.empty()) {
            Function f = funcs_to_split.front();
            funcs_to_split.pop();

            if (!check_for_pure_split(f, split_info[i])) {
                create_recursive_split(f, split_info[i]);
            }

            // all newly created Func's have to be split for the next dimension
            for (size_t k=0; k<split_info[i].func_list.size(); k++) {
                funcs_to_split_in_next_dimension.push(split_info[i].func_list[k]);
            }

            // clear the list of sub Func's created
            split_info[i].func_list.clear();
        }

        // transfer all sub Func's to list to Func's that will be split in the next dimension
        funcs_to_split = funcs_to_split_in_next_dimension;
    }


    // modify the original function to index into the split function
    {
        vector<string> args = F.args();
        vector<Expr> call_args(Fsplit.args().size());
        vector<Expr> values(F.outputs());
        for (size_t i=0; i<call_args.size(); i++) {
            string arg = Fsplit.args()[i];
            if (arg == SCAN_STAGE_ARG) {
                call_args[i] = max_scan_stage;
            } else {
                call_args[i] = Var(arg);
            }
            for (size_t j=0; j<var.size(); j++) {
                if (inner_var[j].name() == arg) {
                    call_args[i] = var[j] % tile[j];
                }
                else if (outer_var[j].name() == arg) {
                    call_args[i] = var[j] / tile[j];
                }
            }
        }
        for (size_t i=0; i<F.outputs(); i++) {
            values[i] = Call::make(Fsplit, call_args, i);
        }
        F.clear_all_definitions();
        F.define(args, values);
    }

    func = Func(F);

    // inline the helper function F-split because it is no longer required
    inline_function(func, Fsplit.name());
}
