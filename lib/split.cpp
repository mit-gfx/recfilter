#include "split.h"

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
    int filter_order;
    int filter_dim;
    Var var;
    Var inner_var;
    Var outer_var;
    RDom rdom;
    RDom split_rdom;
    RDom inner_rdom;
    RDom outer_rdom;
    Expr tile_width;
    Func filter_weights;
    vector<Function> func_list;
};

// -----------------------------------------------------------------------------

static Func tail_weights(Func filter_weights, Expr tile_width, int filter_dim, int filter_order) {
    assert(filter_weights.defined());

    Var i("i"), j("j"), r("r");
    RDom c(0, filter_order, "c");
    RDom m(0, filter_order, 0, filter_order, 0, filter_order, 1, tile_width-1, "m");

    Func A("A");
    Func Ar("Ar");

    // filter matrix to power r= A^r_F [Nehab et al 2011, appendix A]
    A(i,j) = select(j==filter_order-1,
            filter_weights(filter_dim, filter_order-1-i),
            select(i==j+1, 1, 0));


    Ar(i, j, r) = select(r==0, A(i,j), 0);
    Ar(m.y, m.z, m.w) += Ar(m.y, m.x, m.w-1) * A(m.x, m.z);

    Ar.compute_root();
    Ar.bound(i,0,filter_order).bound(j,0,filter_order).bound(r,0,tile_width);

    // last row of matrix
    Func W("Weight_NO_REVEAL_" + int_to_string(filter_dim));
    W(r,j) = Ar(filter_order-1-j, filter_order-1, r);

    W.compute_root();
    W.bound(j,0,filter_order).bound(r,0,tile_width);

    Image<float> img = W.realize(4,filter_order);
    cerr << "W" << endl << img << endl;

    return W;
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
        // replace rxo by xo in args
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

    vector<Expr> call_args;
    vector<Expr> values;

    // args for calling F_intra, same as its args with xi replaced by tail indices
    for (size_t i=0; i<F_intra.args().size(); i++) {
        if (xi.name() == F_intra.args()[i]) {
            call_args.push_back(tile-xi-1);
        } else {
            call_args.push_back(Var(F_intra.args()[i]));
        }
    }

    // call intra tile term at elements
    for (int j=0; j<F_intra.outputs(); j++) {
        Expr value = Call::make(F_intra, call_args, j);
        values.push_back(value);
    }

    function.define(F_intra.args(), values);

    return function;
}

static Function create_complete_tail_term(Function F_tail, SplitInfo split_info, string func_name) {
    Function function(func_name);

    Expr tile = split_info.tile_width;
    Var  xi   = split_info.inner_var;
    Var  xo   = split_info.outer_var;
    RDom rxo  = split_info.outer_rdom;
    int  dim  = split_info.filter_dim;
    int  order= split_info.filter_order;

    Func weight = split_info.filter_weights;

    // pure definition
    {
        // pure definition simply initializes with the intra tile tail
        vector<Expr>   call_args;
        vector<Expr>   values;
        for (size_t j=0; j<F_tail.args().size(); j++) {
            call_args.push_back(Var(F_tail.args()[j]));
        }
        for (int j=0; j<F_tail.outputs(); j++) {
            values.push_back(Call::make(F_tail, call_args, j));
        }
        function.define(F_tail.args(), values);
    }

    // reduction definition
    {
        vector<Expr> values;
        vector<Expr> args;
        vector<Expr> call_args_curr_tile;
        vector<Expr> call_args_prev_tile;

        // use rxo for current tile term and rxo-1 previous tile term
        // remove xo as arg
        for (size_t j=0; j<F_tail.args().size(); j++) {
            if (F_tail.args()[j] == xo.name()) {
                args.push_back(rxo);
                call_args_curr_tile.push_back(rxo);
                call_args_prev_tile.push_back(max(rxo-1,0));
            } else {
                args.push_back(Var(F_tail.args()[j]));
                call_args_curr_tile.push_back(Var(F_tail.args()[j]));
                call_args_prev_tile.push_back(Var(F_tail.args()[j]));
            }
        }

        // xi corresponds to k elements of the tail, where k is filter order
        for (int j=0; j<F_tail.outputs(); j++) {
            Expr val = Call::make(function, call_args_curr_tile, j) +
                weight(tile-1, xi) * select(rxo>0, Call::make(function, call_args_prev_tile, j), 0);
            values.push_back(val);
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

    string s0 = F.name() + DELIM_START + INTRA_TILE_RESULT    + "_" + x.name() +  DELIM_END;
    string s1 = F.name() + DELIM_START + INTRA_TILE_SUB_RESULT+ "_" + x.name() +  DELIM_END;
    string s2 = F.name() + DELIM_START + INTRA_TILE_TAIL_TERM + "_" + x.name() +  DELIM_END;
    string s3 = F.name() + DELIM_START + INTER_TILE_TAIL_SUM  + "_" + x.name() +  DELIM_END;
    string s4 = F.name() + DELIM_START + INTER_TILE_DEPENDENCY+ "_" + x.name() +  DELIM_END;

    Function F_intra = create_intra_tile_term   (F,       split_info, s0.c_str());
    Function F_intra2= create_intra_tile_term   (F,       split_info, s1.c_str());
    Function F_tail  = create_intra_tail_term   (F_intra2,split_info, s2.c_str());
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
    split_info.func_list.push_back(F_intra2);
    split_info.func_list.push_back(F_tail);
    split_info.func_list.push_back(F_ctail);
    split_info.func_list.push_back(F_inter_deps);
}

// -----------------------------------------------------------------------------

static Function split_function_dimensions(
        Function F,
        vector<int>  dimension,
        vector<Var>  var,
        vector<Var>  inner_var,
        vector<Var>  outer_var,
        vector<RDom> rdom,
        vector<RDom> split_rdom,
        vector<Expr> tile)
{
    Function Fsplit(F.name() + DELIM_START + "split" + DELIM_END);

    // set of dimensions to split
    map<int,Expr>    dim_to_tile_width;
    map<string,Var>  var_to_inner_var;
    map<string,Var>  var_to_outer_var;
    map<string,Expr> var_to_inner_expr;
    map<string,Expr> var_to_outer_expr;

    // setup mappings between variables, RDoms and their inner/outer variants
    for (size_t i=0; i<dimension.size(); i++) {
        if (dim_to_tile_width.find(dimension[i]) != dim_to_tile_width.end()) {
            if (!equal(dim_to_tile_width[dimension[i]], tile[i])) {
                cerr << "Different tile widths specified for splitting same dimension" << endl;
                assert(false);
            }
        } else {
            dim_to_tile_width[dimension[i]] = tile[i];
        }

        var_to_inner_var[var[i].name()]  = inner_var[i];
        var_to_outer_var[var[i].name()]  = outer_var[i];
        var_to_inner_expr[var[i].name()] = inner_var[i];
        var_to_outer_expr[var[i].name()] = outer_var[i];
        var_to_inner_expr[rdom[i].x.name()]= split_rdom[i].x;
        var_to_outer_expr[rdom[i].x.name()]= split_rdom[i].y;
    }

    // go to each function dimension and replace the Var
    // with inner and outer Var, replace RDom with inner
    // and outer RDoms
    vector<string> pure_args = F.args();
    vector<Expr>   pure_values = F.values();
    vector<ReductionDefinition> reductions = F.reductions();


    for (map<int,Expr>::iterator it=dim_to_tile_width.begin(); it!=dim_to_tile_width.end(); it++) {
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

        // change the reduction defintions that involve rx
        for (size_t i=0; i<reductions.size(); i++) {
            vector<string> expr_vars = extract_vars_in_expr(
                    F.reductions()[i].args[dim]);

            if (expr_vars.size() > 1) {
                cerr << "Only one Var or RDom can be referenced in reduction definition  "
                    << "arg " << F.reductions()[i].args[var_index] << ", found multiple" << endl;
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

            // reduction definition args: replace rx by rxi,rxo
            reductions[i].args[var_index] = substitute(rx, rxi, reductions[i].args[var_index]);
            reductions[i].args.insert(reductions[i].args.begin()+var_index+1, rxo);

            // change reduction definition RHS values
            for (size_t j=0; j<reductions[i].values.size(); j++) {
                Expr value = reductions[i].values[j];

                // change calls to original Func by split Func
                // add rxo as calling arg and replace rx by rxi to this function
                value = substitute_func_call(F.name(), Fsplit, value);
                value = insert_arg_to_func_call(Fsplit.name(), var_index+1, rxo, value);
                value = substitute(rx, rxi, value);
                //value = substitute_in_func_call(Fsplit.name(), rx, rxi, value);
                reductions[i].values[j] = value;
            }

            Fsplit.define_reduction(reductions[i].args, reductions[i].values);
        }
    }

    return Fsplit;
}

// -----------------------------------------------------------------------------

void split(
        Func& func,
        Func filter_weights,
        vector<int>  dimension,
        vector<Var>  var,
        vector<Var>  inner_var,
        vector<Var>  outer_var,
        vector<RDom> rdom,
        vector<RDom> inner_rdom,
        vector<RDom> outer_rdom,
        vector<int>  order)
{
    size_t num_splits = var.size();

    assert(num_splits == dimension.size()  && "Each split must have a mapped function dimension");
    assert(num_splits == rdom.size()       && "Each split must have a mapped RDom");
    assert(num_splits == inner_var.size()  && "Each split must have a mapped inner Var");
    assert(num_splits == outer_var.size()  && "Each split must have a mapped outer Var");
    assert(num_splits == inner_rdom.size() && "Each split must have a mapped inner RDom");
    assert(num_splits == outer_rdom.size() && "Each split must have a mapped outer RDom");
    assert(num_splits == order.size()      && "Each split must have a mapped filter order");

    // individual tile boundaries
    vector<Expr> tile;
    for (size_t i=0; i<num_splits; i++) {
        Expr inner_rdom_extent = simplify(inner_rdom[i].x.extent()+1);
        tile.push_back(inner_rdom_extent);
    }

    Function F = func.function();

    assert(F.has_pure_definition() &&  "Func to be split must be defined");
    assert(!F.is_pure() && "Use Halide::Func::split for pure Funcs");


    // basic checks
    {
        // check variables
        for (size_t k=0; k<num_splits; k++) {
            int dim = dimension[k];

            // RDom to be split must be 1D, each reduction definition should be 1D reduction
            if (rdom[k].dimensions() != 1) {
                cerr << "RDom to split must be 1D, each reduction "
                    << "definition should use a unique be 1D RDom";
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

    // create 2D RDom which contain inner and outer RDom as two dimensions
    vector<RDom> split_rdom;
    for (size_t i=0; i<num_splits; i++) {
        split_rdom.push_back(RDom(inner_rdom[i].x.min(), inner_rdom[i].x.extent(),
               outer_rdom[i].x.min(), outer_rdom[i].x.extent(), rdom[i].x.name()));
    }


    // create a function whose dimensions are split without
    // affecting computation, all split operations affect this function;
    Function Fsplit = split_function_dimensions(F, dimension, var, inner_var,
            outer_var, rdom, split_rdom, tile);


    // modify the original function to index into the split function
    vector<string> args = F.args();
    vector<Expr> call_args(Fsplit.args().size());
    vector<Expr> values(F.outputs());
    for (size_t i=0; i<call_args.size(); i++) {
        call_args[i] = Var(Fsplit.args()[i]);
        for (size_t j=0; j<var.size(); j++) {
            if (inner_var[j].name() == Fsplit.args()[i]) {
                call_args[i] = var[j] % tile[j];
            }
            else if (outer_var[j].name() == Fsplit.args()[i]) {
                call_args[i] = var[j] / tile[j];
            }
        }
    }
    for (size_t i=0; i<F.outputs(); i++) {
        values[i] = Call::make(Fsplit, call_args, i);
    }
    F.clear_all_definitions();
    F.define(args, values);
    func = Func(F);

    // apply the individual split for each reduction variable - separate
    // the intra tile copmutation from inter tile computation
    queue<Function> funcs_to_split;
    funcs_to_split.push(Fsplit);

    for (size_t i=0; i<num_splits && !funcs_to_split.empty(); i++) {
        // populate the split info
        SplitInfo split_info;
        split_info.var            = var[i];
        split_info.rdom           = rdom[i];
        split_info.inner_var      = inner_var[i];
        split_info.outer_var      = outer_var[i];
        split_info.inner_rdom     = inner_rdom[i];
        split_info.outer_rdom     = outer_rdom[i];
        split_info.split_rdom     = split_rdom[i];
        split_info.tile_width     = tile[i];
        split_info.filter_order   = order[i];
        split_info.filter_dim     = dimension[i];
        split_info.filter_weights = tail_weights(filter_weights, tile[i], dimension[i], order[i]);

        queue<Function> funcs_to_split_in_next_dimension;

        while (!funcs_to_split.empty()) {
            Function f = funcs_to_split.front();
            funcs_to_split.pop();
            if (!check_for_pure_split(f, split_info)) {
                create_recursive_split(f, split_info);
            }

            // accumulate all sub Func's created to list
            // all these Func's have to be split for the next dimension
            for (size_t k=0; k<split_info.func_list.size(); k++) {
                funcs_to_split_in_next_dimension.push(split_info.func_list[k]);
            }

            // clear the list of sub Func's created
            split_info.func_list.clear();
        }

        // transfer all sub Func's to list to Func's that will be split in the next dimension
        funcs_to_split = funcs_to_split_in_next_dimension;
    }
}

// -----------------------------------------------------------------------------

void split(
        Func& func,
        int  dimension,
        Var  var,
        Var  inner_var,
        Var  outer_var,
        RDom rdom,
        RDom inner_rdom,
        RDom outer_rdom,
        int order)
{
    // default weights = 1
    Var x,y;
    Func weight;
    weight(x,y) = 1;

    split(func, weight, vec(dimension), vec(var), vec(inner_var), vec(outer_var),
            vec(rdom), vec(inner_rdom), vec(outer_rdom), vec(order));
}

void split(
        Func& func,
        vector<int>  dimensions,
        vector<Var>  vars,
        vector<Var>  inner_vars,
        vector<Var>  outer_vars,
        vector<RDom> rdoms,
        vector<RDom> inner_rdoms,
        vector<RDom> outer_rdoms,
        int order)
{
    // default weights = 1
    Var x,y;
    Func weight;
    weight(x,y) = 1;

    vector<int> orders(vars.size(), order);
    split(func, weight, dimensions, vars, inner_vars, outer_vars, rdoms, inner_rdoms,outer_rdoms, orders);
}

void split(
        Func& func,
        Func  weight,
        vector<int>  dimensions,
        vector<Var>  vars,
        vector<Var>  inner_vars,
        vector<Var>  outer_vars,
        vector<RDom> rdoms,
        vector<RDom> inner_rdoms,
        vector<RDom> outer_rdoms,
        int order)
{
    vector<int> orders(vars.size(), order);
    split(func, weight, dimensions, vars, inner_vars, outer_vars, rdoms, inner_rdoms,outer_rdoms, orders);
}
