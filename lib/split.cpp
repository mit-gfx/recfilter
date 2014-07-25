#include "recfilter.h"
#include "coefficients.h"
#include "modifiers.h"

using namespace Halide;
using namespace Halide::Internal;

using std::string;
using std::cerr;
using std::endl;
using std::vector;
using std::map;

// -----------------------------------------------------------------------------

#define INTRA_TILE_RESULT      "Intra"
#define INTRA_TILE_TAIL_TERM   "Tail"
#define INTER_TILE_TAIL_SUM    "CTail"
#define COMPLETE_TAIL_RESIDUAL "TDeps"
#define FINAL_RESULT_RESIDUAL  "Deps"
#define FINAL_TERM             "Final"
#define PRE_FINAL_TERM         "Sub"
#define DELIMITER              '-'

// -----------------------------------------------------------------------------

static vector<SplitInfo> group_scans_by_dimension(Function F, vector<SplitInfo> split_info) {
    vector<string> args = F.args();
    vector<Expr>  values = F.values();
    vector<ReductionDefinition> reductions = F.reductions();

    // split info struct must contain info about each dimension
    assert(split_info.size() == args.size());

    vector<ReductionDefinition> new_reductions;
    vector<SplitInfo>           new_split_info = split_info;

    // use all scans with dimension 0 first, then 1 and so on
    for (int i=0; i<split_info.size(); i++) {
        for (int j=0; j<split_info[i].num_splits; j++) {
            int curr = split_info[i].num_splits-1-j;
            int scan = split_info[i].scan_id[curr];
            new_reductions.push_back(reductions[scan]);
            new_split_info[i].scan_id[curr] = new_reductions.size()-1;
        }
    }
    assert(new_reductions.size() == reductions.size());

    // reorder the reduction definitions as per the new order
    F.clear_all_definitions();
    F.define(args, values);
    for (int i=0; i<new_reductions.size(); i++) {
        F.define_reduction(new_reductions[i].args, new_reductions[i].values);
    }

    return new_split_info;
}

// -----------------------------------------------------------------------------

static void extract_tails_from_each_scan(Function F_intra, vector<SplitInfo> split_info) {
    vector<string> pure_args = F_intra.args();
    vector<Expr> pure_values = F_intra.values();
    vector<ReductionDefinition> reductions = F_intra.reductions();

    // pure definitions remain unchanged
    F_intra.clear_all_definitions();
    F_intra.define(pure_args, pure_values);

    // new reductions to be added for all the split reductions
    map<int, std::pair< vector<Expr>, vector<Expr> > > new_reductions;

    // create the new reduction definitions to extract the tail
    // of each scan that is split
    for (int i=0; i<split_info.size(); i++) {
        int  dim   = -1;
        int  order = split_info[i].filter_order;
        Var  xi    = split_info[i].inner_var;
        Expr tile  = split_info[i].tile_width;

        // get the scan dimension
        for (int j=0; j<F_intra.args().size(); j++) {
            if (xi.name() == F_intra.args()[j]) {
                dim = j;
            }
        }
        assert(dim >= 0);

        // new reduction to extract the tail of each split scan
        for (int j=0; j<split_info[i].num_splits; j++) {
            int  scan_id     = split_info[i].scan_id[j];
            bool scan_causal = split_info[i].scan_causal[j];
            RDom rxt         = split_info[i].tail_rdom[j];

            vector<Expr> args      = reductions[scan_id].args;
            vector<Expr> call_args = reductions[scan_id].args;
            vector<Expr> values;

            // store the tail in a buffer of width equal to order
            // each scan's tail is stored at the end of the tile
            args     [dim] = simplify(tile + order*j+ rxt);
            call_args[dim] = simplify(scan_causal ? (tile-1-rxt) : rxt);

            for (int k=0; k<reductions[scan_id].values.size(); k++) {
                values.push_back(Call::make(F_intra, call_args, k));
            }

            new_reductions[scan_id] = std::make_pair(args, values);
        }
    }

    for (int i=0; i<reductions.size(); i++) {

        // RDom of the current reduction
        ReductionDomain rcurrent = reductions[i].domain;
        ReductionVariable rcurrent_var = rcurrent.domain()[0];

        // find the dimension of the reduction
        int split_dim = -1;
        int split_id  = -1;
        for (int j=0; split_dim<0 && j<split_info.size(); j++) {
            for (int k=0; split_id<0 && k<split_info[j].num_splits; k++) {
                if (rcurrent.same_as(split_info[j].inner_rdom[k].domain())) {
                    split_dim = j;
                    split_id  = k;
                }
            }
        }
        assert(split_dim>=0 && split_id>=0);

        // create a new reduction domain with as many dimensions as the input
        // and each dimension has the same min and extent as the individual
        // inner vars for each split
        vector<ReductionVariable> new_rvars;
        for (int j=0; j<split_info.size(); j++) {
            ReductionVariable rvar;
            rvar.var = "r" + split_info[split_dim].inner_var.name() + "." + split_info[j].var.name();

            if (j==split_dim) {
                rvar.min    = split_info[j].inner_rdom[split_id].x.min();
                rvar.extent = split_info[j].inner_rdom[split_id].x.extent();
            } else {
                rvar.min    = 0;
                rvar.extent = split_info[j].tile_width;
            }
            new_rvars.push_back(rvar);
        }
        RDom new_r = RDom(ReductionDomain(new_rvars));

        // replace all inner vars by reduction vars over the same domain
        // to restrict the scans from operating over tail buffer
        for (int j=0; j<split_info.size(); j++) {
            string old_var = (j==split_dim ? rcurrent_var.var : split_info[j].inner_var.name());
            RVar rvar = new_r[j];

            for (int k=0; k<reductions[i].args.size(); k++) {
                Expr a = reductions[i].args[k];
                a = substitute(old_var, rvar, a);
                reductions[i].args[k] = a;
            }
            for (int k=0; k<reductions[i].values.size(); k++) {
                Expr a = reductions[i].values[k];
                a = substitute(old_var, rvar, a);
                reductions[i].values[k] = a;
            }
        }
        F_intra.define_reduction(reductions[i].args, reductions[i].values);

        // add extra update steps to copy tail of each scan to another buffer
        // that is beyond the bounds of the intra tile RVars
        if (new_reductions.find(i) != new_reductions.end()) {
            vector<Expr> args   = new_reductions[i].first;
            vector<Expr> values = new_reductions[i].second;
            F_intra.define_reduction(args, values);
        }
    }
}


// -----------------------------------------------------------------------------

static Function create_intra_tile_term(Function F, vector<SplitInfo> split_info) {
    Function F_intra(F.name() + DELIMITER + INTRA_TILE_RESULT);

    // set of dimensions to split
    map<int,Expr>    dim_to_tile_width;
    map<string,Var>  var_to_inner_var;
    map<string,Var>  var_to_outer_var;
    map<string,Expr> var_to_image_width;
    map<string,Expr> var_to_inner_expr;
    map<string,Expr> var_to_outer_expr;
    map<string,Expr> var_to_reduction_extent;

    // setup mappings between variables, RDoms and their inner/outer variants
    for (int i=0; i<split_info.size(); i++) {
        dim_to_tile_width [split_info[i].filter_dim] = split_info[i].tile_width;
        var_to_inner_var  [split_info[i].var.name()] = split_info[i].inner_var;
        var_to_outer_var  [split_info[i].var.name()] = split_info[i].outer_var;
        var_to_inner_expr [split_info[i].var.name()] = split_info[i].inner_var;
        var_to_outer_expr [split_info[i].var.name()] = split_info[i].outer_var;
        var_to_image_width[split_info[i].var.name()] = split_info[i].image_width;
        for (int j=0; j<split_info[i].num_splits; j++) {
            var_to_inner_expr[split_info[i].rdom[j].x.name()]= split_info[i].inner_rdom[j].x;
            var_to_outer_expr[split_info[i].rdom[j].x.name()]= split_info[i].outer_var;
        }
    }

    vector<string> pure_args = F.args();
    vector<Expr>   pure_values = F.values();
    vector<ReductionDefinition> reductions = F.reductions();

    // create the split function
    {
        F_intra.define(pure_args, pure_values);

        // replace calls to original Func by split Func
        for (int i=0; i<reductions.size(); i++) {
            for (int j=0; j<reductions[i].values.size(); j++) {
                Expr value = reductions[i].values[j];
                value = substitute_func_call(F.name(), F_intra, value);
                reductions[i].values[j] = value;
            }
            F_intra.define_reduction(reductions[i].args, reductions[i].values);
        }
    }

    // go to each function dimension and replace the Var with inner
    // and outer Var, replace RDom with inner and outer RDoms
    map<int,Expr>::iterator it = dim_to_tile_width.begin();
    map<int,Expr>::iterator end= dim_to_tile_width.end();
    for (; it != end; it++) {
        int dim = it->first;
        string x = F.args()[dim];
        Expr tile_width = it->second;

        // redefine F_intra after splitting each dimension
        F_intra.clear_all_definitions();

        int var_index = -1;
        for (int i=0; var_index<0 && i<pure_args.size(); i++) {
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
            for (int i=0; i<pure_values.size(); i++) {
                pure_values[i] = substitute(x, tile_width*xo+xi, pure_values[i]);
            }

            F_intra.define(pure_args, pure_values);
        }

        // change the reduction definitions that involve rx
        for (int i=0; i<reductions.size(); i++) {
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

            // reduction definition args: replace rx by rxi,rxo and
            // image_width by tile_width (for anticausal filter)
            reductions[i].args[var_index] = substitute(rx, rxi, reductions[i].args[var_index]);
            reductions[i].args[var_index] = substitute(image_width, tile_width, reductions[i].args[var_index]);
            reductions[i].args.insert(reductions[i].args.begin()+var_index+1, rxo);

            // change reduction definition RHS values
            for (int j=0; j<reductions[i].values.size(); j++) {
                Expr value = reductions[i].values[j];

                // change calls to original Func by split Func
                // add rxo as calling arg and replace rx by rxi to this function
                value = substitute_func_call(F.name(), F_intra, value);
                value = insert_arg_in_func_call(F_intra.name(), var_index+1, rxo, value);
                value = substitute(rx, rxi, value);
                value = substitute(image_width, tile_width, value);
                reductions[i].values[j] = value;
            }

            F_intra.define_reduction(reductions[i].args, reductions[i].values);
        }
    }

    return F_intra;
}

// -----------------------------------------------------------------------------

static void apply_zero_boundary_on_inner_tiles(Function F_intra, vector<SplitInfo> split_info) {
    vector<string> pure_args = F_intra.args();
    vector<Expr>   pure_values = F_intra.values();
    vector<ReductionDefinition> reductions = F_intra.reductions();

    // go to each scan that is split and apply the inner tile condition on
    // all feedback terms
    for (int i=0; i<split_info.size(); i++) {
        Var xo         = split_info[i].outer_var;
        Expr num_tiles = split_info[i].num_tiles;
        Expr tile_width= split_info[i].tile_width;

        for (int j=0; j<split_info[i].num_splits; j++) {
            bool causal = split_info[i].scan_causal[j];
            int scan_id = split_info[i].scan_id[j];

            Expr boundary = (causal ? 0 : tile_width);
            Expr inner_tile_condition = (causal ? (xo==0) : (xo==num_tiles-1));

            int dim = -1;
            Expr arg;
            RDom rxi = split_info[i].inner_rdom[j];
            for (int k=0; k<reductions[scan_id].args.size(); k++) {
                if (expr_depends_on_var(reductions[scan_id].args[k], rxi.x.name())) {
                    dim = k;
                    arg = reductions[scan_id].args[k];
                }
            }
            assert(dim >= 0);

            for (int k=0; k<reductions[scan_id].values.size(); k++) {
                Expr val = reductions[scan_id].values[k];
                val = apply_zero_boundary_in_func_call(F_intra.name(),
                        dim, arg, boundary, inner_tile_condition, val);
                reductions[scan_id].values[k] = val;
            }
        }
    }

    // redefine the function if boundary conditions are applied
    F_intra.clear_all_definitions();
    F_intra.define(pure_args, pure_values);
    for (int i=0; i<reductions.size(); i++) {
        F_intra.define_reduction(reductions[i].args, reductions[i].values);
    }
}

// -----------------------------------------------------------------------------

static Function create_copy(Function F, string func_name) {
    Function B(func_name);

    // same pure definition
    B.define(F.args(), F.values());

    // replace all calls to intra tile term with the new term
    for (int i=0; i<F.reductions().size(); i++) {
        ReductionDefinition r = F.reductions()[i];
        vector<Expr> values;
        for (int j=0; j<r.values.size(); j++) {
            values.push_back(substitute_func_call(F.name(), B, r.values[j]));
        }
        B.define_reduction(r.args, values);
    }
    return B;
}

// -----------------------------------------------------------------------------

static vector<Function> create_intra_tail_term(
        Function F_intra,
        SplitInfo split_info,
        string func_name)
{
    Expr tile = split_info.tile_width;
    Var  xi   = split_info.inner_var;

    vector<Function> tail_functions_list;

    for (int k=0; k<split_info.num_splits; k++) {
        Function function(func_name + "_" + int_to_string(split_info.scan_id[k]));

        int scan_id = split_info.scan_id[k];
        int order   = split_info.filter_order;

        vector<string> pure_args;
        vector<Expr> call_args;
        vector<Expr> pure_values;

        // pure args same as F_intra with xi replaced by tail indices
        for (int i=0; i<F_intra.args().size(); i++) {
            string arg = F_intra.args()[i];
            if (arg == xi.name()) {
                pure_args.push_back(xi.name());
                call_args.push_back(tile + order*k + xi);
            } else {
                pure_args.push_back(arg);
                call_args.push_back(Var(arg));
            }
        }

        // call intra tile term at tail elements
        for (int i=0; i<F_intra.outputs(); i++) {
            pure_values.push_back(Call::make(F_intra, call_args, i));
        }

        function.define(pure_args, pure_values);

        tail_functions_list.push_back(function);
    }

    assert(tail_functions_list.size() == split_info.num_splits);

    return tail_functions_list;
}

// -----------------------------------------------------------------------------

static vector<Function> create_complete_tail_term(
        vector<Function> F_tail,
        SplitInfo split_info,
        string func_name)
{
    assert(split_info.num_splits == F_tail.size());

    Var  x    = split_info.var;
    Var  xi   = split_info.inner_var;
    Var  xo   = split_info.outer_var;
    int  dim  = split_info.filter_dim;
    int  order= split_info.filter_order;
    Expr tile = split_info.tile_width;
    Expr num_tiles = split_info.num_tiles;

    vector<Function> F_ctail;

    for (int k=0; k<split_info.num_splits; k++) {
        Function function(func_name + "_" + int_to_string(split_info.scan_id[k]));

        Image<float> weight = tail_weights(split_info, k);

        RDom rxo = split_info.outer_rdom[k];

        // pure definition
        {
            // pure definition simply initializes with the intra tile tail
            vector<Expr> call_args;
            vector<Expr> values;

            for (int i=0; i<F_tail[k].args().size(); i++) {
                call_args.push_back(Var(F_tail[k].args()[i]));
            }
            for (int i=0; i<F_tail[k].outputs(); i++) {
                values.push_back(Call::make(F_tail[k], call_args, i));
            }

            function.define(F_tail[k].args(), values);
        }

        // reduction definition
        {
            vector<Expr> args;
            vector<Expr> values;
            vector<Expr> call_args_curr_tile;
            vector<Expr> call_args_prev_tile;

            for (int i=0; i<F_tail[k].args().size(); i++) {
                string arg = F_tail[k].args()[i];
                if (arg == xo.name()) {
                    // replace xo by rxo.z or rxo.z-1 as tile idx,
                    if (split_info.scan_causal[k]) {
                        args.push_back(rxo.z);
                        call_args_curr_tile.push_back(rxo.z);
                        call_args_prev_tile.push_back(rxo.z-1);
                    } else {
                        args.push_back(simplify(num_tiles-1-rxo.z));
                        call_args_curr_tile.push_back(simplify(num_tiles-1-rxo.z));
                        call_args_prev_tile.push_back(num_tiles-rxo.z);
                    }
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

            // multiply each tail element with its weight before adding
            for (int i=0; i<F_tail[k].outputs(); i++) {
                Expr val = Call::make(function, call_args_curr_tile, i) +
                        weight(simplify(tile-rxo.y-1), rxo.x) *
                        Call::make(function, call_args_prev_tile, i);
                values.push_back(val);
            }

            function.define_reduction(args, values);
        }
        F_ctail.push_back(function);
    }

    assert(F_ctail.size() == split_info.num_splits);

    return F_ctail;
}

// -----------------------------------------------------------------------------

static vector<Function> create_tail_residual_term(
        vector<Function> F_ctail,
        SplitInfo split_info,
        string func_name)
{
    int order       = split_info.filter_order;
    Var  xi         = split_info.inner_var;
    Var  xo         = split_info.outer_var;
    Expr num_tiles  = split_info.num_tiles;
    Expr tile       = split_info.tile_width;

    int num_args    = F_ctail[0].args().size();
    int num_outputs = F_ctail[0].outputs();
    Type type       = F_ctail[0].output_types()[0];

    assert(split_info.num_splits == F_ctail.size());

    vector<Function> dependency_functions;

    for (int u=0; u<split_info.num_splits; u++) {
        Function function(func_name + "_" + int_to_string(split_info.scan_id[u]));

        // args are same as completed tail terms
        vector<string> args = F_ctail[0].args();
        vector<Expr> values(num_outputs, make_zero(type));

        // accumulate the completed tails of all the preceedings scans
        // the list F_ctail is in reverse order just as split_info struct
        for (int j=u+1; j<F_ctail.size(); j++) {

            // weight matrix for accumulating completed tail elements from scan u to scan j
            Image<float> weight  = tail_weights(split_info, j, u);

            // weight matrix for accumulating completed tail elements from scan u to scan j
            // for a tile that is clamped on all borders
            Image<float> c_weight= tail_weights(split_info, j, u, split_info.clamp_border);

            // expressions for prev tile and checking for first tile for causal/anticausal scan j
            Expr first_tile = (split_info.scan_causal[j] ? (xo==0) : (xo==num_tiles-1));
            Expr prev_tile  = (split_info.scan_causal[j] ? max(xo-1,0) : min(xo+1, simplify(num_tiles-1)));

            // size of tail is equal to filter order, accumulate all
            // elements of the tail
            for (int k=0; k<order; k++) {
                vector<Expr> call_args;
                for (int i=0; i<num_args; i++) {
                    string arg = F_ctail[j].args()[i];
                    if (xo.name() == arg) {
                        call_args.push_back(prev_tile);
                    } else if (arg == xi.name()) {
                        call_args.push_back(k);
                    } else {
                        call_args.push_back(Var(arg));
                    }
                }

                for (int i=0; i<num_outputs; i++) {
                    Expr val = Call::make(F_ctail[j], call_args, i);
                    Expr wt  = (split_info.scan_causal[j] ? weight(xi,k) : weight(tile-1-xi,k));
                    Expr cwt = (split_info.scan_causal[j] ? c_weight(xi,k): c_weight(tile-1-xi,k));

                    // use weights for clamped image if border clamping is requested
                    // change the weight for the first tile only, only this tile is affected
                    // by clamping the image at all borders
                    if (split_info.clamp_border && split_info.scan_causal[j]!=split_info.scan_causal[u]) {
                        Expr first_tile_u = (split_info.scan_causal[u] ? (xo==0) : (xo==num_tiles-1));
                        wt = select(first_tile_u, cwt, wt);
                    }

                    values[i] += select(first_tile, make_zero(type), wt*val);
                }
            }
        }
        function.define(args, values);
        dependency_functions.push_back(function);
    }

    assert(dependency_functions.size() == F_ctail.size());

    return dependency_functions;
}

// -----------------------------------------------------------------------------

static vector<Function> create_final_residual_term(
        vector<Function> F_ctail,
        SplitInfo split_info,
        string func_name)
{
    int order       = split_info.filter_order;
    Var  xi         = split_info.inner_var;
    Var  xo         = split_info.outer_var;
    Expr num_tiles  = split_info.num_tiles;
    Expr tile       = split_info.tile_width;

    assert(split_info.num_splits == F_ctail.size());

    // accumulate all completed tails
    vector<Function> F_deps;

    // accumulate contribution from each completed tail
    for (int j=0; j<split_info.num_splits; j++) {
        Function function(func_name + "_" + int_to_string(split_info.scan_id[j]));

        int num_args    = F_ctail[j].args().size();
        int num_outputs = F_ctail[j].outputs();
        Type type       = F_ctail[j].output_types()[0];

        // args are same as completed tail terms
        vector<string> args = F_ctail[j].args();
        vector<Expr> values(num_outputs, make_zero(type));

        // weight matrix for accumulating completed tail elements
        // of scan after applying only current scan
        Image<float> weight = tail_weights(split_info, j);

        // size of tail is equal to filter order, accumulate all
        // elements of the tail
        for (int k=0; k<order; k++) {
            vector<Expr> call_args;
            for (int i=0; i<num_args; i++) {
                string arg = F_ctail[j].args()[i];
                if (xo.name() == arg) {
                    if (split_info.scan_causal[j]) {
                        call_args.push_back(max(xo-1,0));  // prev tile
                    } else {
                        call_args.push_back(min(xo+1, simplify(num_tiles-1)));
                    }
                } else if (arg == xi.name()) {
                    call_args.push_back(k);
                } else {
                    call_args.push_back(Var(arg));
                }
            }

            for (int i=0; i<num_outputs; i++) {
                Expr val;
                if (split_info.scan_causal[j]) {
                    val = select(xo>0,
                            weight(xi,k) * Call::make(F_ctail[j], call_args, i),
                            make_zero(type));
                } else {
                    val = select(xo<num_tiles-1,
                            weight(simplify(tile-1-xi),k) * Call::make(F_ctail[j], call_args, i),
                            make_zero(type));
                }
                values[i] = simplify(values[i] + val);
            }
        }
        function.define(args, values);

        F_deps.push_back(function);
    }

    return F_deps;
}

// -----------------------------------------------------------------------------

static void add_residual_to_tails(
        vector<Function> F_tail,
        vector<Function> F_deps,
        SplitInfo split_info)
{
    Var  xi   = split_info.inner_var;
    Expr tile = split_info.tile_width;

    // add the dependency term of each scan to the preceeding scan
    // the first scan does not have any residuals
    for (int j=0; j<split_info.num_splits-1; j++) {
        vector<string> args = F_tail[j].args();
        vector<Expr> values = F_tail[j].values();
        vector<Expr> call_args;

        for (int i=0; i<F_tail[j].args().size(); i++) {
            string arg = F_tail[j].args()[i];
            if (arg == xi.name()) {
                if (split_info.scan_causal[j]) {
                    call_args.push_back(simplify(tile-xi-1));
                } else {
                    call_args.push_back(xi);
                }
            } else {
                call_args.push_back(Var(arg));
            }
        }

        for (int i=0; i<values.size(); i++) {
            values[i] += Call::make(F_deps[j], call_args, i);
        }

        // reductions remain unaffected
        vector<ReductionDefinition> reductions = F_tail[j].reductions();

        // redefine tail
        F_tail[j].clear_all_definitions();
        F_tail[j].define(args, values);
        for (int i=0; i<reductions.size(); i++) {
            F_tail[j].define_reduction(reductions[i].args, reductions[i].values);
        }
    }
}

// -----------------------------------------------------------------------------

static void add_prev_dimension_residual_to_tails(
        Function F_intra,
        vector<Function> F_tail,
        vector<Function> F_tail_prev,
        SplitInfo split_info,
        SplitInfo split_info_prev)
{
    Var  x    = split_info.var;
    Var  xi   = split_info.inner_var;
    Var  xo   = split_info.outer_var;
    Expr tile = split_info.tile_width;
    Expr num_tiles = split_info.num_tiles;

    Var  yi             = split_info_prev.inner_var;
    Var  yo             = split_info_prev.outer_var;
    Expr num_tiles_prev = split_info_prev.num_tiles;

    // add the residual term of previous dimension to the completed
    // tail of current dimension
    for (int j=0; j<F_tail.size(); j++) {
        vector<string> pure_args = F_tail[j].args();
        vector<Expr> pure_values = F_tail[j].values();

        // all tails of prev dimension add a residual
        for (int k=0; k<F_tail_prev.size(); k++) {

            // first scan the tail in the current dimension according using intra term
            Function F_tail_prev_scanned(F_tail_prev[k].name() + DELIMITER + x.name()
                    + DELIMITER + int_to_string(split_info.scan_id[j]));

            // pure def simply calls the completed tail
            vector<Expr> prev_call_args;
            vector<Expr> prev_values;
            for (int i=0; i<F_tail_prev[k].args().size(); i++) {
                prev_call_args.push_back(Var(F_tail_prev[k].args()[i]));
            }
            for (int i=0; i<F_tail_prev[k].outputs(); i++) {
                prev_values.push_back(Call::make(F_tail_prev[k], prev_call_args, i));
            }
            F_tail_prev_scanned.define(F_tail_prev[k].args(), prev_values);

            // apply all scans between the prev dimension and current dimension
            int first_scan = split_info_prev.scan_id[0]+1;
            int last_scan  = split_info.scan_id[j];

            for (int i=first_scan; i<=last_scan; i++) {
                vector<Expr> args;
                vector<Expr> values;
                for (int u=0; u<F_intra.reductions()[i].args.size(); u++) {
                    args.push_back(F_intra.reductions()[i].args[u]);
                }
                for (int u=0; u<F_intra.reductions()[i].values.size(); u++) {
                    Expr val = F_intra.reductions()[i].values[u];
                    val = substitute_func_call(F_intra.name(), F_tail_prev_scanned, val);
                    values.push_back(val);
                }
                F_tail_prev_scanned.define_reduction(args, values);
            }


            // weight matrix for accumulating completed tail elements
            // of scan after applying all subsequent scans
            Image<float> weight  = tail_weights(split_info_prev, k, 0);

            // weight matrix for accumulating completed tail elements
            // of scan after applying all subsequent scans
            // for a tile that is clamped on all borders
            Image<float> c_weight= tail_weights(split_info_prev, k, 0, split_info_prev.clamp_border);

            // size of tail is equal to filter order, accumulate all
            // elements of the tail
            for (int o=0; o<split_info_prev.filter_order; o++) {
                vector<Expr> call_args;
                for (int i=0; i<F_tail_prev_scanned.args().size(); i++) {
                    string arg = F_tail_prev_scanned.args()[i];
                    if (arg == yo.name()) {
                        // if the scan in prev dimension was causal, then accumulate
                        // the tail of prev tile in that dimension, else next tile
                        if (split_info_prev.scan_causal[k]) {
                            call_args.push_back(yo-1);
                        } else {
                            call_args.push_back(yo+1);
                        }
                    }
                    else if (arg == yi.name()) {
                        call_args.push_back(o);
                    } else if (arg == xi.name()) {
                        // if current scan is causal then accumulate the last
                        // elements of prev dimension tail
                        if (split_info.scan_causal[j]) {
                            call_args.push_back(simplify(tile-1-xi));
                        } else {
                            call_args.push_back(xi);
                        }
                    } else {
                        call_args.push_back(Var(arg));
                    }
                }

                // accumulate the tail of scan in the prev dimension
                for (int i=0; i<pure_values.size(); i++) {
                    // expression for checking for first tile for causal/anticausal scan
                    Expr first_tile = (split_info_prev.scan_causal[k] ? (yo==0) : (yo==num_tiles-1));
                    Expr last_tile  = (split_info_prev.scan_causal[k] ? (yo==num_tiles-1) : (yo==0));
                    Expr zero= make_zero(F_tail_prev_scanned.output_types()[i]);
                    Expr val = Call::make(F_tail_prev_scanned, call_args, i);
                    Expr wt  = (split_info_prev.scan_causal[k] ? weight(yi,o)  : weight(simplify(tile-1-yi),o));
                    Expr cwt = (split_info_prev.scan_causal[k] ? c_weight(yi,o): c_weight(simplify(tile-1-yi),o));
                    // use weights for clamped image if border clamping is requested
                    // change the weight for the first tile only, only this tile is affected
                    // by clamping the image at all borders
                    if (split_info_prev.clamp_border) {
                        wt = select(last_tile, cwt, wt);
                    }

                    pure_values[i] += select(first_tile, zero, wt*val);
                }
            }
        }

        // reduction defs of the tail remain unaffected
        vector<ReductionDefinition> reductions = F_tail[j].reductions();

        // redefine tail
        F_tail[j].clear_all_definitions();
        F_tail[j].define(pure_args, pure_values);
        for (int i=0; i<reductions.size(); i++) {
            F_tail[j].define_reduction(reductions[i].args, reductions[i].values);
        }
    }
}

// -----------------------------------------------------------------------------

static void add_all_residuals_to_final_result(
        Function F,
        vector< vector<Function> >  F_deps,
        vector<SplitInfo> split_info)
{
    vector<string> pure_args   = F.args();
    vector<Expr>   pure_values = F.values();
    vector<ReductionDefinition> reductions = F.reductions();

    assert(split_info.size() == F_deps.size());

    // define a function as a copy of the function F
    Function F_sub(F.name() + DELIMITER + PRE_FINAL_TERM);
    F_sub.define(pure_args, pure_values);

    // each F_deps represents the residuals from one dimension
    // add this residual to the first scan in next dimension
    Function F_deps_last;

    for (int i=0; i<F_deps.size(); i++) {
        Expr tile_width = split_info[i].tile_width;
        Expr num_tiles  = split_info[i].num_tiles;

        for (int j=0; j<F_deps[i].size(); j++) {
            int next_scan = split_info[i].scan_id[j]+1;

            if (next_scan < reductions.size()) {
                vector<Expr> call_args = reductions[next_scan].args;

                // multiply curr scan's residual to next scan after multiplying
                // by next scan's feedforward coeff;
                Expr fwd = split_info[i].feedfwd_coeff(next_scan);

                // special case if the image is clamped on all borders
                // add 1 to the multuplicant if next scan is in same
                // dimension and reverse causality or different dimension
                if (split_info[i].clamp_border) {
                    if (j>0) {
                        // next scan is either in same dimension
                        bool next_causal = split_info[i].scan_causal[j-1];
                        RVar next_rxi    = split_info[i].inner_rdom[j-1];
                        Var  next_xo     = split_info[i].outer_var;
                        Expr cond        = (next_rxi==0);
                        cond = cond && (next_causal ? (next_xo==0) : (next_xo==num_tiles-1));
                        fwd = select(cond, 1.0f, fwd);

                    } else {
                        // or next scan in next dimension
                        int a = split_info[i+1].num_splits-1;
                        bool next_causal = split_info[i+1].scan_causal[a];
                        RVar next_rxi    = split_info[i+1].inner_rdom[a];
                        Var  next_xo     = split_info[i+1].outer_var;
                        Expr cond        = (next_rxi==0);
                        cond = cond && (next_causal ? (next_xo==0) : (next_xo==num_tiles-1));
                        fwd = select(cond, 1.0f, fwd);
                    }
                }

                for (int k=0; k<reductions[next_scan].values.size(); k++) {
                    reductions[next_scan].values[k] +=
                        fwd * Call::make(F_deps[i][j], call_args, k);
                }
            } else {
                // add this to the final result of all the scans
                F_deps_last = F_deps[i][j];
            }
        }
    }

    // replace calls to F by F_sub in the reduction def of F_sub
    for (int i=0; i<reductions.size(); i++) {
        vector<Expr> values;
        for (int j=0; j<reductions[i].values.size(); j++) {
            Expr val = substitute_func_call(F.name(), F_sub, reductions[i].values[j]);
            values.push_back(val);
        }
        F_sub.define_reduction(reductions[i].args, values);
    }

    // add the residual of the last scan and above computed
    // function to get the final result
    vector<Expr> final_call_args;
    vector<Expr> final_pure_values;
    for (int i=0; i<pure_args.size(); i++) {
        final_call_args.push_back(Var(pure_args[i]));
    }
    for (int i=0; i<F.outputs(); i++) {
        Expr val  = Call::make(F_sub, final_call_args, i) +
            Call::make(F_deps_last, final_call_args, i);
        final_pure_values.push_back(val);
    }

    F.clear_all_definitions();
    F.define(pure_args, final_pure_values);
}

// -----------------------------------------------------------------------------

static vector< vector<Function> > split_scans(
        Function F_intra,
        vector<SplitInfo> &split_info)
{
    vector< vector<Function> > F_tail_list;
    vector< vector<Function> > F_ctail_list;
    vector< vector<Function> > F_tdeps_list;
    vector< vector<Function> > F_deps_list;

    /// TODO: this order of Function generation requires that F_ctail are
    /// redefined after they are called in F_tdeps, change the order of
    /// Function generation to fix this.

    for (int i=0; i<split_info.size(); i++) {
        string x = split_info[i].var.name();

        string s0 = F_intra.name() + DELIMITER + INTRA_TILE_TAIL_TERM  + "_" + x;
        string s1 = F_intra.name() + DELIMITER + INTER_TILE_TAIL_SUM   + "_" + x;
        string s2 = F_intra.name() + DELIMITER + COMPLETE_TAIL_RESIDUAL+ "_" + x;
        string s3 = F_intra.name() + DELIMITER + FINAL_RESULT_RESIDUAL + "_" + x;

        vector<Function> F_tail  = create_intra_tail_term    (F_intra, split_info[i], s0);
        vector<Function> F_ctail = create_complete_tail_term (F_tail,  split_info[i], s1);
        vector<Function> F_tdeps = create_tail_residual_term (F_ctail, split_info[i], s2);
        vector<Function> F_deps  = create_final_residual_term(F_ctail, split_info[i], s3);

        // add the dependency from each scan to the tail of the next scan
        // this ensures that the tail of each scan includes the complete
        // result from all previous scans
        add_residual_to_tails(F_ctail, F_tdeps, split_info[i]);

        // add the residuals from split up scans in all previous
        // dimensions to this scan
        for (int j=0; j<i; j++) {
            add_prev_dimension_residual_to_tails(F_intra, F_ctail,
                    F_ctail_list[j], split_info[i], split_info[j]);
        }

        F_tail_list .push_back(F_tail);
        F_ctail_list.push_back(F_ctail);
        F_tdeps_list.push_back(F_tdeps);
        F_deps_list .push_back(F_deps);
    }
    return F_deps_list;
}

// -----------------------------------------------------------------------------

void RecFilter::split(map<string,Expr> dim_tile) {
    Function F = contents.ptr->recfilter.function();

    // flush out any previous splitting info
    for (int i=0; i<contents.ptr->split_info.size(); i++) {
        contents.ptr->split_info[i].tile_width = Expr();
        contents.ptr->split_info[i].num_tiles  = Expr();
        contents.ptr->split_info[i].inner_var  = Var();
        contents.ptr->split_info[i].outer_var  = Var();
        contents.ptr->split_info[i].inner_rdom.clear();
        contents.ptr->split_info[i].outer_rdom.clear();
        contents.ptr->split_info[i].tail_rdom.clear();
    }

    // populate splitting info to all dimensions
    for (map<string,Expr>::iterator it=dim_tile.begin(); it!=dim_tile.end(); it++) {
        int dimension = -1;

        string  x = it->first;
        Expr tile_width = it->second;

        for (int j=0; j<F.args().size(); j++) {
            if (F.args()[j] == x) {
                dimension = j;
            }
        }
        if (dimension == -1) {
            cerr << "Variable " << x << " does not correspond to any "
                << "dimension of the recursive filter " << F.name() << endl;
            assert(false);
        }

        SplitInfo s = contents.ptr->split_info[dimension];

        // set tile width and number of tiles
        s.tile_width = tile_width;
        s.num_tiles  = s.image_width / s.tile_width;

        // set inner var and outer var
        s.inner_var = Var(s.var.name() + "i");
        s.outer_var = Var(s.var.name() + "o");

        // remove any info of previous splitting operations
        s.inner_rdom.clear();
        s.outer_rdom.clear();
        s.tail_rdom.clear();

        // set inner rdom, outer rdom and rdom to extract tails
        for (int j=0; j<s.num_splits; j++) {
            Expr   m = s.rdom[j].x.min();
            Expr   e = s.rdom[j].x.extent();
            string n = s.rdom[j].x.name();

            s.inner_rdom.push_back(RDom(m/tile_width, e/tile_width, n+"i"));

            // outer_rdom.x: over tail elems of prev tile to compute tail of current tile
            // outer_rdom.y: over all tail elements of current tile
            // outer_rdom.z: over all tiles
            s.outer_rdom.push_back(RDom(
                        0, s.filter_order,
                        0, s.filter_order,
                        simplify(m/tile_width+1), s.num_tiles-1,
                        n+"o"));

            // tail_rdom.x: over all tail elems of current tile
            s.tail_rdom.push_back(RDom(0,s.filter_order, n+"t"));
        }

        contents.ptr->split_info[dimension] = s;
    }

    // group scans in same dimension together and change the order of splits accordingly
    contents.ptr->split_info = group_scans_by_dimension(F, contents.ptr->split_info);

    // apply the actual splitting
    Function F_final;
    {
        // create a copy of the split info structs retaining only the
        // dimensions to be split
        vector<SplitInfo> split_info_current;
        for (int i=0; i<contents.ptr->split_info.size(); i++) {
            if (contents.ptr->split_info[i].tile_width.defined()) {
                split_info_current.push_back(contents.ptr->split_info[i]);
            }
        }

        // compute the intra tile result
        Function F_intra = create_intra_tile_term(F, split_info_current);

        // apply clamped boundary on border tiles and zero boundary on
        // inner tiles if explicit border clamping is requested
        // TODO
        //{
        //    for (int j=0; j<split_info_current.size(); j++) {
        //        split_info_current[j].clamp_border = true;
        //    }
        //    apply_zero_boundary_on_inner_tiles(F_intra, split_info_current);
        //}

        // create a function will hold the final result, copy of the intra tile term
        F_final = create_copy(F_intra, F.name() + DELIMITER + FINAL_TERM);

        // compute the residuals from splits in each dimension
        vector< vector<Function> > F_deps = split_scans(F_intra, split_info_current);

        // transfer the tail of each scan to another buffer
        extract_tails_from_each_scan(F_intra, split_info_current);

        // add all the residuals to the final term
        add_all_residuals_to_final_result(F_final, F_deps, split_info_current);
    }

    // change the original function to index into the final term
    {
        vector<string> args = F.args();
        vector<Expr> values;
        vector<Expr> call_args;
        for (int i=0; i<F_final.args().size(); i++) {
            string arg = F_final.args()[i];
            call_args.push_back(Var(arg));

            for (int j=0; j<contents.ptr->split_info.size(); j++) {
                Var var         = contents.ptr->split_info[j].var;
                Var inner_var   = contents.ptr->split_info[j].inner_var;
                Var outer_var   = contents.ptr->split_info[j].outer_var;
                Expr tile_width = contents.ptr->split_info[j].tile_width;
                if (arg == inner_var.name()) {
                    call_args[i] = substitute(arg, var%tile_width, call_args[i]);
                } else if (arg == outer_var.name()) {
                    call_args[i] = substitute(arg, var/tile_width, call_args[i]);
                }
            }
        }
        for (int i=0; i<F.outputs(); i++) {
            Expr val = Call::make(F_final, call_args, i);
            values.push_back(val);
        }
        F.clear_all_definitions();
        F.define(args, values);
    }

    contents.ptr->recfilter = Func(F);

    // clear the depenendency graph of recursive filter
    contents.ptr->func_map.clear();
    contents.ptr->func_list.clear();
}

void RecFilter::split(Var x, Expr tx) {
    map<string,Expr> dim_tile;
    dim_tile[x.name()] = tx;
    split(dim_tile);
}

void RecFilter::split(Var x, Expr tx, Var y, Expr ty) {
    map<string,Expr> dim_tile;
    dim_tile[x.name()] = tx;
    dim_tile[y.name()] = ty;
    split(dim_tile);
}
