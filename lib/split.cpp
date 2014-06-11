#include "split.h"
#include "split_macros.h"
#include "split_utils.h"

using namespace Halide;
using namespace Halide::Internal;

using std::string;
using std::cerr;
using std::endl;
using std::vector;
using std::queue;
using std::map;

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

    //Var xs(SCAN_STAGE_ARG);

    vector<string> pure_args = F.args();
    vector<Expr>   pure_values = F.values();
    vector<ReductionDefinition> reductions = F.reductions();

    // create the split function and add the scan stage arg
    {
        //int scan_stage_var_index = pure_args.size();

        // add the scan stage arg to pure args
        //pure_args.push_back(xs.name());

        // set pure values to zero for all scan stages other than 0
        //for (int i=0; i<pure_values.size(); i++) {
        //    pure_values[i] = select(xs==0, pure_values[i], 0);
        //}
        F_intra.define(pure_args, pure_values);

        for (int i=0; i<reductions.size(); i++) {
            // find the actual scan stage for this reduction
            //int scan_stage = -1;
            //for (int j=0; j<split_info.size(); j++) {
            //    for (int k=0; k<split_info[j].num_splits; k++) {
            //        if (split_info[j].scan_id[k] == i) {
            //            scan_stage = split_info[j].scan_stage[k];
            //        }
            //    }
            //}
            //if (scan_stage < 0) {
            //    cerr << "Scan stage not found for reduction definition " << i << ", "
            //        << "most probably because splits have not been specified "
            //        << "for all reduction definitions" << endl;
            //    assert(false);
            //}

            //// add scan stage value as arg to reductions, so that each
            //// reduction computes at its own stage
            //reductions[i].args.push_back(scan_stage);

            // replace calls to original Func by split Func
            // add scan stage value as arg to reduction args and recursive calls
            for (int j=0; j<reductions[i].values.size(); j++) {
                Expr value = reductions[i].values[j];
                value = substitute_func_call(F.name(), F_intra, value);
            //  value = insert_arg_in_func_call(F_intra.name(),
            //        scan_stage_var_index, scan_stage, value);
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

            // reduction definition args: replace rx by rxi,rxo and image_width by tile_width (for anticausal filter)
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

        int scan_stage = split_info.scan_stage[k];

        vector<string> pure_args;
        vector<Expr> call_args;
        vector<Expr> pure_values;

        // pure args same as F_intra except for scan stage arg
        // args for calling F_intra
        // - xi replaced by causal/anticausal tail indices
        // - scan stage arg set to actual scan stage value
        for (int i=0; i<F_intra.args().size(); i++) {
            string arg = F_intra.args()[i];
            if (arg == SCAN_STAGE_ARG) {
                call_args.push_back(scan_stage);
            } else if (arg == xi.name()) {
                pure_args.push_back(xi.name());
                if (split_info.scan_causal[k]) {
                    call_args.push_back(tile-xi-1);
                } else {
                    call_args.push_back(xi);
                }
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
                        args.push_back(num_tiles-1-rxo.z);
                        call_args_curr_tile.push_back(num_tiles-1-rxo.z);
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
                values.push_back(Call::make(function, call_args_curr_tile, i) +
                        weight(simplify(tile-rxo.y-1), rxo.x) *
                        Call::make(function, call_args_prev_tile, i));
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

            // weight matrix for accumulating completed tail elements
            // from scan u to scan j
            Image<float> weight = tail_weights(split_info, j, u);

            // size of tail is equal to filter order, accumulate all
            // elements of the tail
            for (int k=0; k<order; k++) {
                vector<Expr> call_args;
                for (int i=0; i<num_args; i++) {
                    string arg = F_ctail[j].args()[i];
                    if (xo.name() == arg) {
                        // prev or next tile as per causality
                        if (split_info.scan_causal[j]) {
                            call_args.push_back(max(simplify(xo-1),0));
                        } else {
                            call_args.push_back(min(simplify(xo+1), simplify(num_tiles-1)));
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
                                weight(tile-1-xi,k) * Call::make(F_ctail[j], call_args, i),
                                make_zero(type));
                    }
                    values[i] = simplify(values[i] + val);
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

static Function create_final_residual_term(
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
    Function function(func_name);

    int num_args    = F_ctail[0].args().size();
    int num_outputs = F_ctail[0].outputs();
    Type type       = F_ctail[0].output_types()[0];

    // args are same as completed tail terms
    vector<string> args = F_ctail[0].args();
    vector<Expr> values(num_outputs, make_zero(type));

    // accumulate contribution from each completed tail
    for (int j=0; j<split_info.num_splits; j++) {

        // weight matrix for accumulating completed tail elements
        // of scan after applying all subsequent scans
        Image<float> weight = tail_weights(split_info, j, 0);

        // size of tail is equal to filter order, accumulate all
        // elements of the tail
        for (int k=0; k<order; k++) {
            vector<Expr> call_args;
            for (int i=0; i<num_args; i++) {
                string arg = F_ctail[j].args()[i];
                if (xo.name() == arg) {
                    if (split_info.scan_causal[j]) {
                        call_args.push_back(max(simplify(xo-1),0));  // prev tile
                    } else {
                        call_args.push_back(min(simplify(xo+1), simplify(num_tiles-1)));
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
                            weight(tile-1-xi,k) * Call::make(F_ctail[j], call_args, i),
                            make_zero(type));
                }
                values[i] = simplify(values[i] + val);
            }
        }
    }
    function.define(args, values);

    return function;
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
                    call_args.push_back(tile-xi-1);
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

            // apply scans in current dimension the tail from prev dimension
            int first_scan_in_this_dimension = split_info.scan_id[split_info.num_splits-1];
            int last_scan_in_this_dimension  = split_info.scan_id[j];
            int scan_stage_var_index = -1;
            for (int i=0; i<F_intra.args().size(); i++) {
                if (F_intra.args()[i] == SCAN_STAGE_ARG) {
                    scan_stage_var_index = i;
                }
            }

            for (int i=first_scan_in_this_dimension; i<=last_scan_in_this_dimension; i++) {
                // do not add the scan if its scan stage is less than scan stage of scan in
                // current dimension
                const int  prev_scan_stage = split_info_prev.scan_stage[k];
                const int* scan_stage = as_const_int(
                        F_intra.reductions()[i].args[scan_stage_var_index]);
                assert(scan_stage && "Could not convert scan stage Expr to const int");
                if (prev_scan_stage >= *scan_stage) {
                    continue;
                }
                vector<Expr> args;
                vector<Expr> values;
                for (int u=0; u<F_intra.reductions()[i].args.size(); u++) {
                    args.push_back(F_intra.reductions()[i].args[u]);
                }
                args.erase(args.begin()+scan_stage_var_index);
                for (int u=0; u<F_intra.reductions()[i].values.size(); u++) {
                    Expr val = F_intra.reductions()[i].values[u];
                    val = remove_arg_from_func_call(F_intra.name(), scan_stage_var_index, val);
                    val = substitute_func_call(F_intra.name(), F_tail_prev_scanned, val);
                    values.push_back(val);
                }
                F_tail_prev_scanned.define_reduction(args, values);
            }


            // weight matrix for accumulating completed tail elements
            // of scan after applying all subsequent scans
            Image<float> weight = tail_weights(split_info_prev, k, 0);

            // size of tail is equal to filter order, accumulate all
            // elements of the tail
            for (int o=0; o<split_info_prev.filter_order; o++) {
                vector<Expr> call_args;
                for (int i=0; i<F_tail_prev_scanned.args().size(); i++) {
                    string arg = F_tail_prev_scanned.args()[i];
                    if (arg == yo.name()) {
                        if (split_info_prev.scan_causal[k]) {
                            call_args.push_back(yo-1);
                        } else {
                            call_args.push_back(yo+1);
                        }
                    } else if (arg == yi.name()) {
                        call_args.push_back(o);
                    } else if (arg == xi.name()) {
                        if (split_info_prev.scan_causal[k]) {
                            call_args.push_back(tile-1-xi);
                        } else {
                            call_args.push_back(xi);
                        }
                    } else {
                        call_args.push_back(Var(arg));
                    }
                }

                for (int i=0; i<pure_values.size(); i++) {
                    Expr val;
                    if (split_info_prev.scan_causal[k]) {
                        val = select(yo>0,
                                weight(yi,o) * Call::make(F_tail_prev_scanned, call_args, i),
                                make_zero(F_tail_prev_scanned.output_types()[i]));
                    } else {
                        val = select(yo<num_tiles_prev-1,
                                weight(tile-1-yi,o) * Call::make(F_tail_prev_scanned, call_args, i),
                                make_zero(F_tail_prev_scanned.output_types()[i]));
                    }
                    pure_values[i] += val;
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
        vector<Function>  F_deps,
        vector<SplitInfo> split_info)
{
    vector<string> pure_args   = F.args();
    vector<Expr>   pure_values = F.values();
    vector<ReductionDefinition> reductions = F.reductions();

    assert(split_info.size() == F_deps.size());

    // each F_deps represents the residuals from that dimension
    // add this residual to the first scan in next dimension

    // define a function that computes the above
    Function F_sub(F.name() + DELIMITER + PRE_FINAL_TERM);
    F_sub.define(pure_args, pure_values);

    for (int i=0; i<F_deps.size()-1; i++) {
        int first_scan_next_dim = split_info[i+1].scan_id[ split_info[i+1].num_splits-1 ];
        vector<Expr> call_args = reductions[first_scan_next_dim].args;
        for (int j=0; j<reductions[first_scan_next_dim].values.size(); j++) {
            reductions[first_scan_next_dim].values[j] += Call::make(F_deps[i], call_args, j);
        }
    }

    for (int i=0; i<reductions.size(); i++) {
        vector<Expr> values;
        for (int j=0; j<reductions[i].values.size(); j++) {
            Expr val = substitute_func_call(F.name(), F_sub, reductions[i].values[j]);
            values.push_back(val);
        }
        F_sub.define_reduction(reductions[i].args, values);
    }

    // add the residual of the last dimension and above computed function to
    // get the final result
    vector<Expr> final_call_args;
    vector<Expr> final_pure_values;
    for (int i=0; i<pure_args.size(); i++) {
        final_call_args.push_back(Var(pure_args[i]));
    }
    for (int i=0; i<F.outputs(); i++) {
        final_pure_values.push_back(Call::make(F_sub, final_call_args, i) +
                Call::make(F_deps[F_deps.size()-1], final_call_args, i));
    }

    F.clear_all_definitions();
    F.define(pure_args, final_pure_values);

    cerr << F << F_sub << endl;
}

// -----------------------------------------------------------------------------

static vector<Function> create_recursive_split(
        Function F_intra,
        vector<SplitInfo> &split_info)
{
    vector< vector<Function> > F_tail_list;
    vector< vector<Function> > F_ctail_list;
    vector< vector<Function> > F_tdeps_list;
    vector< Function         > F_deps_list;

    for (int i=0; i<split_info.size(); i++) {
        string x = split_info[i].var.name();

        string s0 = F_intra.name() + DELIMITER + INTRA_TILE_TAIL_TERM  + "_" + x;
        string s1 = F_intra.name() + DELIMITER + INTER_TILE_TAIL_SUM   + "_" + x;
        string s2 = F_intra.name() + DELIMITER + COMPLETE_TAIL_RESIDUAL+ "_" + x;
        string s3 = F_intra.name() + DELIMITER + FINAL_RESULT_RESIDUAL + "_" + x;

        vector<Function> F_tail  = create_intra_tail_term    (F_intra, split_info[i], s0);
        vector<Function> F_ctail = create_complete_tail_term (F_tail,  split_info[i], s1);
        vector<Function> F_tdeps = create_tail_residual_term (F_ctail, split_info[i], s2);
        Function         F_deps  = create_final_residual_term(F_ctail, split_info[i], s3);

        // add the dependency from each scan to the tail of the next scan
        // this ensures that the tail of each scan includes the complete
        // result from all previous scans
        add_residual_to_tails(F_ctail, F_tdeps, split_info[i]);

        // add the residuals from split up scans in previous dimensions
        // to this scan
        if (i>0) {
            add_prev_dimension_residual_to_tails(F_intra, F_ctail,
                    F_ctail_list[i-1], split_info[i], split_info[i-1]);
        }

        F_tail_list .push_back(F_tail);
        F_ctail_list.push_back(F_ctail);
        F_tdeps_list.push_back(F_tdeps);
        F_deps_list .push_back(F_deps);
    }
    return F_deps_list;
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
    vector<int> order(dimension.size(), 1);         // default first order

    Image<float> weights(dimension.size(), 1);      // default filter weight = 1
    for (int i=0; i<weights.width(); i++) {
        for (int j=0; j<weights.height(); j++) {
            weights(i,j) = 1.0f;
        }
    }

    split(func, weights, dimension, var, inner_var, outer_var,
            rdom, inner_rdom, order);
}

void split(
        Func& func,
        Image<float> filter_weights,
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

    int num_splits = var.size();

    vector<Expr> tile_width;
    vector<RDom> outer_rdom;
    vector<Expr> image_width;
    vector<Expr> num_tiles;

    for (int i=0; i<num_splits; i++) {
        // individual tile boundaries
        Expr inner_rdom_extent = simplify(inner_rdom[i].x.extent());

        tile_width .push_back(inner_rdom_extent);
        image_width.push_back(rdom[i].x.extent());
        num_tiles  .push_back(image_width[i]/tile_width[i]);

        // tile width for splitting multiple scans in same dimension must be same
        for (int j=0; j<i-1; j++) {
            if (dimension[j] == dimension[i] && !equal(tile_width[i], tile_width[j])) {
                cerr << "Different tile widths specified for splitting same dimension" << endl;
                assert(false);
            }
        }

        // extent of reduction along dimensions to be split
        assert(extract_params_in_expr(rdom[i].x.extent()).size()==1 &&
                "RDom extent must have a single image parameter");

        // outer_rdom.x: over tail elems of prev tile to compute tail of current tile
        // outer_rdom.y: over all tail elements of current tile
        // outer_rdom.z: over all tiles
        outer_rdom.push_back(RDom(
                    0,order[i],
                    0,order[i],
                    1, simplify(num_tiles[i]-1),
                    "r"+var[i].name()+"o"));
    }

    // list of structs with all the info about split in each dimension
    vector<SplitInfo> split_info(F.args().size());

    // loop over all reduction definitions in reverse order and
    // populate the split_info struct with info on splitting each reduction
    for (int i=F.reductions().size()-1; i>=0; i--) {
        assert(F.reductions()[i].domain.defined() &&
                "Reduction definition has no reduction domain");

        int index = -1;

        // extract the RDom in the reduction definition and
        // compare with the reduction domain to be split
        for (int j=0; j<num_splits; j++) {
            if (F.reductions()[i].domain.same_as(rdom[j].domain())) {
                index = j;
            }
        }

        if (index < 0) {
            cerr << "Could not find a split for scan " << i << endl;
            assert(false);
        }

        SplitInfo s = split_info[ dimension[index] ];

        s.filter_order = order[index];
        s.filter_dim   = dimension[index];
        s.var          = var[index];
        s.inner_var    = inner_var[index];
        s.outer_var    = outer_var[index];
        s.image_width  = image_width[index];
        s.tile_width   = tile_width[index];
        s.num_tiles    = num_tiles[index];
        s.filter_weights = filter_weights;

        s.scan_id    .push_back(i);
        s.scan_stage .push_back(0);
        s.rdom       .push_back(rdom[index]);
        s.inner_rdom .push_back(inner_rdom[index]);
        s.outer_rdom .push_back(outer_rdom[index]);
        s.scan_causal.push_back(check_causal_scan(F, rdom[index], i, s.filter_dim));
        s.num_splits++;

        split_info[ dimension[index] ] = s;
    }

    // group scans in same dimension together
    // change the order of splits accordingly
    split_info = group_scans_by_dimension(F, split_info);

    // remove split_info structs for dimensions which are not split
    for (int i=0; i<split_info.size(); i++) {
        if (split_info[i].num_splits == 0) {
            split_info.erase(split_info.begin()+i);
            i--;
        }
    }

    // splits in each dimension have scan stages from 0 to max_scan_stage
    // change the stages number so that stages from second dimension
    // start after stages from first dimension
    for (int i=0, next_scan_stage=0; i<split_info.size(); i++) {
        int scan_stage_offset = next_scan_stage;
        for (int j=0; j<split_info[i].num_splits; j++) {
            split_info[i].scan_stage[j] += scan_stage_offset;
            next_scan_stage = std::max(split_info[i].scan_stage[j]+1, next_scan_stage);
        }
    }

    // compute the intra tile result
    Function F_intra = create_intra_tile_term(F, split_info);

    // create a function will hold the final result,
    // just a copy of the intra tile computation for now
    Function F_final = create_copy(F_intra, F.name() + DELIMITER + FINAL_TERM);

    // add scan stages to the intra tile computation
    // this is necessary for computing tails
    // not required for the final result
    add_intra_tile_scan_stages(F_intra, split_info);

    // compute the residuals from splits in each dimension
    vector<Function> F_deps = create_recursive_split(F_intra, split_info);

    // transfer results between scan stages in intra tile computation
    fix_intra_tile_scan_stages(F_intra);

    // add all the residuals to the final term
    add_all_residuals_to_final_result(F_final, F_deps, split_info);

    // change the original function to index into the final term computed here
    {
        vector<string> args = F.args();
        vector<Expr> values;
        vector<Expr> call_args;
        for (int i=0; i<F_final.args().size(); i++) {
            string arg = F_final.args()[i];
            call_args.push_back(Var(arg));
            for (int j=0; j<var.size(); j++) {
                if (arg == inner_var[j].name()) {
                    call_args[i] = substitute(arg, var[j]%tile_width[j], call_args[i]);
                } else if (arg == outer_var[j].name()) {
                    call_args[i] = substitute(arg, var[j]/tile_width[j], call_args[i]);
                }
            }
        }
        for (int i=0; i<F.outputs(); i++) {
            values.push_back(Call::make(F_final, call_args, i));
        }
        F.clear_all_definitions();
        F.define(args, values);
    }

    func = Func(F);
}
