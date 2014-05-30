#include "split.h"
#include "split_utils.h"

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
    for (int i=0; i<split_info.size(); i++) {
        dim_to_tile_width [split_info[i].filter_dim]   = split_info[i].tile_width;
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

    // create the split function and add the scan stage arg
    {
        int scan_stage_var_index = pure_args.size();

        // add the scan stage arg to pure args
        pure_args.push_back(xs.name());

        // set pure values to zero for all scan stages other than 0
        for (int i=0; i<pure_values.size(); i++) {
            pure_values[i] = select(xs==0, pure_values[i], 0);
        }
        Fsplit.define(pure_args, pure_values);

        for (int i=0; i<reductions.size(); i++) {
            // find the actual scan stage for this reduction
            int scan_stage = -1;
            for (int j=0; j<split_info.size(); j++) {
                if (split_info[j].scan_id == i) {
                    scan_stage = split_info[j].scan_stage;
                }
            }
            if (scan_stage < 0) {
                cerr << "Split operation not defined for all scans" << endl;
                assert(false);
            }

            // add scan stage value as arg to reductions, so that each
            // reduction computes at its own stage
            reductions[i].args.push_back(scan_stage);

            for (int j=0; j<reductions[i].values.size(); j++) {
                Expr value = reductions[i].values[j];

                // replace calls to original Func by split Func
                value = substitute_func_call(F.name(), Fsplit, value);

                // add scan stage value as arg to reduction args and recursive calls
                value = insert_arg_to_func_call(Fsplit.name(),
                        scan_stage_var_index, scan_stage, value);

                // transfer data from one stage of computation to another
                // since each stage computes in its own buffer
                if (i>0) {
                    Expr prev_scan_stage = reductions[i-1].args[scan_stage_var_index];
                    if (!equal(prev_scan_stage,scan_stage)) {
                        vector<Expr> call_args;
                        for (int k=0; k<reductions[i].args.size(); k++) {
                            if (equal(reductions[i].args[k], scan_stage)) {
                                call_args.push_back(prev_scan_stage);
                            } else {
                                call_args.push_back(reductions[i].args[k]);
                            }
                        }
                        value = Call::make(Fsplit, call_args, j) + value;
                    }
                }
                reductions[i].values[j] = value;
            }
            Fsplit.define_reduction(reductions[i].args, reductions[i].values);
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

        // redefine Fsplit after splitting each dimension
        Fsplit.clear_all_definitions();

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

            Fsplit.define(pure_args, pure_values);
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
                value = substitute_func_call(F.name(), Fsplit, value);
                value = insert_arg_to_func_call(Fsplit.name(), var_index+1, rxo, value);
                value = substitute(rx, rxi, value);
                value = substitute(image_width, tile_width, value);
                reductions[i].values[j] = value;
            }

            Fsplit.define_reduction(reductions[i].args, reductions[i].values);
        }
    }

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
    for (int i=0; i<F.reductions().size(); i++) {
        vector<Expr> args = F.reductions()[i].args;
        vector<Expr> values = F.reductions()[i].values;

        // replace r2D.outer_rvar by xo in args
        // replace r2D.inner_rvar by rxi in args
        for (int j=0; j<args.size(); j++) {
            args[j] = substitute(r2D.x.name(), rxi, args[j]);
            args[j] = substitute(r2D.y.name(), xo, args[j]);
        }

        // change calls to original Func by intra tile Func
        // replace outer RDom by xo in args
        for (int j=0; j<values.size(); j++) {
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
    // - xi replaced by causal/anticausal tail indices
    // - scan stage arg set to actual scan stage
    for (int i=0; i<F_intra.args().size(); i++) {
        if (SCAN_STAGE_ARG == F_intra.args()[i]) {
            call_args.push_back(split_info.scan_stage);
        }
        else if (xi.name() == F_intra.args()[i]) {
            args.push_back(xi.name());
            if (split_info.scan_causal) {
                call_args.push_back(tile-xi-1);
            } else {
                call_args.push_back(xi);
            }
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
    Expr num_tiles = split_info.num_tiles;

    Image<float> weight = split_info.complete_tail_weight;

    // pure definition
    {
        // pure definition simply initializes with the intra tile tail
        vector<Expr> call_args;
        vector<Expr> values;

        for (int i=0; i<F_tail.args().size(); i++) {
            call_args.push_back(Var(F_tail.args()[i]));
        }
        for (int i=0; i<F_tail.outputs(); i++) {
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

        for (int i=0; i<F_tail.args().size(); i++) {
            string arg = F_tail.args()[i];
            if (arg == xo.name()) {
                // replace xo by rxo.z or rxo.z-1 as tile idx,
                if (split_info.scan_causal) {
                    args.push_back(rxo.z);
                    call_args_curr_tile.push_back(rxo.z);
                    call_args_prev_tile.push_back(
                            max(simplify(rxo.z-1), 0));
                } else {
                    args.push_back(num_tiles-1-rxo.z);
                    call_args_curr_tile.push_back(num_tiles-1-rxo.z);
                    call_args_prev_tile.push_back(
                            min(simplify(num_tiles-rxo.z), simplify(num_tiles-1)));
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
        for (int i=0; i<F_tail.outputs(); i++) {
            values.push_back(
                    Call::make(function, call_args_curr_tile, i) +
                    select(rxo.z>0, weight(simplify(tile-rxo.y-1), rxo.x) *
                        Call::make(function, call_args_prev_tile, i), 0));
        }

        function.define_reduction(args, values);
    }

    return function;
}

// -----------------------------------------------------------------------------

static void create_recursive_split(Function F, SplitInfo& split_info) {
    int  dim  = split_info.filter_dim;
    int  order= split_info.filter_order;
    Var  x    = split_info.var;
    Var  xi   = split_info.inner_var;
    Var  xo   = split_info.outer_var;
    Expr tile = split_info.tile_width;
    Expr num_tiles = split_info.num_tiles;

    Image<float> weight = split_info.complete_result_weight;

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
        vector<string> args;
        for (int i=0; i<F_intra.args().size(); i++) {
            if (F_intra.args()[i] != SCAN_STAGE_ARG)
                args.push_back(F_intra.args()[i]);
        }
        vector<Expr> values(F.outputs());
        for (int k=0; k<order; k++) {
            vector<Expr> call_args;
            for (int i=0; i<F_ctail.args().size(); i++) {
                string arg = F_ctail.args()[i];
                if (xi.name() == arg)
                    call_args.push_back(k);
                else
                    call_args.push_back(Var(arg));
            }
            for (int i=0; i<F_ctail.outputs(); i++) {
                Expr val;
                if (split_info.scan_causal) {
                    val = weight(xi,k) * Call::make(F_ctail, call_args, i);
                } else {
                    val = weight(tile-1-xi,k) * Call::make(F_ctail, call_args, i);
                }
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
        for (int i=0; i<F_intra.args().size(); i++) {
            string arg = F_intra.args()[i];
            if (xo.name() == arg) {
                intra_call_args.push_back(xo);
                if (split_info.scan_causal) {
                    inter_call_args.push_back(
                            max(simplify(xo-1),0));  // prev tile
                } else {
                    inter_call_args.push_back(
                            min(simplify(xo+1), simplify(num_tiles-1)));
                }
            } else {
                intra_call_args.push_back(Var(arg));
                if (arg != SCAN_STAGE_ARG) {
                    inter_call_args.push_back(Var(arg));
                }
            }
        }
        for (int i=0; i<F.outputs(); i++) {
            if (split_info.scan_causal) {
                Expr val = Call::make(F_intra, intra_call_args, i) +
                    select(xo>0, Call::make(F_inter_deps, inter_call_args, i), 0);
                values.push_back(val);
            } else {
                Expr val = Call::make(F_intra, intra_call_args, i) +
                    select(xo<num_tiles-1, Call::make(F_inter_deps, inter_call_args, i), 0);
                values.push_back(val);
            }
        }

        // redefine original Func to point to calls to split Func
        F.clear_all_definitions();
        F.define(args, values);
    }

    // extra functions generated
    split_info.intra_tile_scan = F_intra;
    split_info.incomplete_tail = F_tail;
    split_info.complete_tail   = F_ctail;
    split_info.dependencies    = F_inter_deps;
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
    vector<RDom> split_rdom;
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

        // 2D RDom which contains intra tile and inter tile RDom as two dimensions
        split_rdom.push_back(RDom(inner_rdom[i].x.min(), inner_rdom[i].x.extent(),
               outer_rdom[i].z.min(), outer_rdom[i].z.extent(), rdom[i].x.name()));
    }

    // list of structs with all the info about each split
    vector<SplitInfo> split_info;

    // reorder the splits such that the last scan is split first
    // loop over all reduction definitions in reverse order and split
    // them if needed
    for (int i=F.reductions().size()-1; i>=0; i--) {
        assert(F.reductions()[i].domain.defined() &&
                "Reduction definition has no reduction domain");

        int split_id = -1;

        // extract the RDom in the reduction definition and
        // compare with the reduction domain to be split
        for (int j=0; j<num_splits; j++) {
            if (F.reductions()[i].domain.same_as(rdom[j].domain())) {
                split_id = j;
            }
        }

        if (split_id < 0) {
            cerr << "Could not find a split for scan " << i << endl;
            assert(false);
        }

        SplitInfo s;
        s.scan_id       = i;
        s.scan_causal   = true;            // default, change it in next section
        s.scan_stage    = 0;               // default, change it in next section
        s.filter_order  = order[split_id];
        s.filter_dim    = dimension[split_id];
        s.var           = var[split_id];
        s.rdom          = rdom[split_id];
        s.inner_var     = inner_var[split_id];
        s.outer_var     = outer_var[split_id];
        s.inner_rdom    = inner_rdom[split_id];
        s.outer_rdom    = outer_rdom[split_id];
        s.split_rdom    = split_rdom[split_id];
        s.image_width   = image_width[split_id];
        s.tile_width    = tile_width[split_id];
        s.num_tiles     = num_tiles[split_id];

        // construct the matrix of weight coefficients for completing tails
        const int* tile_width_ptr = as_const_int(s.tile_width);
        if (!tile_width_ptr) {
            cerr << "Could not convert tile width expression "
                 << s.tile_width << " to integer" << endl;
            assert(false);
        }

        Image<float> A_FP = weight_matrix_A_FP(filter_weights,
                s.scan_id, *tile_width_ptr);
        s.complete_tail_weight = weight_matrix_transpose(A_FP);

        // check if there was another scan in the same image dimension
        // accummulate weight coefficients because of all such scans
        for (int j=split_info.size()-1; j>=0; j--) {
            if (s.filter_dim == split_info[j].filter_dim) {
                Image<float> A_FB = weight_matrix_A_FB(filter_weights,
                        split_info[j].scan_id, *tile_width_ptr);
                cerr << A_FB << endl;
                A_FP = weight_matrix_mult(A_FB, A_FP);
            }
        }

        // last row of the weight matrix is the coefficients for tail elements
        s.complete_result_weight = weight_matrix_transpose(A_FP);
        cerr << s.complete_tail_weight << endl;
        cerr << s.complete_result_weight << endl;

        split_info.push_back(s);
    }


    // go to all dimensions, find all scans in that dimension and assign scan stages
    // all causal scans in same dimension must be computed in different stages so that
    // they don't stomp on each other's tails; same for anticausal scans
    for (int d=0; d<F.args().size(); d++) {
        int num_causal_scans = 0;
        int num_anticausal_scans = 0;
        for (int i=split_info.size()-1; i>=0; i--) {    // reverse traversal because first
            if (split_info[i].filter_dim == d) {        // scan will appear last in split_info list
                if (check_causal_scan(F, split_info[i].rdom.x, split_info[i].scan_id, d)) {
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

    // change the order of applying splits - group splits in same dimension together
    split_info = group_scans_by_dimension(F, split_info);

    // change the order of the scans - group similar scan stages together
    split_info = group_scans_by_stage(F, split_info);

    // create a function whose dimensions are split without
    // affecting computation
    Function Fsplit = split_function_dimensions(F, split_info);

    int max_scan_stage = -1;

    for (int i=0; i<split_info.size(); i++) {
        max_scan_stage = std::max(max_scan_stage, split_info[i].scan_stage);

        Function function_to_split;

        // operate on Fsplit for the first split, subsequent splits operate on
        // result of previous split
        if (i==0) {
            function_to_split = Fsplit;
        } else {
            function_to_split = split_info[i-1].intra_tile_scan;
        }

        // check if there is a reduction in the current dimension
        if (check_for_pure_split(function_to_split, split_info[i])) {
            continue;
        }

        create_recursive_split(function_to_split, split_info[i]);
    }

    // modify the original function to index into the split function
    {
        vector<string> args = F.args();
        vector<Expr> call_args(Fsplit.args().size());
        vector<Expr> values(F.outputs());
        for (int i=0; i<call_args.size(); i++) {
            string arg = Fsplit.args()[i];
            if (arg == SCAN_STAGE_ARG) {
                call_args[i] = max_scan_stage;
            } else {
                call_args[i] = Var(arg);
            }
            for (int j=0; j<var.size(); j++) {
                if (inner_var[j].name() == arg) {
                    call_args[i] = simplify(var[j] % tile_width[j]);
                }
                else if (outer_var[j].name() == arg) {
                    call_args[i] = simplify(var[j] / tile_width[j]);
                }
            }
        }
        for (int i=0; i<F.outputs(); i++) {
            values[i] = Call::make(Fsplit, call_args, i);
        }
        F.clear_all_definitions();
        F.define(args, values);
    }

    func = Func(F);

    // inline the helper function F-split because it is no longer required
    inline_function(func, Fsplit.name());
}
