#include "recfilter.h"
#include "recfilter_utils.h"
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
        int  dim   = split_info[i].filter_dim;
        int  order = split_info[i].filter_order;
        RDom rxi   = split_info[i].inner_rdom;
        RDom rxt   = split_info[i].tail_rdom;
        Expr tile  = split_info[i].tile_width;

        // new reduction to extract the tail of each split scan
        for (int j=0; j<split_info[i].num_splits; j++) {
            int  scan_id     = split_info[i].scan_id[j];
            bool scan_causal = split_info[i].scan_causal[j];

            vector<Expr> args      = reductions[scan_id].args;
            vector<Expr> call_args = reductions[scan_id].args;
            vector<Expr> values;

            // replace rxi by rxt (involves replacing rxi.x by rxt.x etc)
            // for the dimension undergoing scan, use a buffer of length filter_order
            // beyond tile boundary as LHS args and tile-1-rxt on RHS to extract
            // the tail elements
            for (int k=0; k<rxi.dimensions(); k++) {
                for (int u=0; u<args.size(); u++) {
                    if (expr_depends_on_var(args[u], rxi[dim].name())) {
                        args     [u] = simplify(tile + order*j+ rxt[dim]);
                        call_args[u] = simplify(scan_causal ? (tile-1-rxt[dim]) : rxt[dim]);
                    } else {
                        args[u]      = substitute(rxi[k].name(), rxt[k], args[u]);
                        call_args[u] = substitute(rxi[k].name(), rxt[k], call_args[u]);
                    }
                }
            }

            for (int k=0; k<reductions[scan_id].values.size(); k++) {
                values.push_back(Call::make(F_intra, call_args, k));
            }

            new_reductions[scan_id] = std::make_pair(args, values);
        }
    }

    // add extra update steps to copy tail of each scan to another buffer
    // that is beyond the bounds of the intra tile RVars
    for (int i=0; i<reductions.size(); i++) {
        F_intra.define_reduction(reductions[i].args, reductions[i].values);
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

    vector<string> pure_args = F.args();
    vector<Expr>   pure_values = F.values();
    vector<ReductionDefinition> reductions = F.reductions();

    // create the intra tile function, replace calls to original
    // function by the new intra tile function
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

    for (int i=0; i<split_info.size(); i++) {
        F_intra.clear_all_definitions();

        Var x    = split_info[i].var;
        Var xi   = split_info[i].inner_var;
        Var xo   = split_info[i].outer_var;
        RDom rx  = split_info[i].rdom;
        RDom rxi = split_info[i].inner_rdom;
        RDom rxo = split_info[i].outer_rdom;
        Expr tile_width  = split_info[i].tile_width;
        Expr image_width = split_info[i].image_width;
        int filter_dim   = split_info[i].filter_dim;
        int var_index    = -1;

        // replace x by xi,xo in LHS pure args
        // replace x by tile*xo + xi in RHS values
        for (int j=0; j<pure_args.size(); j++) {
            if (pure_args[j] == x.name()) {
                pure_args[j] = xi.name();
                var_index = j;
            }
        }
        assert(var_index >= 0);
        pure_args.insert(pure_args.begin()+var_index+1, xo.name());
        for (int j=0; j<pure_values.size(); j++) {
            pure_values[j] = substitute(x, tile_width*xo+xi, pure_values[j]);
        }
        F_intra.define(pure_args, pure_values);

        // replace rx by rxi,xo
        for (int k=0; k<reductions.size(); k++) {
            string rx_var = rx.x.name();
            RVar   rxi_var = rxi[filter_dim];
            for (int j=0; j<reductions[k].args.size(); j++) {
                reductions[k].args[j] = substitute(rx_var, rxi_var, reductions[k].args[j]);
            }
            reductions[k].args.insert(reductions[k].args.begin()+var_index+1, xo);
            for (int j=0; j<reductions[k].values.size(); j++) {
                Expr val = reductions[k].values[j];
                val = substitute(rx_var, rxi_var, val);
                val = substitute(x.name(), rxi_var, val);
                val = insert_arg_in_func_call(F_intra.name(), var_index+1, xo, val);
                reductions[k].values[j] = val;
            }
            F_intra.define_reduction(reductions[k].args, reductions[k].values);
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
        Var  xo         = split_info[i].outer_var;
        RDom rxi        = split_info[i].inner_rdom;
        Expr num_tiles  = split_info[i].num_tiles;
        Expr tile_width = split_info[i].tile_width;

        for (int j=0; j<split_info[i].num_splits; j++) {
            bool causal = split_info[i].scan_causal[j];
            int scan_id = split_info[i].scan_id[j];

            Expr boundary = (causal ? 0 : tile_width);
            Expr inner_tile_condition = (causal ? (xo==0) : (xo==num_tiles-1));

            int dim = -1;
            Expr arg;
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
                call_args.push_back(simplify(tile + order*k + xi));
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
    RDom rxo  = split_info.outer_rdom;
    int  dim  = split_info.filter_dim;
    int  order= split_info.filter_order;
    Expr tile = split_info.tile_width;
    Expr num_tiles = split_info.num_tiles;

    vector<Function> F_ctail;

    for (int k=0; k<split_info.num_splits; k++) {
        Function function(func_name + "_" + int_to_string(split_info.scan_id[k]));

        Image<float> weight = tail_weights(split_info, k);


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
        contents.ptr->split_info[i].inner_rdom = RDom();
        contents.ptr->split_info[i].outer_rdom = RDom();
        contents.ptr->split_info[i].tail_rdom  = RDom();
    }

    // populate tile size and number of tiles for each dimension
    for (map<string,Expr>::iterator it=dim_tile.begin(); it!=dim_tile.end(); it++) {
        bool found = false;
        string x = it->first;
        Expr tile_width = it->second;
        for (int j=0; !found && j<F.args().size(); j++) {
            if (F.args()[j] == x) {
                string name       = contents.ptr->split_info[j].var.name();
                Expr image_width  = contents.ptr->split_info[j].image_width;

                // set tile width and number of tiles
                contents.ptr->split_info[j].tile_width = tile_width;
                contents.ptr->split_info[j].num_tiles  = image_width / tile_width;

                // set inner var and outer var
                contents.ptr->split_info[j].inner_var = Var(name + "i");
                contents.ptr->split_info[j].outer_var = Var(name + "o");

                found = true;
            }
        }
        if (!found) {
            cerr << "Variable " << x << " does not correspond to any "
                << "dimension of the recursive filter " << F.name() << endl;
            assert(false);
        }
    }

    // inner RDom - has dimensionality equal to dimensions of the image
    // each dimension runs from 0 to tile width of the respective dimension
    vector<ReductionVariable> inner_scan_rvars;
    for (int i=0; i<contents.ptr->split_info.size(); i++) {
        Expr extent;
        if (contents.ptr->split_info[i].tile_width.defined()) {
            extent = contents.ptr->split_info[i].tile_width;
        } else {
            extent = 1;     // for non split dimensions
        }
        ReductionVariable r;
        r.min    = 0;
        r.extent = extent;
        r.var    = "r" + contents.ptr->split_info[i].var.name() + "i";
        inner_scan_rvars.push_back(r);
    }

    // populate the inner, outer and tail reduction domains to all dimensions
    for (map<string,Expr>::iterator it=dim_tile.begin(); it!=dim_tile.end(); it++) {
        bool found = false;
        string x = it->first;
        Expr tile_width = it->second;
        for (int j=0; !found && j<F.args().size(); j++) {
            if (F.args()[j] == x) {
                string name       = contents.ptr->split_info[j].var.name();
                Expr num_tiles    = contents.ptr->split_info[j].num_tiles;
                int  filter_order = contents.ptr->split_info[j].filter_order;

                // set inner rdom, same for all dimensions
                contents.ptr->split_info[j].inner_rdom = RDom(ReductionDomain(inner_scan_rvars));

                // same as inner rdom except that the extent of scan dimension
                // is filter order rather than tile width
                vector<ReductionVariable> inner_tail_rvars = inner_scan_rvars;
                inner_tail_rvars[j].var    = "r"+name+"t";
                inner_tail_rvars[j].min    = 0;
                inner_tail_rvars[j].extent = filter_order;
                contents.ptr->split_info[j].tail_rdom = RDom(ReductionDomain(inner_tail_rvars));

                // outer_rdom.x: over tail elems of prev tile to compute tail of current tile
                // outer_rdom.y: over all tail elements of current tile
                // outer_rdom.z: over all tiles
                contents.ptr->split_info[j].outer_rdom = RDom(0, filter_order, 0, filter_order,
                        1, num_tiles-1, "r"+name+"o");

                found = true;
            }
        }
        assert(found);
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

void RecFilter::split(Var x, Var y, Expr t) {
    map<string,Expr> dim_tile;
    dim_tile[x.name()] = t;
    dim_tile[y.name()] = t;
    split(dim_tile);
}

void RecFilter::split(Var x, Var y, Var z, Expr t) {
    map<string,Expr> dim_tile;
    dim_tile[x.name()] = t;
    dim_tile[y.name()] = t;
    dim_tile[z.name()] = t;
    split(dim_tile);
}

void RecFilter::split(vector<Var> vars, Expr t) {
    map<string,Expr> dim_tile;
    for (int i=0; i<vars.size(); i++) {
        dim_tile[vars[i].name()] = t;
    }
    split(dim_tile);
}
