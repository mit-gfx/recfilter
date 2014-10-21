#include "recfilter.h"
#include "recfilter_utils.h"
#include "coefficients.h"
#include "schedule.h"
#include "modifiers.h"

using namespace Halide;
using namespace Halide::Internal;

using std::string;
using std::cerr;
using std::endl;
using std::vector;
using std::map;
using std::set;
using std::pair;
using std::make_pair;

// -----------------------------------------------------------------------------

#define INTRA_TILE_RESULT      "Intra"
#define INTRA_TILE_TAIL_TERM   "Tail"
#define INTER_TILE_TAIL_SUM    "CTail"
#define COMPLETE_TAIL_RESIDUAL "TDeps"
#define FINAL_RESULT_RESIDUAL  "Deps"
#define FINAL_TERM             "Final"
#define SUB                    "Sub"
#define DASH                   '_'

// -----------------------------------------------------------------------------

static vector<SplitInfo> group_scans_by_dimension(Function F, vector<SplitInfo> split_info) {
    vector<string> args = F.args();
    vector<Expr>  values = F.values();
    vector<UpdateDefinition> updates = F.updates();

    // split info struct must contain info about each dimension
    assert(split_info.size() == args.size());

    vector<UpdateDefinition> new_updates;
    vector<SplitInfo>        new_split_info = split_info;

    // use all scans with dimension 0 first, then 1 and so on
    for (int i=0; i<split_info.size(); i++) {
        for (int j=0; j<split_info[i].num_splits; j++) {
            int curr = split_info[i].num_splits-1-j;
            int scan = split_info[i].scan_id[curr];
            new_updates.push_back(updates[scan]);
            new_split_info[i].scan_id[curr] = new_updates.size()-1;
        }
    }
    assert(new_updates.size() == updates.size());

    // reorder the update definitions as per the new order
    F.clear_all_definitions();
    F.define(args, values);
    for (int i=0; i<new_updates.size(); i++) {
        F.define_update(new_updates[i].args, new_updates[i].values);
    }

    return new_split_info;
}

// -----------------------------------------------------------------------------

static void move_init_to_update_def(Function F, vector<SplitInfo> split_info) {
    vector<string> pure_args = F.args();
    vector<Expr> values = F.values();
    vector<UpdateDefinition> updates = F.updates();

    // leave the pure def undefined
    {
        for (int i=0; i<split_info.size(); i++) {
            Var x  = split_info[i].var;
            Var xi = split_info[i].inner_var;
            Var xo = split_info[i].outer_var;

            // replace x by xi in LHS pure args
            // replace x by tile*xo+xi in RHS values
            for (int j=0; j<pure_args.size(); j++) {
                if (pure_args[j] == x.name()) {
                    pure_args[j] = xi.name();
                    pure_args.insert(pure_args.begin()+j+1, xo.name());
                }
            }
        }
        vector<Expr> undef_values(F.outputs(), FLOAT_UNDEF);
        F.clear_all_definitions();
        F.define(pure_args, undef_values);
    }

    // initialize the buffer in the first update def
    {
        cerr << F.name() << endl;
        vector<Expr> args;
        for (int j=0; j<pure_args.size(); j++) {
            args.push_back(Var(pure_args[j]));
        }
        for (int i=0; i<split_info.size(); i++) {
            Var  x    = split_info[i].var;
            Var  xo   = split_info[i].outer_var;
            Var  xi   = split_info[i].inner_var;
            RVar rxi  = split_info[i].inner_rdom[ split_info[i].filter_dim ];
            Expr tile = split_info[i].tile_width;
            for (int j=0; j<args.size(); j++) {
                args[j] = substitute(xi.name(), rxi, args[j]);
                cerr << args[j] << endl;
            }
            for (int j=0; j<values.size(); j++) {
                values[j] = substitute(xi.name(), rxi, values[j]);
                cerr << values[j] << endl;
            }
        }
        F.define_update(args, values);
    }

    // add all the other scans
    for (int i=0; i<updates.size(); i++) {
        F.define_update(updates[i].args, updates[i].values);
    }
}

// -----------------------------------------------------------------------------

static void extract_tails_from_each_scan(Function F_intra, vector<SplitInfo> split_info) {
    vector<string> pure_args = F_intra.args();
    vector<Expr> pure_values = F_intra.values();
    vector<UpdateDefinition> updates = F_intra.updates();

    // pure definitions remain unchanged
    F_intra.clear_all_definitions();
    F_intra.define(pure_args, pure_values);

    // new updates to be added for all the split updates
    map<int, std::pair< vector<Expr>, vector<Expr> > > new_updates;

    // create the new update definitions to extract the tail
    // of each scan that is split
    for (int i=0; i<split_info.size(); i++) {
        int  dim   = split_info[i].filter_dim;
        int  order = split_info[i].filter_order;
        RDom rxi   = split_info[i].inner_rdom;
        RDom rxt   = split_info[i].tail_rdom;
        Expr tile  = split_info[i].tile_width;

        // new update to extract the tail of each split scan
        for (int j=0; j<split_info[i].num_splits; j++) {
            int  scan_id     = split_info[i].scan_id[j];
            bool scan_causal = split_info[i].scan_causal[j];

            vector<Expr> args      = updates[scan_id].args;
            vector<Expr> call_args = updates[scan_id].args;
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

            for (int k=0; k<updates[scan_id].values.size(); k++) {
                values.push_back(Call::make(F_intra, call_args, k));
            }

            new_updates[scan_id] = make_pair(args, values);
        }
    }

    // add extra update steps to copy tail of each scan to another buffer
    // that is beyond the bounds of the intra tile RVars
    for (int i=0; i<updates.size(); i++) {
        F_intra.define_update(updates[i].args, updates[i].values);
        if (new_updates.find(i) != new_updates.end()) {
            vector<Expr> args   = new_updates[i].first;
            vector<Expr> values = new_updates[i].second;
            F_intra.define_update(args, values);
        }
    }
}

// -----------------------------------------------------------------------------

static void add_padding_to_avoid_bank_conflicts(Function F, vector<SplitInfo> split_info, bool flag) {
//    // find all the dimensions containing a scan
//    set<int> update_dimensions;
//    for (int i=0; i<F.updates().size(); i++) {
//        UpdateDefinition u = F.updates()[i];
//        ReductionDomain r = u.domain;
//        for (int j=0; j<u.args.size(); j++) {
//            for (int k=0; k<r.domain().size(); k++) {
//                if (expr_depends_on_var(u.args[j], r.domain()[k].var)) {
//                    update_dimensions.insert(j);
//                }
//            }
//        }
//    }
//
//    // bank conflicts expected if there are scans in multiple dimensions
//    bool bank_conflicts = (update_dimensions.size()>1);
//    if (!bank_conflicts) {
//        return;
//    }

    // map inner var of first dimension to tile width +
    // map inner vars of all other dimensions to 0
    map<string,Expr> var_val;
    for (int i=0; i<split_info.size(); i++) {
        if (split_info[i].num_splits>0) {
            string x = split_info[i].inner_var.name();
            if (var_val.empty()) {
                var_val[x] = split_info[i].tile_width;
                if (flag) {
                    var_val[x] += split_info[i].filter_order*split_info[i].num_splits+1;
                }
            } else {
                var_val[x] = 0;
            }
        }
    }

    vector<Expr> args;
    for (int i=0; i<F.args().size(); i++) {
        string a = F.args()[i];
        if (var_val.find(a) != var_val.end()) {
            args.push_back(var_val.find(a)->second);
        } else {
            args.push_back(Var(a));
        }
    }

    vector<Expr> values(F.outputs(), FLOAT_UNDEF);
    F.define_update(args, values);
}

// -----------------------------------------------------------------------------

static Function create_intra_tile_term(Function F, vector<SplitInfo> split_info) {
    Function F_intra(F.name() + DASH + INTRA_TILE_RESULT);

    // manipulate the pure def
    vector<string> pure_args   = F.args();
    vector<Expr>   pure_values = F.values();
    for (int i=0; i<split_info.size(); i++) {
        Var x            = split_info[i].var;
        Var xi           = split_info[i].inner_var;
        Var xo           = split_info[i].outer_var;
        RDom rx          = split_info[i].rdom;
        RDom rxi         = split_info[i].inner_rdom;
        Expr tile_width  = split_info[i].tile_width;
        Expr image_width = split_info[i].image_width;
        int filter_dim   = split_info[i].filter_dim;

        // replace x by xi in LHS pure args
        // replace x by tile*xo+xi in RHS values
        for (int j=0; j<pure_args.size(); j++) {
            if (pure_args[j] == x.name()) {
                pure_args[j] = xi.name();
                pure_args.insert(pure_args.begin()+j+1, xo.name());
            }
        }
        for (int j=0; j<pure_values.size(); j++) {
            pure_values[j] = substitute(x.name(), tile_width*xo+xi, pure_values[j]);
        }
        F_intra.clear_all_definitions();
        F_intra.define(pure_args, pure_values);
    }

    // split info object and split id for each scan
    vector< pair<int,int> > scan(F.updates().size());
    for (int i=0; i<split_info.size(); i++) {
        for (int j=0; j<split_info[i].num_splits; j++) {
            scan[ split_info[i].scan_id[j] ] = make_pair(i,j);
        }
    }

    // create the scans from the split info object
    vector<UpdateDefinition> updates;
    for (int i=0; i<scan.size(); i++) {
        SplitInfo s = split_info[ scan[i].first ];

        Var x            = s.var;
        Var xi           = s.inner_var;
        Var xo           = s.outer_var;
        RDom rx          = s.rdom;
        RDom rxi         = s.inner_rdom;
        Expr tile_width  = s.tile_width;
        Expr num_tiles   = s.num_tiles;
        Expr image_width = s.image_width;
        Expr border_expr = s.border_expr[ scan[i].second ];

        int filter_dim   = s.filter_dim;
        int filter_order = s.filter_order;
        bool causal      = s.scan_causal[ scan[i].second ];
        int  dimension   = -1;

        float feedfwd = s.feedfwd_coeff(i);
        vector<float> feedback(filter_order,0.0f);
        for (int j=0; j<s.feedback_coeff.height(); j++) {
            feedback[j] = s.feedback_coeff(i,j);
        }

        // update args: replace rx by the RVar of this dimension in rxi and xo
        // replace all other pure args with their respective RVar in rxi
        vector<Expr> args;
        for (int j=0; j<F.args().size(); j++) {
            string a = F.args()[j];
            if (a == x.name()) {
                if (causal) {
                    args.push_back(rxi[filter_dim]);
                } else {
                    args.push_back(tile_width-1-rxi[filter_dim]);
                }
                dimension = args.size()-1;
                args.push_back(xo);
            }
            else {
                bool found = false;
                for (int k=0; !found && k<split_info.size(); k++) {
                    if (a == split_info[k].var.name()) {
                        args.push_back(rxi[ split_info[k].filter_dim ]);
                        if (!equal(split_info[k].tile_width, 1)) {
                            args.push_back(split_info[k].outer_var);
                        }
                        found = true;
                    }
                }
                if (!found) {
                    args.push_back(Var(a));
                }
            }
        }
        assert(dimension >= 0);

        // border expression: replace all variables x in border_expr with xo*tile+rxi
        if (border_expr.defined()) {
            for (int j=0; j<F.args().size(); j++) {
                string a = F.args()[j];
                bool found = false;
                for (int k=0; !found && k<split_info.size(); k++) {
                    if (a == split_info[k].var.name()) {
                        Var  temp_xo  = split_info[k].outer_var;
                        RVar temp_rxi = rxi[ split_info[k].filter_dim ];
                        Expr temp_tile= split_info[k].tile_width;
                        border_expr = substitute(a, temp_tile*temp_xo+temp_rxi, border_expr);
                        found = true;
                    }
                }
            }
        }

        // update values: create the intra tile scans with special
        // borders for all tile on image boundary is border_expr as specified
        // border for all internal tiles is zero
        vector<Expr> values(F_intra.outputs());
        for (int j=0; j<values.size(); j++) {
            values[j] = feedfwd * Call::make(F_intra, args, j);

            for (int k=0; k<feedback.size(); k++) {
                if (feedback[k] != 0.0f) {
                    vector<Expr> call_args = args;
                    Expr first_tile = (causal ? (xo==0) : (xo==num_tiles-1));
                    if (causal) {
                        call_args[dimension] = max(call_args[dimension]-(k+1),0);
                    } else {
                        call_args[dimension] = min(call_args[dimension]+(k+1),tile_width-1);
                    }

                    // inner tiles must always be clamped to zero beyond tile borders
                    // tiles on the image border must be clamped as specified in
                    // RecFilter::addScan
                    if (border_expr.defined()) {
                        values[j] += feedback[k] * select(rxi[filter_dim]>k,
                                Call::make(F_intra,call_args,j),
                                select(first_tile, border_expr, FLOAT_ZERO));
                    } else {
                        values[j] += feedback[k] *
                            select(first_tile || rxi[filter_dim]>k,
                                Call::make(F_intra,call_args,j), FLOAT_ZERO);
                    }
                }
            }
            values[j] = simplify(values[j]);
        }
        F_intra.define_update(args, values);
    }

    return F_intra;
}

// -----------------------------------------------------------------------------

static Function create_copy(Function F, string func_name) {
    Function B(func_name);

    // same pure definition
    B.define(F.args(), F.values());

    // replace all calls to old function with the copy
    for (int i=0; i<F.updates().size(); i++) {
        UpdateDefinition r = F.updates()[i];
        vector<Expr> args;
        vector<Expr> values;
        for (int j=0; j<r.args.size(); j++) {
            args.push_back(substitute_func_call(F.name(), B, r.args[j]));
        }
        for (int j=0; j<r.values.size(); j++) {
            values.push_back(substitute_func_call(F.name(), B, r.values[j]));
        }
        B.define_update(args, values);
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
        Function function(func_name + DASH + int_to_string(split_info.scan_id[k]));

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
        Function function(func_name + DASH + int_to_string(split_info.scan_id[k])
                + DASH + SUB);

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

        // update definition
        {
            vector<Expr> args;
            vector<Expr> values;
            vector<Expr> call_args_curr_tile;
            vector< vector<Expr> > call_args_prev_tile(order);

            for (int i=0; i<F_tail[k].args().size(); i++) {
                string arg = F_tail[k].args()[i];
                if (arg == xo.name()) {
                    // replace xo by rxo.y or rxo.y-1 as tile idx,
                    if (split_info.scan_causal[k]) {
                        args.push_back(rxo.y);
                        call_args_curr_tile.push_back(rxo.y);
                        for (int j=0; j<order; j++) {
                            call_args_prev_tile[j].push_back(max(rxo.y-1,0));
                        }
                    }
                    else {
                        args.push_back(simplify(num_tiles-1-rxo.y));
                        call_args_curr_tile.push_back(num_tiles-1-rxo.y);
                        for (int j=0; j<order; j++) {
                            call_args_prev_tile[j].push_back(min(num_tiles-rxo.y,num_tiles-1));
                        }
                    }
                }
                else if (arg == xi.name()) {
                    // replace xi by rxo.x as tail element index in args and current tile term
                    // replace xi by order number as tail element in prev tile term
                    args.push_back(rxo.x);
                    call_args_curr_tile.push_back(rxo.x);
                    for (int j=0; j<order; j++) {
                        call_args_prev_tile[j].push_back(j);
                    }
                }
                else {
                    args.push_back(Var(arg));
                    call_args_curr_tile.push_back(Var(arg));
                    for (int j=0; j<order; j++) {
                        call_args_prev_tile[j].push_back(Var(arg));
                    }
                }
            }

            // multiply each tail element with its weight before adding
            for (int i=0; i<F_tail[k].outputs(); i++) {
                Expr prev_tile_expr = FLOAT_ZERO;
                for (int j=0; j<order; j++) {
                    prev_tile_expr += weight(tile-rxo.x-1, j) *
                        Call::make(function, call_args_prev_tile[j], i);
                }

                Expr val = Call::make(function, call_args_curr_tile, i) +
                    select(rxo.y>0, prev_tile_expr, FLOAT_ZERO);
                values.push_back(simplify(val));
            }

            function.define_update(args, values);
        }
        F_ctail.push_back(function);
    }

    assert(F_ctail.size() == split_info.num_splits);

    return F_ctail;
}

// -----------------------------------------------------------------------------

static vector<Function> wrap_complete_tail_term(
        vector<Function> F_ctail,
        SplitInfo split_info,
        string func_name)
{
    vector<Function> F_ctailw;

    for (int k=0; k<F_ctail.size(); k++) {
        Function function(func_name + DASH + int_to_string(split_info.scan_id[k]));

        // simply create a pure function that calls the completed tail
        vector<string> args;
        vector<Expr>   values;
        vector<Expr>   call_args;

        for (int i=0; i<F_ctail[k].args().size(); i++) {
            args.push_back(F_ctail[k].args()[i]);
            call_args.push_back(Var(F_ctail[k].args()[i]));
        }
        for (int i=0; i<F_ctail[k].outputs(); i++) {
            values.push_back(Call::make(F_ctail[k], call_args, i));
        }

        function.define(args, values);

        F_ctailw.push_back(function);
    }

    return F_ctailw;
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

    assert(split_info.num_splits == F_ctail.size());

    vector<Function> dependency_functions;

    for (int u=0; u<split_info.num_splits; u++) {
        Function function(func_name + DASH + int_to_string(split_info.scan_id[u]));

        // args are same as completed tail terms
        vector<string> args = F_ctail[0].args();
        vector<Expr> values(num_outputs, FLOAT_ZERO);

        // accumulate the completed tails of all the preceedings scans
        // the list F_ctail is in reverse order just as split_info struct
        for (int j=u+1; j<F_ctail.size(); j++) {

            // weight matrix for accumulating completed tail elements from scan u to scan j
            Image<float> weight  = tail_weights(split_info, j, u);

            // weight matrix for accumulating completed tail elements from scan u to scan j
            // for a tile that is clamped on all borders
            Image<float> c_weight = weight;
            if (!equal(split_info.border_expr[j], FLOAT_ZERO)) {
                c_weight = tail_weights(split_info, j, u, true);
            }

            // expressions for prev tile and checking for first tile for causal/anticausal scan j
            Expr first_tile = (split_info.scan_causal[j] ? (xo==0) : (xo==num_tiles-1));
            Expr prev_tile  = (split_info.scan_causal[j] ? max(xo-1,0) : min(xo+1, num_tiles-1));

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

                    // change the weight for the first tile only, only this tile is
                    // affected by clamping the image at all borders
                    if (split_info.scan_causal[j]!=split_info.scan_causal[u]) {
                        Expr first_tile_u = (split_info.scan_causal[u] ? (xo==0) : (xo==num_tiles-1));
                        wt = select(first_tile_u, cwt, wt);
                    }

                    values[i] += simplify(select(first_tile, FLOAT_ZERO, wt*val));
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

    // create functions that just reindex the completed tails
    // this seemingly redundant reindexing allows the application to
    // read the completed tail into shared memory in
    vector<Function> F_deps;

    for (int j=0; j<split_info.num_splits; j++) {
        Function function(func_name + DASH + int_to_string(split_info.scan_id[j]));
        vector<Expr> values;
        vector<Expr> call_args;
        for (int i=0; i<F_ctail[j].args().size(); i++) {
            call_args.push_back(Var(F_ctail[j].args()[i]));
        }
        for (int i=0; i<F_ctail[j].values().size(); i++) {
            values.push_back(Call::make(F_ctail[j], call_args, i));
        }
        function.define(F_ctail[j].args(), values);
        F_deps.push_back(function);
    }

    // accumulate all completed tails
    vector<Function> F_deps_sub;

    // accumulate contribution from each completed tail
    for (int j=0; j<split_info.num_splits; j++) {
        Function function(func_name + DASH + int_to_string(split_info.scan_id[j]) + DASH + SUB);

        int num_args    = F_deps[j].args().size();
        int num_outputs = F_deps[j].outputs();

        // args are same as completed tail terms
        vector<string> args = F_deps[j].args();
        vector<Expr> values(num_outputs, FLOAT_ZERO);

        // weight matrix for accumulating completed tail elements
        // of scan after applying only current scan
        // Image<float> weight = tail_weights(split_info, j);
        Image<float> weight(order, order);
        for (int u=0; u<order; u++) {
            for (int v=0; v<order; v++) {
                weight(u,v) = ((u+v<order) ? split_info.feedback_coeff(split_info.scan_id[j], u+v) : 0.0f);
            }
        }

        // size of tail is equal to filter order, accumulate all
        // elements of the tail
        for (int k=0; k<order; k++) {
            vector<Expr> call_args;
            for (int i=0; i<num_args; i++) {
                string arg = F_deps[j].args()[i];
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
                            weight(xi,k) * Call::make(F_deps[j], call_args, i),
                            FLOAT_ZERO);
                } else {
                    val = select(xo<num_tiles-1,
                            weight(simplify(tile-1-xi),k) * Call::make(F_deps[j], call_args, i),
                            FLOAT_ZERO);
                }
                values[i] = simplify(values[i] + val);
            }
        }
        function.define(args, values);

        F_deps_sub.push_back(function);
    }

    return F_deps_sub;
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

        // updates remain unaffected
        vector<UpdateDefinition> updates = F_tail[j].updates();

        // redefine tail
        F_tail[j].clear_all_definitions();
        F_tail[j].define(args, values);
        for (int i=0; i<updates.size(); i++) {
            F_tail[j].define_update(updates[i].args, updates[i].values);
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

    Var  yi   = split_info_prev.inner_var;
    Var  yo   = split_info_prev.outer_var;
    RDom ryi  = split_info_prev.inner_rdom;
    RDom ryt  = split_info_prev.tail_rdom;
    Expr num_tiles_prev = split_info_prev.num_tiles;

    // add the residual term of previous dimension to the completed
    // tail of current dimension
    for (int j=0; j<F_tail.size(); j++) {
        vector<string> pure_args = F_tail[j].args();
        vector<Expr> pure_values = F_tail[j].values();

        // each scan of prev dimension adds one residual
        for (int k=0; k<F_tail_prev.size(); k++) {

            // first scan the tail in the current dimension according using intra term
            Function F_tail_prev_scanned_sub(F_tail_prev[k].name() + DASH + x.name()
                    + DASH + int_to_string(split_info.scan_id[j]) + DASH + SUB);

            // pure def simply calls the completed tail
            vector<Expr> prev_call_args;
            vector<Expr> prev_values;
            for (int i=0; i<F_tail_prev[k].args().size(); i++) {
                prev_call_args.push_back(Var(F_tail_prev[k].args()[i]));
            }
            for (int i=0; i<F_tail_prev[k].outputs(); i++) {
                prev_values.push_back(Call::make(F_tail_prev[k], prev_call_args, i));
            }
            F_tail_prev_scanned_sub.define(F_tail_prev[k].args(), prev_values);

            // apply all scans between the prev dimension and current dimension
            int first_scan = split_info_prev.scan_id[0]+1;
            int last_scan  = split_info.scan_id[j];

            for (int i=first_scan; i<=last_scan; i++) {
                vector<Expr> args;
                vector<Expr> values;
                for (int u=0; u<F_intra.updates()[i].args.size(); u++) {
                    Expr a = F_intra.updates()[i].args[u];
                    for (int v=0; v<ryi.dimensions(); v++) {
                        a = substitute(ryi[v].name(), ryt[v], a);
                    }
                    args.push_back(a);
                }
                for (int u=0; u<F_intra.updates()[i].values.size(); u++) {
                    Expr val = F_intra.updates()[i].values[u];
                    val = substitute_func_call(F_intra.name(), F_tail_prev_scanned_sub, val);
                    for (int v=0; v<ryi.dimensions(); v++) {
                        val = substitute(ryi[v].name(), ryt[v], val);
                    }
                    values.push_back(val);
                }
                F_tail_prev_scanned_sub.define_update(args, values);
            }

            // create a pure function as a wrapper for the above function
            // allows compute_at schedules
            Function F_tail_prev_scanned(F_tail_prev[k].name() + DASH + x.name()
                    + DASH + int_to_string(split_info.scan_id[j]));
            {
                vector<Expr> call_args;
                vector<Expr> values;
                for (int u=0; u<F_tail_prev_scanned_sub.args().size(); u++) {
                    call_args.push_back(Var(F_tail_prev_scanned_sub.args()[u]));
                }
                for (int u=0; u<F_tail_prev_scanned_sub.outputs(); u++) {
                    values.push_back(Call::make(F_tail_prev_scanned_sub, call_args, u));
                }
                F_tail_prev_scanned.define(F_tail_prev_scanned_sub.args(), values);
            }

            // weight matrix for accumulating completed tail elements
            // of scan after applying all subsequent scans
            Image<float> weight  = tail_weights(split_info_prev, k, 0);

            // weight matrix for accumulating completed tail elements
            // of scan after applying all subsequent scans
            // for a tile that is clamped on all borders
            Image<float> c_weight = weight;
            if (!equal(split_info_prev.border_expr[k], FLOAT_ZERO)) {
                c_weight= tail_weights(split_info_prev, k, 0, true);
            }

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
                            call_args.push_back(max(yo-1,0));
                        } else {
                            call_args.push_back(min(yo+1,num_tiles));
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
                    Expr val = Call::make(F_tail_prev_scanned, call_args, i);
                    Expr wt  = (split_info_prev.scan_causal[k] ? weight(yi,o)  : weight(tile-1-yi,o));
                    Expr cwt = (split_info_prev.scan_causal[k] ? c_weight(yi,o): c_weight(tile-1-yi,o));

                    // change the weight for the first tile only, only this tile is affected
                    // by clamping the image at all borders
                    wt = select(last_tile, cwt, wt);

                    pure_values[i] += simplify(select(first_tile, FLOAT_ZERO, wt*val));
                }
            }
        }

        // update defs of the tail remain unaffected
        vector<UpdateDefinition> updates = F_tail[j].updates();

        // redefine tail
        F_tail[j].clear_all_definitions();
        F_tail[j].define(pure_args, pure_values);
        for (int i=0; i<updates.size(); i++) {
            F_tail[j].define_update(updates[i].args, updates[i].values);
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
    vector<UpdateDefinition> updates = F.updates();

    assert(split_info.size() == F_deps.size());

    // new updates to be added for all the split updates
    map<int, std::pair< vector<Expr>, vector<Expr> > > new_updates;

    // create the new update definition for each scan that add the scans
    // residual to its first k elements (k = filter order)
    for (int i=0; i<F_deps.size(); i++) {
        Expr tile_width = split_info[i].tile_width;
        Expr num_tiles  = split_info[i].num_tiles;

        for (int j=0; j<F_deps[i].size(); j++) {
            int  curr_scan   = split_info[i].scan_id[j];
            RDom rxi         = split_info[i].inner_rdom;
            RDom rxt         = split_info[i].tail_rdom;
            RDom rxf         = split_info[i].truncated_inner_rdom;
            vector<Expr> args= updates[curr_scan].args;

            vector<Expr> values;
            for (int k=0; k<F_deps[i][j].outputs(); k++) {
                values.push_back(updates[curr_scan].values[k] + Call::make(F_deps[i][j], args, k));
            }

            for (int k=0; k<rxi.dimensions(); k++) {
                // replace rxi by rxt (involves replacing rxi.x by rxt.x etc)
                // for the dimension undergoing scan
                for (int u=0; u<args.size(); u++) {
                    args[u] = substitute(rxi[k].name(), rxt[k], args[u]);
                }
                for (int u=0; u<values.size(); u++) {
                    values[u] = substitute(rxi[k].name(), rxt[k], values[u]);
                }

                // the new update runs the scan for the first t elements
                // change the reduction domain of the original update to
                // run from t onwards, t = filter order
                for (int u=0; u<updates[curr_scan].args.size(); u++) {
                    updates[curr_scan].args[u] = substitute(rxi[k].name(), rxf[k], updates[curr_scan].args[u]);
                }
                for (int u=0; u<updates[curr_scan].values.size(); u++) {
                    updates[curr_scan].values[u] = substitute(rxi[k].name(), rxf[k], updates[curr_scan].values[u]);
                }
            }

            new_updates[curr_scan] = make_pair(args, values);
        }
    }

    // add extra update steps
    F.clear_all_definitions();
    F.define(pure_args, pure_values);
    for (int i=0; i<updates.size(); i++) {
        if (new_updates.find(i) != new_updates.end()) {
            vector<Expr> args   = new_updates[i].first;
            vector<Expr> values = new_updates[i].second;
            F.define_update(args, values);
        }
        F.define_update(updates[i].args, updates[i].values);
    }

    // add padding to avoid bank conflicts
    add_padding_to_avoid_bank_conflicts(F, split_info, false);

    // move initialization to update def in intra tile function
    move_init_to_update_def(F, split_info);
}

// -----------------------------------------------------------------------------

static vector< vector<Function> > split_scans(
        Function F_intra,
        vector<SplitInfo> &split_info)
{
    vector< vector<Function> > F_ctail_list;
    vector< vector<Function> > F_deps_list;

    /// TODO: this order of Function generation requires that F_ctail are
    /// redefined after they are called in F_tdeps, change the order of
    /// Function generation to fix this.

    for (int i=0; i<split_info.size(); i++) {
        string x = split_info[i].var.name();

        string s0 = F_intra.name() + DASH + INTRA_TILE_TAIL_TERM  + DASH + x;
        string s1 = F_intra.name() + DASH + INTER_TILE_TAIL_SUM   + DASH + x;
        string s2 = F_intra.name() + DASH + COMPLETE_TAIL_RESIDUAL+ DASH + x;
        string s3 = F_intra.name() + DASH + FINAL_RESULT_RESIDUAL + DASH + x;

        vector<Function> F_tail   = create_intra_tail_term    (F_intra,  split_info[i], s0);
        vector<Function> F_ctail  = create_complete_tail_term (F_tail,   split_info[i], s1);
        vector<Function> F_ctailw = wrap_complete_tail_term   (F_ctail,  split_info[i], s1);
        vector<Function> F_tdeps  = create_tail_residual_term (F_ctail,  split_info[i], s2);
        vector<Function> F_deps   = create_final_residual_term(F_ctailw, split_info[i], s3);

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

        F_ctail_list.push_back(F_ctailw);
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
    RDom inner_rdom = RDom(ReductionDomain(inner_scan_rvars));

    // populate the inner, outer and tail update domains to all dimensions
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
                contents.ptr->split_info[j].inner_rdom = inner_rdom;

                // same as inner rdom except that the extent of scan dimension
                // is filter order rather than tile width
                vector<ReductionVariable> inner_tail_rvars = inner_scan_rvars;
                inner_tail_rvars[j].var    = "r"+name+"t";
                inner_tail_rvars[j].min    = 0;
                inner_tail_rvars[j].extent = filter_order;
                contents.ptr->split_info[j].tail_rdom = RDom(ReductionDomain(inner_tail_rvars));

                // same as inner rdom except that the domain is from filter_order to tile_width-1
                // instead of 0 to tile_width-1
                vector<ReductionVariable> inner_truncated_rvars = inner_scan_rvars;
                inner_truncated_rvars[j].var    = "r"+name+"f";
                inner_truncated_rvars[j].min    = filter_order;
                inner_truncated_rvars[j].extent = simplify(max(inner_truncated_rvars[j].extent-filter_order,0));
                contents.ptr->split_info[j].truncated_inner_rdom = RDom(ReductionDomain(inner_truncated_rvars));

                // outer_rdom.x: over all tail elements of current tile
                // outer_rdom.y: over all tiles
                contents.ptr->split_info[j].outer_rdom = RDom(0, filter_order,
                        0, num_tiles, "r"+name+"o");

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

        // create a function will hold the final result, copy of the intra tile term
        F_final = create_copy(F_intra, F.name() + DASH + FINAL_TERM);

        // compute the residuals from splits in each dimension
        vector< vector<Function> > F_deps = split_scans(F_intra, split_info_current);

        // transfer the tail of each scan to another buffer
        extract_tails_from_each_scan(F_intra, split_info_current);

        // add all the residuals to the final term
        add_all_residuals_to_final_result(F_final, F_deps, split_info_current);

        // add padding to intra tile terms to avoid bank conflicts
        add_padding_to_avoid_bank_conflicts(F_intra, split_info_current, true);

        // move initialization to update def in intra tile function
        move_init_to_update_def(F_intra, split_info_current);

        // inline all residual functions not required any more
        map<string,Function> func_map = find_direct_calls(F_final);
        map<string,Function>::iterator f = func_map.begin();
        for (; f!=func_map.end(); f++) {
            for (int i=0; i<F_deps.size(); i++) {
                for (int j=0; j<F_deps[i].size(); j++) {
                    if (!F_deps[i][j].same_as(f->second)) {
                        inline_func(Func(F_deps[i][j]), Func(f->second));
                    }
                }
            }
        }
    }

    // add the generated final term
    add_generated_func(F);

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
}

void RecFilter::split(Expr tx) {
    map<string,Expr> dim_tile;
    for (int i=0; i<contents.ptr->split_info.size(); i++) {
        if (contents.ptr->split_info[i].num_splits>0) {
            Var x = contents.ptr->split_info[i].var;
            dim_tile[x.name()] = tx;
        }
    }
    split(dim_tile);
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
