#include "recfilter.h"
#include "recfilter_internals.h"
#include "coefficients.h"
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
using std::stringstream;

// -----------------------------------------------------------------------------

/** String constants used to construct names of intermediate functions generated during tiling */
// {@
#define INTRA_TILE_RESULT      "Intra"
#define INTRA_TILE_TAIL_TERM   "Tail"
#define INTER_TILE_TAIL_SUM    "CTail"
#define COMPLETE_TAIL_RESIDUAL "TDeps"
#define FINAL_RESULT_RESIDUAL  "Deps"
#define FINAL_TERM             "Final"
#define SUB                    "Sub"
#define DASH                   '_'
// @}

// -----------------------------------------------------------------------------

/** Info required to split a particular dimension of the recursive filter */
struct SplitInfo {
    int filter_order;                   ///< order of recursive filter in a given dimension
    int filter_dim;                     ///< dimension id
    int num_scans;                      ///< number of scans in the dimension that must be tiled
    bool clamped_border;                ///< Image border expression (from RecFilterContents)

    int tile_width;                     ///< tile width for splitting
    int image_width;                    ///< image width in this dimension
    int num_tiles;                      ///< number of tile in this dimension

    Halide::Type type;                  ///< filter output type
    Halide::Var  var;                   ///< variable that represents this dimension
    Halide::Var  inner_var;             ///< inner variable after splitting
    Halide::Var  outer_var;             ///< outer variable or tile index after splitting

    Halide::RDom rdom;                  ///< RDom update domain of each scan
    Halide::RDom inner_rdom;            ///< inner RDom of each scan
    Halide::RDom truncated_inner_rdom;  ///< inner RDom width a truncated
    Halide::RDom outer_rdom;            ///< outer RDom of each scan
    Halide::RDom tail_rdom;             ///< RDom to extract the tail of each scan

    std::vector<bool> scan_causal;      ///< causal or anticausal flag for each scan
    std::vector<int>  scan_id;          ///< scan or update definition id of each scan

    Halide::Image<double> feedfwd_coeff; ///< Feedforward coeffs (from RecFilterContents)
    Halide::Image<double> feedback_coeff;///< Feedback coeffs  (from RecFilterContents)
};

/** Tiling info for each dimension of the filter */
static vector<SplitInfo> recfilter_split_info;

/** All recursive filter funcs created during splitting transformations */
static map<string, RecFilterFunc> recfilter_func_list;

// -----------------------------------------------------------------------------


/** @brief Weight coefficients (tail_size x tile_width) for
 * applying scans corresponding to split indices split_id1 to
 * split_id2 in the SplitInfo struct (defined in coefficients.cpp).
 * It is meaningful to apply subsequent scans on the tail of any scan
 * as it undergoes other scans only if they happen after the first
 * scan. The SpliInfo object stores the scans in reverse order, hence indices
 * into the SplitInfo object split_id1 and split_id2 must be decreasing
 */
Image<double> tail_weights(SplitInfo s, int split_id1, int split_id2, bool clamp_border=false) {
    assert(split_id1 >= split_id2);

    int  tile_width  = s.tile_width;
    int  scan_id     = s.scan_id[split_id1];
    bool scan_causal = s.scan_causal[split_id1];

    Image<double> R = matrix_R(s.feedback_coeff, scan_id, tile_width);

    // accummulate weight coefficients because of all subsequent scans
    // traversal is backwards because SplitInfo contains scans in the
    // reverse order
    for (int j=split_id1-1; j>=split_id2; j--) {
        if (scan_causal != s.scan_causal[j]) {
            Image<double> B = matrix_B(s.feedfwd_coeff,
                    s.feedback_coeff, s.scan_id[j], tile_width, clamp_border);
            Image<double> I = matrix_antidiagonal(R.height());
            R = matrix_mult(I, R);
            R = matrix_mult(B, R);
            R = matrix_mult(I, R);
        }
        else {
            Image<double> B = matrix_B(s.feedfwd_coeff,
                    s.feedback_coeff, s.scan_id[j], tile_width, false);
            R = matrix_mult(B, R);
        }
    }

    return matrix_transpose(R);
}

/** @brief Weight coefficients (tail_size x tile_width) for
 * applying scan's corresponding to split indices split_id1
 */
Image<double> tail_weights(SplitInfo s, int split_id1, bool clamp_border=false) {
    return tail_weights(s, split_id1, split_id1, clamp_border);
}

// -----------------------------------------------------------------------------

static vector<FilterInfo> group_scans_by_dimension(Function F, vector<FilterInfo> filter_info) {
    vector<string> args = F.args();
    vector<Expr>  values = F.values();
    vector<UpdateDefinition> updates = F.updates();

    vector<UpdateDefinition> new_updates;
    vector<FilterInfo>       new_filter_info = filter_info;

    // use all scans with dimension 0 first, then 1 and so on
    for (int i=0; i<filter_info.size(); i++) {
        for (int j=0; j<filter_info[i].num_scans; j++) {
            int curr = filter_info[i].num_scans-1-j;
            int scan = filter_info[i].scan_id[curr];
            new_updates.push_back(updates[scan]);
            new_filter_info[i].scan_id[curr] = new_updates.size()-1;
        }
    }
    assert(new_updates.size() == updates.size());

    // reorder the update definitions as per the new order
    F.clear_all_definitions();
    F.define(args, values);
    for (int i=0; i<new_updates.size(); i++) {
        F.define_update(new_updates[i].args, new_updates[i].values);
    }

    return new_filter_info;
}

// -----------------------------------------------------------------------------

static void move_init_to_update_def(RecFilterFunc& rF, vector<SplitInfo> split_info) {
    assert(!split_info.empty());

    Function F = rF.func;

    vector<string> pure_args = F.args();
    vector<Expr> values = F.values();
    vector<UpdateDefinition> updates = F.updates();

    // scheduling tags for the new update def are same as pure def
    map<string,VarTag> update_var_category = rF.pure_var_category;

    // leave the pure def undefined
    vector<Expr> undef_values(F.outputs(), undef(split_info[0].type));
    F.clear_all_definitions();
    F.define(pure_args, undef_values);

    // initialize the buffer in the first update def
    // replace the scheduling tags of xi by rxi
    {
        vector<Expr> args;
        for (int j=0; j<pure_args.size(); j++) {
            args.push_back(Var(pure_args[j]));
        }
        for (int i=0; i<split_info.size(); i++) {
            Var  xo   = split_info[i].outer_var;
            Var  xi   = split_info[i].inner_var;
            RVar rxi  = split_info[i].inner_rdom[ split_info[i].filter_dim ];
            for (int j=0; j<args.size(); j++) {
                args[j] = substitute(xi.name(), rxi, args[j]);
            }
            for (int j=0; j<values.size(); j++) {
                values[j] = substitute(xi.name(), rxi, values[j]);
            }
            update_var_category.insert(make_pair(rxi.name(), update_var_category[xi.name()]));
            update_var_category.erase(xi.name());
        }
        F.define_update(args, values);
    }

    // add all the other scans
    for (int i=0; i<updates.size(); i++) {
        F.define_update(updates[i].args, updates[i].values);
    }

    // add the scheduling tags for the new update def in front of tags for
    // all other update defs
    rF.update_var_category.insert(rF.update_var_category.begin(), update_var_category);
}

// -----------------------------------------------------------------------------

static void extract_tails_from_each_scan(RecFilterFunc& rF_intra, vector<SplitInfo> split_info) {
    Function F_intra = rF_intra.func;

    vector<string> pure_args = F_intra.args();
    vector<Expr> pure_values = F_intra.values();
    vector<UpdateDefinition> updates = F_intra.updates();
    vector< map<string,VarTag> > update_var_category = rF_intra.update_var_category;

    // pure definitions remain unchanged
    rF_intra.update_var_category.clear();
    F_intra.clear_all_definitions();
    F_intra.define(pure_args, pure_values);

    // new updates and scheduling tags for the new updates to be
    // added for all the split updates
    map<int, std::pair< vector<Expr>, vector<Expr> > > new_updates;
    map<int, map<string,VarTag> > new_update_var_category;

    // create the new update definitions to extract the tail
    // of each scan that is split
    for (int i=0; i<split_info.size(); i++) {
        int  dim   = split_info[i].filter_dim;
        int  order = split_info[i].filter_order;
        RDom rxi   = split_info[i].inner_rdom;
        RDom rxt   = split_info[i].tail_rdom;
        int  tile  = split_info[i].tile_width;

        // new update to extract the tail of each split scan
        for (int j=0; j<split_info[i].num_scans; j++) {
            int  scan_id     = split_info[i].scan_id[j];
            bool scan_causal = split_info[i].scan_causal[j];

            vector<Expr> args      = updates[scan_id].args;
            vector<Expr> call_args = updates[scan_id].args;
            vector<Expr> values;

            // copy the scheduling tags from the original update def
            new_update_var_category[scan_id] = update_var_category[scan_id];

            // replace rxi by rxt (involves replacing rxi.x by rxt.x etc) for the
            // dimension undergoing scan, use a buffer of length filter_order beyond
            // tile boundary as LHS args and tile-1-rxt on RHS to extract tail elements;
            // same replacement in scheduling tags
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
                VarTag vc = update_var_category[scan_id][rxi[k].name()];
                new_update_var_category[scan_id].erase(rxi[k].name());
                new_update_var_category[scan_id].insert(make_pair(rxt[k].name(), vc));
            }

            for (int k=0; k<updates[scan_id].values.size(); k++) {
                values.push_back(Call::make(F_intra, call_args, k));
            }

            new_updates[scan_id] = make_pair(args, values);
        }
    }

    // add extra update steps to copy tail of each scan to another buffer
    // that is beyond the bounds of the intra tile RVars
    // for each update def add the corresponding mapping of update var scheduling tags
    for (int i=0; i<updates.size(); i++) {
        F_intra.define_update(updates[i].args, updates[i].values);
        rF_intra.update_var_category.push_back(update_var_category[i]);
        if (new_updates.find(i) != new_updates.end()) {
            vector<Expr> args   = new_updates[i].first;
            vector<Expr> values = new_updates[i].second;
            F_intra.define_update(args, values);
            rF_intra.update_var_category.push_back(new_update_var_category[i]);
        }
    }
}

// -----------------------------------------------------------------------------

static void add_padding_to_avoid_bank_conflicts(RecFilterFunc rF, vector<SplitInfo> split_info, bool flag) {
    // bank conflicts expected if there are scans in multiple dimensions
    if (split_info.size()<=1) {
        return;
    }

    Function F = rF.func;

    // map inner var of first dimension to tile width +
    // map inner vars of all other dimensions to 0
    map<string,Expr> var_val;
    for (int i=0; i<split_info.size(); i++) {
        if (split_info[i].num_scans>0) {
            string x = split_info[i].inner_var.name();
            if (var_val.empty()) {
                var_val[x] = split_info[i].tile_width;
                if (flag) {
                    var_val[x] += split_info[i].filter_order*split_info[i].num_scans+1;
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

    vector<Expr> undef_values(F.outputs(), undef(split_info[0].type));
    F.define_update(args, undef_values);

    // no need for scheduling tags for the last update def
}

// -----------------------------------------------------------------------------

static RecFilterFunc create_intra_tile_term(RecFilterFunc rF, vector<SplitInfo> split_info) {

    assert(!split_info.empty());

    Function F = rF.func;
    Function F_intra(F.name() + DASH + INTRA_TILE_RESULT);

    // filter type
    Type type = split_info[0].type;

    // scheduling tags for function dimensions
    map<string,VarTag>          pure_var_category   = rF.pure_var_category;
    vector<map<string,VarTag> > update_var_category = rF.update_var_category;

    // manipulate the pure def
    vector<string> pure_args   = F.args();
    vector<Expr>   pure_values = F.values();
    for (int i=0, o_cnt=0, i_cnt=0; i<split_info.size(); i++) {
        Var x           = split_info[i].var;
        Var xi          = split_info[i].inner_var;
        Var xo          = split_info[i].outer_var;
        RDom rx         = split_info[i].rdom;
        RDom rxi        = split_info[i].inner_rdom;
        int tile_width  = split_info[i].tile_width;
        int image_width = split_info[i].image_width;
        int num_tiles   = split_info[i].num_tiles;
        int filter_dim  = split_info[i].filter_dim;

        // replace x by xi in LHS pure args
        // replace x by tile*xo+xi in RHS values
        for (int j=0; j<pure_args.size(); j++) {
            if (pure_args[j] == x.name()) {
                pure_args[j] = xi.name();
                pure_args.insert(pure_args.begin()+j+1, xo.name());
                pure_var_category.erase (x.name());
                pure_var_category.insert(make_pair(xi.name(), VarTag(INNER,i_cnt++)));
                pure_var_category.insert(make_pair(xo.name(), VarTag(OUTER,o_cnt++)));
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
        for (int j=0; j<split_info[i].num_scans; j++) {
            scan[ split_info[i].scan_id[j] ] = make_pair(i,j);
        }
    }

    // create the scans from the split info object
    vector<UpdateDefinition> updates;
    for (int i=0; i<scan.size(); i++) {
        SplitInfo s = split_info[ scan[i].first ];

        Var x           = s.var;
        Var xi          = s.inner_var;
        Var xo          = s.outer_var;
        RDom rx         = s.rdom;
        RDom rxi        = s.inner_rdom;
        int tile_width  = s.tile_width;
        int num_tiles   = s.num_tiles;
        int image_width = s.image_width;

        int filter_dim   = s.filter_dim;
        int filter_order = s.filter_order;
        bool causal      = s.scan_causal[ scan[i].second ];
        bool clamped_border = s.clamped_border;
        int  dimension   = -1;

        double feedfwd = s.feedfwd_coeff(i);
        vector<double> feedback(filter_order,0.0);
        for (int j=0; j<s.feedback_coeff.height(); j++) {
            feedback[j] = s.feedback_coeff(i,j);
        }

        // number of inner and outer vars in the update def
        int i_cnt = 0;
        int o_cnt = 0;

        // update args: replace rx by the RVar of this dimension in rxi and xo
        // replace all other pure args with their respective RVar in rxi
        vector<Expr> args;
        for (int j=0; j<F.args().size(); j++) {
            string a = F.args()[j];
            if (a == x.name()) {
                RVar rvar = rxi[filter_dim];
                if (causal) {
                    args.push_back(rvar);
                } else {
                    args.push_back(tile_width-1-rvar);
                }
                dimension = args.size()-1;
                args.push_back(xo);
                update_var_category[i].erase (rx.x.name());
                update_var_category[i].insert(make_pair(rvar.name(),INNER|SCAN));
                update_var_category[i].insert(make_pair(xo.name(),  VarTag(OUTER,o_cnt++)));
            }
            else {
                bool found = false;
                for (int k=0; !found && k<split_info.size(); k++) {
                    if (a == split_info[k].var.name()) {
                        RVar rvar = rxi[ split_info[k].filter_dim ];
                        Var  var  = split_info[k].outer_var;
                        args.push_back(rvar);
                        args.push_back(var);

                        update_var_category[i].erase(a);
                        update_var_category[i].insert(make_pair(rvar.name(),VarTag(INNER,i_cnt++)));
                        update_var_category[i].insert(make_pair(var.name(), VarTag(OUTER,o_cnt++)));

                        found = true;
                    }
                }
                if (!found) {
                    args.push_back(Var(a));
                }
            }
        }
        assert(dimension >= 0);

        // update values: create the intra tile scans with special
        // borders for all tile on image boundary is clamped_border as specified
        // border for all internal tiles is zero
        vector<Expr> values(F_intra.outputs());
        for (int j=0; j<values.size(); j++) {
            values[j] = Cast::make(type,feedfwd) * Call::make(F_intra, args, j);

            for (int k=0; k<feedback.size(); k++) {
                vector<Expr> call_args = args;
                Expr first_tile = (causal ? (xo==0) : (xo==num_tiles-1));
                if (causal) {
                    call_args[dimension] = max(call_args[dimension]-(k+1),0);
                } else {
                    call_args[dimension] = min(call_args[dimension]+(k+1),tile_width-1);
                }

                // inner tiles must always be clamped to zero beyond tile borders
                // tiles on the image border unless clamping is specified in
                // which case only inner tiles are clamped to zero beyond
                if (clamped_border) {
                    values[j] += Cast::make(type,feedback[k]) *
                        select(rxi[filter_dim]>k || first_tile,
                                Call::make(F_intra,call_args,j), make_zero(type));
                } else {
                    values[j] += Cast::make(type,feedback[k]) *
                        select(rxi[filter_dim]>k,
                            Call::make(F_intra,call_args,j), make_zero(type));
                }
            }
        }
        F_intra.define_update(args, values);
    }

    RecFilterFunc rF_intra;
    rF_intra.func               = F_intra;
    rF_intra.func_category      = INTRA_N;
    rF_intra.pure_var_category  = pure_var_category;
    rF_intra.update_var_category= update_var_category;

    return rF_intra;
}

// -----------------------------------------------------------------------------

static RecFilterFunc create_copy(RecFilterFunc rF, string func_name) {
    Function F = rF.func;
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

    // copy the scheduling tags
    RecFilterFunc rB;
    rB.func                = B;
    rB.func_category       = rF.func_category;
    rB.pure_var_category   = rF.pure_var_category;
    rB.update_var_category = rF.update_var_category;

    return rB;
}

// -----------------------------------------------------------------------------

static vector<RecFilterFunc> create_intra_tail_term(
        RecFilterFunc rF_intra,
        SplitInfo split_info,
        string func_name)
{
    int  tile = split_info.tile_width;
    Var  xi   = split_info.inner_var;

    Function F_intra = rF_intra.func;

    vector<RecFilterFunc> tail_functions_list;

    for (int k=0; k<split_info.num_scans; k++) {
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

        // copy all scheduling tags from intra tile scan except the update vars
        // and add the tag for the tail vars
        RecFilterFunc rf;
        rf.func = function;
        rf.func_category     = REINDEX;
        rf.callee_func       = rF_intra.func.name();
        rf.pure_var_category = rF_intra.pure_var_category;

        // extract the vartag count of xi, change xi's tag to tail and decrement the
        // count of all other INNER vars whose count is more than the count of xi
        int count_xi = rf.pure_var_category[xi.name()].count();
        rf.pure_var_category[xi.name()] = TAIL;
        map<string,VarTag>::iterator vit;
        for (vit=rf.pure_var_category.begin(); vit!=rf.pure_var_category.end(); vit++) {
            if (vit->first!=xi.name() && vit->second.check(INNER) && !vit->second.check(SCAN)) {
                int count = vit->second.count();
                if (count > count_xi) {
                    rf.pure_var_category[vit->first] = VarTag(INNER,count-1);
                }
            }
        }

        tail_functions_list.push_back(rf);
    }

    assert(tail_functions_list.size() == split_info.num_scans);

    return tail_functions_list;
}

// -----------------------------------------------------------------------------

static vector<RecFilterFunc> create_complete_tail_term(
        vector<RecFilterFunc> rF_tail,
        SplitInfo split_info,
        string func_name)
{
    vector<Function> F_tail;
    for (int i=0; i<rF_tail.size(); i++) {
        F_tail.push_back(rF_tail[i].func);
    }

    assert(split_info.num_scans == F_tail.size());

    Var  x    = split_info.var;
    Var  xi   = split_info.inner_var;
    Var  xo   = split_info.outer_var;
    RDom rxo  = split_info.outer_rdom;
    int  dim  = split_info.filter_dim;
    int  order= split_info.filter_order;
    int  tile = split_info.tile_width;
    int  num_tiles = split_info.num_tiles;

    vector<RecFilterFunc> rF_ctail;

    for (int k=0; k<split_info.num_scans; k++) {
        Function function(func_name + DASH + int_to_string(split_info.scan_id[k])
                + DASH + SUB);

        Image<double> weight = tail_weights(split_info, k);

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
                Expr prev_tile_expr = make_zero(split_info.type);
                for (int j=0; j<order; j++) {
                    prev_tile_expr += Cast::make(split_info.type, weight(tile-rxo.x-1, j)) *
                        Call::make(function, call_args_prev_tile[j], i);
                }

                Expr val = Call::make(function, call_args_curr_tile, i) +
                    select(rxo.y>0, prev_tile_expr, make_zero(split_info.type));
                values.push_back(simplify(val));
            }

            function.define_update(args, values);
        }

        // create the function scheduling tags
        // pure var tags are same as incomplete tail function tags
        // update var tags are same as pure var tags expect for xi and xo
        RecFilterFunc rf;
        rf.func = function;
        rf.func_category = INTER;
        rf.pure_var_category = rF_tail[k].pure_var_category;
        rf.update_var_category.push_back(rF_tail[k].pure_var_category);
        rf.update_var_category[0].insert(make_pair(rxo.x.name(), OUTER|SCAN));
        rf.update_var_category[0].insert(make_pair(rxo.y.name(), OUTER|SCAN));

        // extract the vartag count of xo, decrement the count of all other
        // OUTER vars whose count is more than the count of xo
        int count_xo = rf.update_var_category[0][xo.name()].count();
        rf.update_var_category[0].erase(xo.name());
        rf.update_var_category[0].erase(xi.name());
        map<string,VarTag>::iterator vit;
        for (vit=rf.update_var_category[0].begin(); vit!=rf.update_var_category[0].end(); vit++) {
            if (vit->second.check(OUTER) && !vit->second.check(SCAN)) {
                int count = vit->second.count();
                if (count>count_xo) {
                    rf.update_var_category[0][vit->first] = VarTag(OUTER,count-1);
                }
            }
        }

        rF_ctail.push_back(rf);
    }

    assert(rF_ctail.size() == split_info.num_scans);

    return rF_ctail;
}

// -----------------------------------------------------------------------------

static vector<RecFilterFunc> wrap_complete_tail_term(
        vector<RecFilterFunc> rF_ctail,
        SplitInfo split_info,
        string func_name)
{
    vector<RecFilterFunc> rF_ctailw;

    for (int k=0; k<rF_ctail.size(); k++) {
        Function F_ctail = rF_ctail[k].func;
        Function function(func_name + DASH + int_to_string(split_info.scan_id[k]));

        // simply create a pure function that calls the completed tail
        vector<string> args;
        vector<Expr>   values;
        vector<Expr>   call_args;

        for (int i=0; i<F_ctail.args().size(); i++) {
            args.push_back(F_ctail.args()[i]);
            call_args.push_back(Var(F_ctail.args()[i]));
        }
        for (int i=0; i<F_ctail.outputs(); i++) {
            values.push_back(Call::make(F_ctail, call_args, i));
        }

        function.define(args, values);

        // copy all scheduling tags from the complete tail term except update vars
        RecFilterFunc rf;
        rf.func = function;
        rf.func_category     = REINDEX;
        rf.callee_func       = rF_ctail[k].func.name();
        rf.pure_var_category = rF_ctail[k].pure_var_category;

        rF_ctailw.push_back(rf);
    }

    return rF_ctailw;
}

// -----------------------------------------------------------------------------

static vector<RecFilterFunc> create_tail_residual_term(
        vector<RecFilterFunc> rF_ctail,
        SplitInfo split_info,
        string func_name)
{
    int  order     = split_info.filter_order;
    Var  xi        = split_info.inner_var;
    Var  xo        = split_info.outer_var;
    int  num_tiles = split_info.num_tiles;
    int  tile      = split_info.tile_width;

    vector<Function> F_ctail;
    for (int i=0; i<rF_ctail.size(); i++) {
        F_ctail.push_back(rF_ctail[i].func);
    }

    int num_args    = F_ctail[0].args().size();
    int num_outputs = F_ctail[0].outputs();

    assert(split_info.num_scans == F_ctail.size());

    vector<RecFilterFunc> rF_tdeps;

    for (int u=0; u<split_info.num_scans; u++) {
        Function function(func_name + DASH + int_to_string(split_info.scan_id[u]));

        // args are same as completed tail terms
        vector<string> args = F_ctail[0].args();
        vector<Expr> values(num_outputs, make_zero(split_info.type));

        // accumulate the completed tails of all the preceedings scans
        // the list F_ctail is in reverse order just as split_info struct
        for (int j=u+1; j<F_ctail.size(); j++) {

            // weight matrix for accumulating completed tail elements from scan u to scan j
            Image<double> weight = tail_weights(split_info, j, u);

            // weight matrix for accumulating completed tail elements from scan u to scan j
            // for a tile that is clamped on all borders
            Image<double> c_weight = weight;
            if (split_info.clamped_border) {
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
                    Expr wt  = Cast::make(split_info.type, (split_info.scan_causal[j] ? weight(xi,k) : weight(tile-1-xi,k)));
                    Expr cwt = Cast::make(split_info.type, (split_info.scan_causal[j] ? c_weight(xi,k): c_weight(tile-1-xi,k)));

                    // change the weight for the first tile only, only this tile is
                    // affected by clamping the image at all borders
                    if (split_info.scan_causal[j]!=split_info.scan_causal[u]) {
                        Expr first_tile_u = (split_info.scan_causal[u] ? (xo==0) : (xo==num_tiles-1));
                        wt = select(first_tile_u, cwt, wt);
                    }

                    values[i] += simplify(select(first_tile, make_zero(split_info.type), wt*val));
                }
            }
        }
        function.define(args, values);

        // mark for inline schedule
        RecFilterFunc rf;
        rf.func = function;
        rf.func_category = INLINE;

        rF_tdeps.push_back(rf);
    }

    assert(rF_tdeps.size() == F_ctail.size());

    return rF_tdeps;
}

// -----------------------------------------------------------------------------

static vector<RecFilterFunc> create_final_residual_term(
        vector<RecFilterFunc> rF_ctail,
        SplitInfo split_info,
        string func_name,
        string final_result_func)
{
    int order     = split_info.filter_order;
    Var xi        = split_info.inner_var;
    Var xo        = split_info.outer_var;
    int num_tiles = split_info.num_tiles;
    int tile      = split_info.tile_width;

    vector<Function> F_ctail;
    for (int i=0; i<rF_ctail.size(); i++) {
        F_ctail.push_back(rF_ctail[i].func);
    }

    assert(split_info.num_scans == F_ctail.size());

    // create functions that just reindex the completed tails
    // this seemingly redundant reindexing allows the application to
    // read the completed tail into shared memory
    vector<Function>       F_deps;
    vector<RecFilterFunc> rF_deps;

    for (int j=0; j<split_info.num_scans; j++) {
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

        // add scheduling tags from completed tail function; mark this for reading
        // since this is useful for reading into shared mem
        RecFilterFunc rf;
        rf.func = function;
        rf.func_category = REINDEX;
        rf.pure_var_category = rF_ctail[j].pure_var_category;
        rf.caller_func = final_result_func;

        // add the genertaed recfilter function to the global list
        recfilter_func_list.insert(make_pair(rf.func.name(), rf));

        rF_deps.push_back(rf);
        F_deps.push_back(rf.func);
    }

    // accumulate all completed tails
    vector<RecFilterFunc> rF_deps_sub;

    // accumulate contribution from each completed tail
    for (int j=0; j<split_info.num_scans; j++) {
        Function function(func_name + DASH + int_to_string(split_info.scan_id[j]) + DASH + SUB);

        int num_args    = F_deps[j].args().size();
        int num_outputs = F_deps[j].outputs();

        // args are same as completed tail terms
        vector<string> args = F_deps[j].args();
        vector<Expr> values(num_outputs, make_zero(split_info.type));

        // weight matrix for accumulating completed tail elements
        // of scan after applying only current scan
        // Image<double> weight = tail_weights(split_info, j);
        Image<double> weight(order, order);
        for (int u=0; u<order; u++) {
            for (int v=0; v<order; v++) {
                weight(u,v) = ((u+v<order) ? split_info.feedback_coeff(split_info.scan_id[j], u+v) : 0.0);
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
                            Cast::make(split_info.type, weight(xi,k)) *
                            Call::make(F_deps[j], call_args, i), make_zero(split_info.type));
                } else {
                    val = select(xo<num_tiles-1,
                            Cast::make(split_info.type, weight(simplify(tile-1-xi),k)) *
                            Call::make(F_deps[j], call_args, i),  make_zero(split_info.type));
                }
                values[i] = simplify(values[i] + val);
            }
        }
        function.define(args, values);

        // mark for inline schedule
        RecFilterFunc rf;
        rf.func = function;
        rf.func_category = INLINE;

        // add the genertaed recfilter function to the global list
        recfilter_func_list.insert(make_pair(rf.func.name(), rf));
        rF_deps_sub.push_back(rf);
    }

    return rF_deps_sub;
}

// -----------------------------------------------------------------------------

static void add_residual_to_tails(
        vector<RecFilterFunc> rF_tail,
        vector<RecFilterFunc> rF_deps,
        SplitInfo split_info)
{
    Var  xi   = split_info.inner_var;
    int  tile = split_info.tile_width;

    vector<Function> F_tail;
    vector<Function> F_deps;
    for (int i=0; i<rF_tail.size(); i++) {
        F_tail.push_back(rF_tail[i].func);
    }
    for (int i=0; i<rF_deps.size(); i++) {
        F_deps.push_back(rF_deps[i].func);
    }

    // add the dependency term of each scan to the preceeding scan
    // the first scan does not have any residuals
    for (int j=0; j<split_info.num_scans-1; j++) {
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

static vector<RecFilterFunc> add_prev_dimension_residual_to_tails(
        RecFilterFunc         rF_intra,
        vector<RecFilterFunc> rF_tail,
        vector<RecFilterFunc> rF_tail_prev,
        SplitInfo split_info,
        SplitInfo split_info_prev)
{
    vector<RecFilterFunc> generated_func;

    Var x    = split_info.var;
    Var xi   = split_info.inner_var;
    Var xo   = split_info.outer_var;
    int tile = split_info.tile_width;
    int num_tiles = split_info.num_tiles;

    Var  yi   = split_info_prev.inner_var;
    Var  yo   = split_info_prev.outer_var;
    RDom ryi  = split_info_prev.inner_rdom;
    RDom ryt  = split_info_prev.tail_rdom;
    int  num_tiles_prev = split_info_prev.num_tiles;

    Function F_intra = rF_intra.func;
    vector<Function> F_tail;
    vector<Function> F_tail_prev;
    for (int i=0; i<rF_tail.size(); i++) {
        F_tail.push_back(rF_tail[i].func);
    }
    for (int i=0; i<rF_tail_prev.size(); i++) {
        F_tail_prev.push_back(rF_tail_prev[i].func);
    }

    // add the residual term of previous dimension to the completed
    // tail of current dimension
    for (int j=0; j<F_tail.size(); j++) {
        vector<string> pure_args = F_tail[j].args();
        vector<Expr> pure_values = F_tail[j].values();

        // each scan of prev dimension adds one residual
        for (int k=0; k<F_tail_prev.size(); k++) {

            // first scan the tail in the current dimension according using intra term
            RecFilterFunc rF_tail_prev_scanned_sub;
            Function F_tail_prev_scanned_sub(F_tail_prev[k].name() + DASH + x.name()
                    + DASH + int_to_string(split_info.scan_id[j]) + DASH + SUB);

            // scheduling tags: copy func type as F_intra, pure def tags from the
            // tail function or F_intra (both same) copy update tags from F_intra
            rF_tail_prev_scanned_sub.func = F_tail_prev_scanned_sub;
            rF_tail_prev_scanned_sub.func_category = INTRA_1;
            rF_tail_prev_scanned_sub.pure_var_category = rF_tail_prev[k].pure_var_category;

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
                map<string, VarTag> uvar_category = rF_intra.update_var_category[i];
                vector<Expr> args = F_intra.updates()[i].args;
                vector<Expr> values = F_intra.updates()[i].values;
                for (int v=0; v<ryi.dimensions(); v++) {
                    for (int u=0; u<args.size(); u++) {
                        args[u] = substitute(ryi[v].name(), ryt[v], args[u]);
                    }
                    VarTag vc = uvar_category[ryi[v].name()];
                    uvar_category.erase(ryi[v].name());
                    uvar_category.insert(make_pair(ryt[v].name(), vc));
                }
                for (int v=0; v<ryi.dimensions(); v++) {
                    for (int u=0; u<values.size(); u++) {
                        values[u] = substitute_func_call(F_intra.name(), F_tail_prev_scanned_sub, values[u]);
                        values[u] = substitute(ryi[v].name(), ryt[v], values[u]);
                    }
                }
                F_tail_prev_scanned_sub.define_update(args, values);

                rF_tail_prev_scanned_sub.update_var_category.push_back(uvar_category);
            }

            // create a pure function as a wrapper for the above function
            // allows compute_at schedules
            RecFilterFunc rF_tail_prev_scanned;
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

                // copy the scheduling tags of the scan
                rF_tail_prev_scanned.func = F_tail_prev_scanned;
                rF_tail_prev_scanned.func_category     = REINDEX;
                rF_tail_prev_scanned.pure_var_category = rF_tail_prev_scanned_sub.pure_var_category;
                rF_tail_prev_scanned.callee_func       = rF_tail_prev_scanned_sub.func.name();
            }

            // weight matrix for accumulating completed tail elements
            // of scan after applying all subsequent scans
            Image<double> weight  = tail_weights(split_info_prev, k, 0);

            // weight matrix for accumulating completed tail elements
            // of scan after applying all subsequent scans
            // for a tile that is clamped on all borders
            Image<double> c_weight = weight;
            if (split_info_prev.clamped_border) {
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
                    Expr wt  = Cast::make(split_info.type, (split_info_prev.scan_causal[k] ? weight(yi,o)  : weight(tile-1-yi,o)));
                    Expr cwt = Cast::make(split_info.type, (split_info_prev.scan_causal[k] ? c_weight(yi,o): c_weight(tile-1-yi,o)));

                    // change the weight for the first tile only, only this tile is affected
                    // by clamping the image at all borders
                    wt = select(last_tile, cwt, wt);

                    pure_values[i] += simplify(select(first_tile, make_zero(split_info.type), wt*val));
                }
            }

            // add the genertaed recfilter function to the global list
            generated_func.push_back(rF_tail_prev_scanned);
            generated_func.push_back(rF_tail_prev_scanned_sub);
        }

        // redefine the pure def to include residuals from prev dimensions
        // update defs of the tail remain unaffected, copy them as they were
        vector<UpdateDefinition> updates = F_tail[j].updates();
        F_tail[j].clear_all_definitions();
        F_tail[j].define(pure_args, pure_values);
        for (int i=0; i<updates.size(); i++) {
            F_tail[j].define_update(updates[i].args, updates[i].values);
        }
    }
    return generated_func;
}

// -----------------------------------------------------------------------------

static void add_all_residuals_to_final_result(
        RecFilterFunc& rF,
        vector< vector<RecFilterFunc> > rF_deps,
        vector<SplitInfo> split_info)
{
    Function F = rF.func;

    vector<vector<Function> > F_deps(rF_deps.size());
    for (int i=0; i<rF_deps.size(); i++) {
        for (int j=0; j<rF_deps[i].size(); j++) {
            F_deps[i].push_back(rF_deps[i][j].func);
        }
    }

    assert(split_info.size() == F_deps.size());

    vector<string> pure_args   = F.args();
    vector<Expr>   pure_values = F.values();
    vector<UpdateDefinition> updates = F.updates();
    vector< map<string,VarTag> > update_var_category = rF.update_var_category;

    // new updates to be added for all the split updates
    map<int, std::pair< vector<Expr>, vector<Expr> > > new_updates;
    map<int, map<string,VarTag> > new_update_var_category;

    // create the new update definition for each scan that add the scans
    // residual to its first k elements (k = filter order)
    for (int i=0; i<F_deps.size(); i++) {
        int tile_width = split_info[i].tile_width;
        int num_tiles  = split_info[i].num_tiles;
        for (int j=0; j<F_deps[i].size(); j++) {
            int  curr_scan   = split_info[i].scan_id[j];
            RDom rxi         = split_info[i].inner_rdom;
            RDom rxt         = split_info[i].tail_rdom;
            RDom rxf         = split_info[i].truncated_inner_rdom;
            vector<Expr> args= updates[curr_scan].args;

            // copy the scheduling tags from the original update def
            new_update_var_category[curr_scan] = update_var_category[curr_scan];

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
                VarTag v = new_update_var_category[curr_scan][rxi[k].name()];
                new_update_var_category[curr_scan].erase(rxi[k].name());
                new_update_var_category[curr_scan].insert(make_pair(rxt[k].name(), v));

                // the new update runs the scan for the first t elements
                // change the reduction domain of the original update to
                // run from t onwards, t = filter order
                for (int u=0; u<updates[curr_scan].args.size(); u++) {
                    updates[curr_scan].args[u] = substitute(rxi[k].name(), rxf[k], updates[curr_scan].args[u]);
                }
                for (int u=0; u<updates[curr_scan].values.size(); u++) {
                    updates[curr_scan].values[u] = substitute(rxi[k].name(), rxf[k], updates[curr_scan].values[u]);
                }
                VarTag vc = update_var_category[curr_scan][rxi[k].name()];
                update_var_category[curr_scan].erase(rxi[k].name());
                update_var_category[curr_scan].insert(make_pair(rxf[k].name(), vc));
            }

            new_updates[curr_scan] = make_pair(args, values);
        }
    }

    // add extra update steps
    rF.update_var_category.clear();
    F.clear_all_definitions();
    F.define(pure_args, pure_values);
    for (int i=0; i<updates.size(); i++) {
        if (new_updates.find(i) != new_updates.end()) {
            vector<Expr> args   = new_updates[i].first;
            vector<Expr> values = new_updates[i].second;
            F.define_update(args, values);
            rF.update_var_category.push_back(new_update_var_category[i]);
        }
        F.define_update(updates[i].args, updates[i].values);
        rF.update_var_category.push_back(update_var_category[i]);
    }
}

// -----------------------------------------------------------------------------

static vector< vector<RecFilterFunc> > split_scans(
        RecFilterFunc F_intra,
        string final_result_func,
        vector<SplitInfo> &split_info)
{
    vector< vector<RecFilterFunc> > F_ctail_list;
    vector< vector<RecFilterFunc> > F_deps_list;

    for (int i=0; i<split_info.size(); i++) {
        string x = split_info[i].var.name();

        string s0 = F_intra.func.name() + DASH + INTRA_TILE_TAIL_TERM  + DASH + x;
        string s1 = F_intra.func.name() + DASH + INTER_TILE_TAIL_SUM   + DASH + x;
        string s2 = F_intra.func.name() + DASH + COMPLETE_TAIL_RESIDUAL+ DASH + x;
        string s3 = F_intra.func.name() + DASH + FINAL_RESULT_RESIDUAL + DASH + x;

        // all these recfilter func have already been added to the global list,
        // no need to add again
        vector<RecFilterFunc> F_tail   = create_intra_tail_term    (F_intra,  split_info[i], s0);
        vector<RecFilterFunc> F_ctail  = create_complete_tail_term (F_tail,   split_info[i], s1);
        vector<RecFilterFunc> F_ctailw = wrap_complete_tail_term   (F_ctail,  split_info[i], s1);
        vector<RecFilterFunc> F_tdeps  = create_tail_residual_term (F_ctail,  split_info[i], s2);
        vector<RecFilterFunc> F_deps   = create_final_residual_term(F_ctailw, split_info[i], s3, final_result_func);

        // add the dependency from each scan to the tail of the next scan
        // this ensures that the tail of each scan includes the complete
        // result from all previous scans
        add_residual_to_tails(F_ctail, F_tdeps, split_info[i]);

        // add the residuals from split up scans in all previous
        // dimensions to this scan
        for (int j=0; j<i; j++) {
            vector<RecFilterFunc> gen_func = add_prev_dimension_residual_to_tails(
                    F_intra, F_ctail, F_ctail_list[j], split_info[i], split_info[j]);

            for (int k=0; k<gen_func.size(); k++) {
                recfilter_func_list.insert(make_pair(gen_func[k].func.name(), gen_func[k]));
            }
        }

        // add all the generated functions to the global list
        for (int j=0; j<F_tail.size(); j++) {
            recfilter_func_list.insert(make_pair(F_tail[j].func.name(), F_tail[j]));
        }
        for (int j=0; j<F_ctail.size(); j++) {
            recfilter_func_list.insert(make_pair(F_ctail[j].func.name(), F_ctail[j]));
        }
        for (int j=0; j<F_ctailw.size(); j++) {
            recfilter_func_list.insert(make_pair(F_ctailw[j].func.name(), F_ctailw[j]));
        }
        for (int j=0; j<F_tdeps.size(); j++) {
            recfilter_func_list.insert(make_pair(F_tdeps[j].func.name(), F_tdeps[j]));
        }
        for (int j=0; j<F_deps.size(); j++) {
            recfilter_func_list.insert(make_pair(F_deps[j].func.name(), F_deps[j]));
        }

        F_ctail_list.push_back(F_ctailw);
        F_deps_list .push_back(F_deps);
    }

    return F_deps_list;
}

// -----------------------------------------------------------------------------

void RecFilter::split(map<string,int> dim_tile) {
    if (contents.ptr->tiled) {
        cerr << "Recursive filter cannot be split twice" << endl;
        assert(false);
    }

    // clear global variables
    // TODO: remove the global vars and make them objects of the RecFilter class in some way
    contents.ptr->finalized = false;
    contents.ptr->compiled  = false;
    recfilter_split_info.clear();
    recfilter_func_list.clear();

    // main function of the recursive filter that contains the final result
    RecFilterFunc& rF = internal_function(contents.ptr->name);
    Function        F = rF.func;

    // group scans in same dimension together and change the order of splits accordingly
    contents.ptr->filter_info = group_scans_by_dimension(F, contents.ptr->filter_info);

    // inner RDom - has dimensionality equal to dimensions of the image
    // each dimension runs from 0 to tile width of the respective dimension
    vector<ReductionVariable> inner_scan_rvars;

    // inject tiling info into the FilterInfo structs
    for (int i=0; i<contents.ptr->filter_info.size(); i++) {
        if (dim_tile.find(contents.ptr->filter_info[i].var.name()) != dim_tile.end()) {

            // check that there are scans in this dimension
            if (contents.ptr->filter_info[i].scan_id.empty()) {
                cerr << "No scans to tile in dimension "
                     << contents.ptr->filter_info[i].var.name() << endl;
                assert(false);
            }

            contents.ptr->filter_info[i].tile_width = dim_tile[contents.ptr->filter_info[i].var.name()];
        }

        Expr extent = 1;
        if (contents.ptr->filter_info[i].tile_width != contents.ptr->filter_info[i].image_width) {
            extent = contents.ptr->filter_info[i].tile_width;
        }

        ReductionVariable r;
        r.min    = 0;
        r.extent = extent;
        r.var    = "r" + contents.ptr->filter_info[i].var.name() + "i";
        inner_scan_rvars.push_back(r);
    }
    RDom inner_rdom = RDom(ReductionDomain(inner_scan_rvars));

    // populate tile size and number of tiles for each dimension
    // populate the inner, outer and tail update domains to all dimensions
    for (map<string,int>::iterator it=dim_tile.begin(); it!=dim_tile.end(); it++) {
        bool found = false;
        string x = it->first;
        int tile_width = it->second;
        for (int j=0; !found && j<contents.ptr->filter_info.size(); j++) {
            if (x != contents.ptr->filter_info[j].var.name()) {
                continue;
            }
            assert(contents.ptr->filter_info[j].tile_width == tile_width);

            SplitInfo s;

            // copy data from filter_info struct to split_info struct
            s.filter_order    = contents.ptr->filter_info[j].filter_order;
            s.filter_dim      = contents.ptr->filter_info[j].filter_dim;
            s.num_scans       = contents.ptr->filter_info[j].num_scans;
            s.var             = contents.ptr->filter_info[j].var;
            s.rdom            = contents.ptr->filter_info[j].rdom;
            s.scan_causal     = contents.ptr->filter_info[j].scan_causal;
            s.scan_id         = contents.ptr->filter_info[j].scan_id;
            s.image_width     = contents.ptr->filter_info[j].image_width;
            s.tile_width      = contents.ptr->filter_info[j].tile_width;
            s.num_tiles       = contents.ptr->filter_info[j].image_width / tile_width;

            s.feedfwd_coeff   = contents.ptr->feedfwd_coeff;
            s.feedback_coeff  = contents.ptr->feedback_coeff;
            s.clamped_border  = contents.ptr->clamped_border;
            s.type            = contents.ptr->type;

            // set inner var and outer var
            s.inner_var = Var(x+"i");
            s.outer_var = Var(x+"o");

            // set inner rdom, same for all dimensions
            s.inner_rdom = inner_rdom;

            // same as inner rdom except that the extent of scan dimension
            // is filter order rather than tile width
            vector<ReductionVariable> inner_tail_rvars = inner_scan_rvars;
            inner_tail_rvars[j].var    = "r"+x+"t";
            inner_tail_rvars[j].min    = 0;
            inner_tail_rvars[j].extent = s.filter_order;
            s.tail_rdom = RDom(ReductionDomain(inner_tail_rvars));

            // same as inner rdom except that the domain is from filter_order to tile_width-1
            // instead of 0 to tile_width-1
            vector<ReductionVariable> inner_truncated_rvars = inner_scan_rvars;
            inner_truncated_rvars[j].var    = "r"+x+"f";
            inner_truncated_rvars[j].min    = s.filter_order;
            inner_truncated_rvars[j].extent = simplify(max(inner_truncated_rvars[j].extent-s.filter_order,0));
            s.truncated_inner_rdom = RDom(ReductionDomain(inner_truncated_rvars));

            // outer_rdom.x: over all tail elements of current tile
            // outer_rdom.y: over all tiles
            s.outer_rdom = RDom(0, s.filter_order, 0, s.num_tiles, "r"+x+"o");

            recfilter_split_info.push_back(s);

            found = true;
        }
        if (!found) {
            cerr << "Variable " << x << " does not correspond to any "
                << "dimension of the recursive filter " << contents.ptr->name << endl;
            assert(false);
        }
    }

    // apply the actual splitting
    RecFilterFunc rF_final;
    {
        // compute the intra tile result
        RecFilterFunc rF_intra = create_intra_tile_term(rF, recfilter_split_info);

        // create a function will hold the final result, copy of the intra tile term
        rF_final = create_copy(rF_intra, F.name() + DASH + FINAL_TERM);

        // compute the residuals from splits in each dimension
        vector< vector<RecFilterFunc> > rF_deps = split_scans(rF_intra, rF_final.func.name(),
                recfilter_split_info);

        // transfer the tail of each scan to another buffer
        extract_tails_from_each_scan(rF_intra, recfilter_split_info);

        // add all the residuals to the final term
        add_all_residuals_to_final_result(rF_final, rF_deps, recfilter_split_info);

        // add the intra and final term to the list functions
        recfilter_func_list.insert(make_pair(rF_intra.func.name(), rF_intra));
        recfilter_func_list.insert(make_pair(rF_final.func.name(), rF_final));
    }

    // change the original function to index into the final term
    // for GPU codegen
    if (contents.ptr->target.has_gpu_feature()) {
        Function F_final = rF_final.func;
        vector<string> args = F_final.args();
        vector<Expr> values;
        vector<Expr> call_args;
        for (int i=0; i<args.size(); i++) {
            call_args.push_back(Var(args[i]));
        }
        for (int i=0; i<F_final.outputs(); i++) {
            values.push_back(Call::make(F_final, call_args, i));
        }
        F.clear_all_definitions();
        F.define(args, values);

        // remove the scheduling tags of the update defs
        rF.func_category = REINDEX;
        rF.callee_func = F_final.name();
        rF.update_var_category.clear();
    } else {
        rF = create_copy(rF_final, contents.ptr->name);
        recfilter_func_list.erase(rF_final.func.name());
    }
    recfilter_func_list.insert(make_pair(rF.func.name(), rF));

//    {
//        Function F_final = rF_final.func;
//        vector<string> args = F.args();
//        vector<Expr> values;
//        vector<Expr> call_args;
//        for (int i=0; i<F_final.args().size(); i++) {
//            string arg = F_final.args()[i];
//            call_args.push_back(Var(arg));
//
//            for (int j=0; j<recfilter_split_info.size(); j++) {
//                Var var        = recfilter_split_info[j].var;
//                Var inner_var  = recfilter_split_info[j].inner_var;
//                Var outer_var  = recfilter_split_info[j].outer_var;
//                int tile_width = recfilter_split_info[j].tile_width;
//                if (arg == inner_var.name()) {
//                    call_args[i] = substitute(arg, var%tile_width, call_args[i]);
//                } else if (arg == outer_var.name()) {
//                    call_args[i] = substitute(arg, var/tile_width, call_args[i]);
//                }
//            }
//        }
//        for (int i=0; i<F.outputs(); i++) {
//            Expr val = Call::make(F_final, call_args, i);
//            values.push_back(val);
//        }
//        F.clear_all_definitions();
//        F.define(args, values);
//
//        // remove the scheduling tags of the update defs
//        rF.func_category = REINDEX;
//        rF.callee_func = F_final.name();
//        rF.update_var_category.clear();
//
//        // split the tiled vars of the final term
//        for (int i=0; i<recfilter_split_info.size(); i++) {
//            Var var        = recfilter_split_info[i].var;
//            Var inner_var  = recfilter_split_info[i].inner_var;
//            Var outer_var  = recfilter_split_info[i].outer_var;
//            int tile_width = recfilter_split_info[i].tile_width;
//
//            Func(F).split(var, outer_var, inner_var, tile_width);
//
//            stringstream s;
//            s << "split(Var(\"" << var.name() << "\"), Var(\""
//                << outer_var.name() << "\"), Var(\"" << inner_var.name() << "\"), "
//                << tile_width << ")";
//
//            rF.pure_var_category.erase(var.name());
//            rF.pure_var_category.insert(make_pair(inner_var.name(), VarTag(INNER,i)));
//            rF.pure_var_category.insert(make_pair(outer_var.name(), VarTag(OUTER,i)));
//            rF.pure_schedule.push_back(s.str());
//        }
//
//        recfilter_func_list.insert(make_pair(rF.func.name(), rF));
//    }

    // add all the generated RecFilterFuncs
    contents.ptr->func.insert(recfilter_func_list.begin(), recfilter_func_list.end());

    contents.ptr->tiled = true;

    // perform generic and target dependent optimizations
    finalize();

    recfilter_func_list.clear();
    recfilter_split_info.clear();
}

void RecFilter::split(RecFilterDim x, int tx) {
    map<string,int> dim_tile;
    dim_tile[x.var().name()] = tx;
    split(dim_tile);
}

void RecFilter::split(RecFilterDim x, int tx, RecFilterDim y, int ty) {
    map<string,int> dim_tile;
    dim_tile[x.var().name()] = tx;
    dim_tile[y.var().name()] = ty;
    split(dim_tile);
}

void RecFilter::split(RecFilterDim x, int tx, RecFilterDim y, int ty, RecFilterDim z, int tz) {
    map<string,int> dim_tile;
    dim_tile[x.var().name()] = tx;
    dim_tile[y.var().name()] = ty;
    dim_tile[z.var().name()] = tz;
    split(dim_tile);
}

// -----------------------------------------------------------------------------

void RecFilter::finalize(void) {
    map<string,RecFilterFunc>::iterator fit;

    if (contents.ptr->tiled) {
        // inline all functions not required any mor
        for (fit=contents.ptr->func.begin(); fit!=contents.ptr->func.end(); fit++) {
            if (fit->second.func_category == INLINE) {
                inline_func(fit->second.func.name());
                fit = contents.ptr->func.begin();            // list changed, start all over again
            }
        }

        // merge functions that reindex same previous result
        // useful for all architectures
        for (fit=contents.ptr->func.begin(); fit!=contents.ptr->func.end(); fit++) {
            RecFilterFunc rF = fit->second;

            if (((rF.func_category==INTRA_1) || (rF.func_category==INTRA_N))
                    && (rF.func.has_update_definition())) {
                // get all functions that reindex the tail from intra tile computation
                vector<string> funcs_to_merge;
                map<string,RecFilterFunc>::iterator git;
                for (git=contents.ptr->func.begin(); git!=contents.ptr->func.end(); git++) {
                    if ((git->second.func_category==REINDEX) &&
                            (git->second.callee_func==rF.func.name())) {
                        funcs_to_merge.push_back(git->second.func.name());
                    }
                }

                // merge these functions
                // memory layout reshaping done inside merge routine
                if (funcs_to_merge.size()>1) {
                    string merged_name = funcs_to_merge[0] + "_merged";
                    merge_func(funcs_to_merge, merged_name);
                    fit = contents.ptr->func.begin();            // list changed, start all over again
                }
            }
        }

        // platform specific optimization
        if (contents.ptr->target.has_gpu_feature()) {
            for (fit=contents.ptr->func.begin(); fit!=contents.ptr->func.end(); fit++) {
                RecFilterFunc& rF = fit->second;

                // move initialization to update def in intra tile computation stages
                // add padding to intra tile terms to avoid bank conflicts

                if (rF.func_category==INTRA_1 && rF.func.has_update_definition()) {
                    move_init_to_update_def(rF, recfilter_split_info);
                }
                if (rF.func_category==INTRA_N && rF.func.has_update_definition()) {
                    move_init_to_update_def(rF, recfilter_split_info);
                    add_padding_to_avoid_bank_conflicts(rF, recfilter_split_info, true);
                }

            }
        }
    }

    // CPU optimizations to be performed with out without tiling
    if (!contents.ptr->target.has_gpu_feature()) {
        // inline all reindexing functions
        for (fit=contents.ptr->func.begin(); fit!=contents.ptr->func.end(); fit++) {
            if (fit->second.func_category == REINDEX) {
                inline_func(fit->second.func.name());
                fit = contents.ptr->func.begin();            // list changed, start all over again
            }
        }
        // make all other functions as compute root
        for (fit=contents.ptr->func.begin(); fit!=contents.ptr->func.end(); fit++) {
            Func(fit->second.func).compute_root();
            fit->second.pure_schedule.push_back("compute_root()");
        }
    }

    contents.ptr->finalized = true;
}
