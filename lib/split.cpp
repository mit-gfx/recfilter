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
using std::pair;
using std::make_pair;
using std::swap;

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

    vector<bool> scan_causal;           ///< causal or anticausal flag for each scan
    vector<int>  scan_id;               ///< scan or update definition id of each scan

    Halide::Image<float> feedfwd_coeff; ///< Feedforward coeffs (from RecFilterContents)
    Halide::Image<float> feedback_coeff;///< Feedback coeffs  (from RecFilterContents)
};

/** Tiling info for each dimension of the filter */
static vector<SplitInfo> recfilter_split_info;

/** All recursive filter funcs created during splitting transformations */
static map<string, RecFilterFunc> recfilter_func_list;

// -----------------------------------------------------------------------------

/** Convert the pure def into the first update def and leave the pure def undefined
 * \param[in,out] rF function to be modified
 * \param[in] split_info tiling metadata
 */
static void convert_pure_def_into_first_update_def(
        RecFilterFunc& rF,
        vector<SplitInfo> split_info)
{
    assert(!split_info.empty());

    Function F = rF.func;

    // nothing needed if the function is pure
    if (F.is_pure()) {
        return;
    }

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
            VarTag vc = update_var_category[xi.name()];
            update_var_category.erase(xi.name());
            update_var_category.insert(make_pair(rxi.name(), vc));
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


/** Weight coefficients (tail_size x tile_width) for applying scans corresponding
 * to split indices split_id1 to split_id2 in split_info object; it is meaningful
 * to apply subsequent scans on the tail of any scan as it undergoes other scans
 * only if they happen after the first scan.
 *
 * Preconditions:
 * - split_id1 and split_id2 must be decreasing because spli_info stores
 *   scans in reverse order
 *
 * \param[in] s tiling metadata for the current dimension
 * \param[in] split_id1 index of one of the scans in the current dimension
 * \param[in] split_id2 index of one of the scans in the current dimension
 * \param[in] clamp_border adjust coefficients for clamped image borders
 * \returns matrix of coefficients
 */
Image<float> tail_weights(SplitInfo s, int split_id1, int split_id2, bool clamp_border=false) {
    assert(split_id1 >= split_id2);

    int  tile_width  = s.tile_width;
    int  scan_id     = s.scan_id[split_id1];
    bool scan_causal = s.scan_causal[split_id1];

    Image<float> R = matrix_R(s.feedback_coeff, scan_id, tile_width);

    // accummulate weight coefficients because of all subsequent scans
    // traversal is backwards because SplitInfo contains scans in the
    // reverse order
    for (int j=split_id1-1; j>=split_id2; j--) {
        if (scan_causal != s.scan_causal[j]) {
            Image<float> B = matrix_B(s.feedfwd_coeff,
                    s.feedback_coeff, s.scan_id[j], tile_width, clamp_border);
            Image<float> I = matrix_antidiagonal(R.height());
            R = matrix_mult(I, R);
            R = matrix_mult(B, R);
            R = matrix_mult(I, R);
        }
        else {
            Image<float> B = matrix_B(s.feedfwd_coeff,
                    s.feedback_coeff, s.scan_id[j], tile_width, false);
            R = matrix_mult(B, R);
        }
    }

    return matrix_transpose(R);
}

/** Weight coefficients (tail_size x tile_width) for applying scan's corresponding
 * to split indices split_id1
 *
 * \param[in] s tiling metadata for the current dimension
 * \param[in] split_id index of one of the scans in the current dimension
 * \param[in] clamp_border adjust coefficients for clamped image borders
 * \returns matrix of coefficients
 */
Image<float> tail_weights(SplitInfo s, int split_id1, bool clamp_border=false) {
    return tail_weights(s, split_id1, split_id1, clamp_border);
}

// -----------------------------------------------------------------------------

/**
 * Reorder the update defs such that update defs in first dimension come first,
 * followed by next dimension and so on; this can be performed because dimensions
 * are separable and this allows clean tiling semantics
 *
 * \param[in] F function containing scans in multiple dimensions
 * \param[in] filter_info scan info aboout all dimensions
*/
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

/** Make sure that all vars with tags INNER, OUTER or FULL have VarTag count in
 * continuous increasing order - this continuity was broken during splitting where
 * vars were replaced by inner/outer/tail vars
 *
 * \param[in,out] var_tags list of variable tags to be modified
 */
static void reassign_vartag_counts(map<string,VarTag>& var_tags) {
    vector<VariableTag> ref_vartag = {INNER, OUTER, FULL};

    for (int u=0; u<ref_vartag.size(); u++) {
        VarTag ref(ref_vartag[u]);

        map<int,string> var_count;

        map<string,VarTag>::iterator vartag_it;
        map<int,string>::iterator    count_it;

        // extract the vars and put them in sorted order according to their count
        for (vartag_it=var_tags.begin(); vartag_it!=var_tags.end(); vartag_it++) {
            string var = vartag_it->first;
            VarTag tag = vartag_it->second;
            if (tag.check(ref) && !tag.check(SCAN)) {
                int count = tag.count();
                var_count.insert(make_pair(count,var));
            }
        }

        // access the vars in sorted order
        int count = 0;
        for (count_it=var_count.begin(); count_it!=var_count.end(); count_it++) {
            string var = count_it->second;
            var_tags[ var ] = VarTag(ref,count);
            count++;
        }
    }
}

// -----------------------------------------------------------------------------

/**
 * Extract the tails from the intra-tile scans; each intra-tile scan runs within
 * the tile boundaries after which the tail is copied outside the tile boundaries
 * to ensure subsequent scans do not overwrite the tails from previous scans; the
 * tails from all the scans are packed into the same dimension - the innermost
 * tiled dimension
 * \param[in,out] rF_intra intra tile term that computes intra tile scans
 * \param[in,out] rF_tail list of empty tail functions for each scan of each dimension
 * \param[in] split_info tiling metadata
 */
static RecFilterFunc extract_tails_from_each_scan(
        RecFilterFunc& rF_intra,
        vector< vector<RecFilterFunc> > rF_tail,
        vector<SplitInfo> split_info)
{
    RecFilterFunc rF_intra_tail;

    Function F_intra = rF_intra.func;

    // the dimension in which all tails have to be packed
    int    tail_dimension_id = -1;
    int    tail_dimension_tile_width = 0;
    string tail_dimension_var;

    // find the innermost tiled dimension
    {
        for (int i=0; i<F_intra.args().size(); i++) {
            string arg = F_intra.args()[i];
            for (int j=0; tail_dimension_id<0 && j<split_info.size(); j++) {
                if (arg==split_info[j].inner_var.name()) {
                    tail_dimension_id  = j;
                    tail_dimension_var = arg;
                    tail_dimension_tile_width = split_info[j].tile_width;
                }
            }
        }
    }
    assert(tail_dimension_id>=0);

    // add extra update steps to extract the tail after each scan copy in
    // memory outside tile boundaries in the dimension found above
    {
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
        map<int, pair< vector<Expr>, vector<Expr> > > new_updates;
        map<int, map<string,VarTag> > new_update_var_category;

        // create the new update definitions to extract the tail
        // of each scan that is split
        int next_tail_offset = tail_dimension_tile_width;
        for (int i=0; i<split_info.size(); i++) {
            int  dim   = split_info[i].filter_dim;
            int  order = split_info[i].filter_order;
            Var  xi    = split_info[i].inner_var;
            RDom rxi   = split_info[i].inner_rdom;
            RDom rxt   = split_info[i].tail_rdom;
            int  tile  = split_info[i].tile_width;
            int  nscans= split_info[i].num_scans;

            // new update to extract the tail of each split scan
            for (int j=0; j<split_info[i].num_scans; j++) {
                int  scan_id     = split_info[i].scan_id[j];
                bool scan_causal = split_info[i].scan_causal[j];

                vector<Expr> args      = updates[scan_id].args;
                vector<Expr> call_args = updates[scan_id].args;
                vector<Expr> values;

                // copy the scheduling tags from the original update def
                new_update_var_category[scan_id] = update_var_category[scan_id];

                // find the dimension the undergoing the scan
                int scan_dimension = -1;
                for (int u=0; scan_dimension<0 && pure_args.size(); u++) {
                    if (xi.name() == pure_args[u]) {
                        scan_dimension = u;
                    }
                }
                assert(scan_dimension>=0);

                // replace rxi by rxt (i.e. rxi.whatever by rxt.whatever)
                // except for the dimension undergoing scan:
                // - arg should be offset+rxt[dim] because it needs to be stored outside tile boundary
                // - calling arg should be tile-1-rxt[dim] because the tails has to be extracted
                // all other dimensions need simple rxi->rxt replacement
                for (int k=0; k<rxi.dimensions(); k++) {
                    for (int u=0; u<args.size(); u++) {
                        if (u==scan_dimension) {
                            args     [u] = simplify(next_tail_offset + rxt[dim]);
                            call_args[u] = simplify(scan_causal ? (tile-1-rxt[dim]) : rxt[dim]);
                        } else {
                            args[u]      = substitute(rxi[k].name(), rxt[k], args[u]);
                            call_args[u] = substitute(rxi[k].name(), rxt[k], call_args[u]);
                        }
                    }

                    // same replacement in scheduling tags
                    if (update_var_category[scan_id].find(rxi[k].name()) != update_var_category[scan_id].end()) {
                        VarTag vc = update_var_category[scan_id][rxi[k].name()];
                        new_update_var_category[scan_id].erase(rxi[k].name());
                        new_update_var_category[scan_id].insert(make_pair(rxt[k].name(), vc));
                    }
                }

                // swap the scan dimension and tail dimension to ensure that tails from
                // scans in any dimension are packed in the same dimension
                swap(args[tail_dimension_id], args[scan_dimension]);

                // increment the offset where the tail from the next scan should be stored
                next_tail_offset += order;

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

        // add a final update def to resolve bank conflicts
        // add one pixel to the innermost split dimension
        {
            int innermost_tiled_dim_id = -1;
            int innermost_tiled_dim_buffer_width = 0;

            // find the innermost tiled dimension
            for (int i=0; i<F_intra.args().size(); i++) {
                string arg = F_intra.args()[i];
                for (int j=0; innermost_tiled_dim_id<0 && j<split_info.size(); j++) {
                    if (arg==split_info[j].inner_var.name()) {
                        innermost_tiled_dim_id = j;
                        innermost_tiled_dim_buffer_width = split_info[j].tile_width;
                    }
                }
            }
            assert(innermost_tiled_dim_id>=0);

            // if this dimension is the same in which the tails have been packed
            // then the width of the buffer is the tile width plus all the tails
            if (innermost_tiled_dim_id == tail_dimension_id) {
                for (int j=0; j<split_info.size(); j++) {
                    int tail_width = split_info[j].filter_order * split_info[j].num_scans;
                    innermost_tiled_dim_buffer_width += tail_width;
                }
            }

            vector<Expr> args;
            vector<Expr> values;
            for (int i=0; i<F_intra.args().size(); i++) {
                if (i==innermost_tiled_dim_id) {
                    args.push_back(innermost_tiled_dim_buffer_width);
                } else {
                    args.push_back(Var(F_intra.args()[i]));
                }
            }
            for (int i=0; i<F_intra.outputs(); i++) {
                values.push_back(undef(F_intra.output_types()[i]));
            }
            F_intra.define_update(args, values);

            // add scheduling tags for the last update def
            map<string,VarTag> uvar_category = rF_intra.pure_var_category;
            uvar_category.erase(F_intra.args()[innermost_tiled_dim_id]);
            rF_intra.update_var_category.push_back(uvar_category);
        }
    }

    // copy the extended buffer into a separate function, this is the function
    // that is actually written to memory that contains all the tails of all scans
    {
        Function F_intra_tail(F_intra.name() + DASH + INTRA_TILE_TAIL_TERM);

        vector<string> pure_args = F_intra.args();
        vector<Expr> call_args;
        vector<Expr> values;
        for (int i=0; i<pure_args.size(); i++) {
            if (pure_args[i] == tail_dimension_var) {
                call_args.push_back(tail_dimension_tile_width + Var(pure_args[i]));
            } else {
                call_args.push_back(Var(pure_args[i]));
            }
        }
        for (int i=0; i<F_intra.outputs(); i++) {
            values.push_back(Call::make(F_intra, call_args, i));
        }

        F_intra_tail.define(F_intra.args(), values);

        rF_intra_tail.func = F_intra_tail;
        rF_intra_tail.func_category = REINDEX;
        rF_intra_tail.callee_func   = F_intra.name();
        rF_intra_tail.pure_var_category = rF_intra.pure_var_category;
        rF_intra_tail.pure_var_category[tail_dimension_var] = TAIL;
    }

    // redefine the tail functions for each scans to index into the
    // above created function, the tail functions were originally left undef
    {
        Function F_intra_tail = rF_intra_tail.func;
        vector<string> args = rF_intra_tail.func.args();
        vector<Expr> values(F_intra_tail.outputs());
        vector<Expr> call_args;

        for (int i=0; i<args.size(); i++) {
            call_args.push_back(Var(args[i]));
        }
        call_args[tail_dimension_id] += 0;

        for (int l=0; l<split_info.size(); l++) {
            int order = split_info[l].filter_order;
            Var xi = split_info[l].inner_var;
            Var yi = args[tail_dimension_id];
            Var t  = unique_name(xi.name() + yi.name());
            for (int k=0; k<split_info[l].num_scans; k++) {
                for (int i=0; i<values.size(); i++) {
                    values[i] = Call::make(F_intra_tail, call_args, i);
                    values[i] = substitute(xi.name(), t,  values[i]);
                    values[i] = substitute(yi.name(), xi, values[i]);
                    values[i] = substitute(t.name(),  yi, values[i]);
                }
                call_args[tail_dimension_id] += order;
                Function ftail = rF_tail[l][k].func;
                ftail.clear_all_definitions();
                ftail.define(args, values);
            }
        }
    }

    return rF_intra_tail;
}

// -----------------------------------------------------------------------------

static RecFilterFunc create_intra_tile_term(
        RecFilterFunc rF,
        vector<SplitInfo> split_info)
{
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
        int tile_width  = split_info[i].tile_width;

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

        float feedfwd = s.feedfwd_coeff(i);
        vector<float> feedback(filter_order,0.0);
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

static vector< vector<RecFilterFunc> > create_intra_tail_term(
        RecFilterFunc rF_intra,
        vector<SplitInfo> split_info)
{
    vector< vector<RecFilterFunc> > tail_functions_list(split_info.size());

    Function F_intra = rF_intra.func;

    for (int l=0; l<split_info.size(); l++) {
        int  tile = split_info[l].tile_width;
        Var  xi   = split_info[l].inner_var;
        Var  x    = split_info[l].var;

        for (int k=0; k<split_info[l].num_scans; k++) {
            string s = F_intra.name() + DASH + INTRA_TILE_TAIL_TERM + DASH
                + x.name() + int_to_string(split_info[l].scan_id[k]);

            Function function(s);

            int scan_id = split_info[l].scan_id[k];
            int order   = split_info[l].filter_order;

            vector<string> args = F_intra.args();
            vector<Expr>   values(F_intra.outputs(), undef(split_info[0].type));

            function.define(args, values);

            RecFilterFunc rf;
            rf.func = function;
            rf.func_category = INLINE;
            rf.pure_var_category = rF_intra.pure_var_category;

            tail_functions_list[l].push_back(rf);
        }
    }

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
                        args.push_back(num_tiles-1-rxo.y);
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
                values.push_back(val);
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
        rf.pure_var_category[xi.name()] = TAIL;
        rf.update_var_category.push_back(rF_tail[k].pure_var_category);
        rf.update_var_category[0].insert(make_pair(rxo.x.name(), OUTER|SCAN));
        rf.update_var_category[0].insert(make_pair(rxo.y.name(), OUTER|SCAN));
        rf.update_var_category[0].erase(xo.name());
        rf.update_var_category[0].erase(xi.name());
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
            Image<float> weight = tail_weights(split_info, j, u);

            // weight matrix for accumulating completed tail elements from scan u to scan j
            // for a tile that is clamped on all borders
            Image<float> c_weight = weight;
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
        // Image<float> weight = tail_weights(split_info, j);
        Image<float> weight(order, order);
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

/**
 * Add the residuals to complete the tails of a particular dimension, the residual
 * for each tail is provided as a separate function which has to be simply added
 * to the corresponding tail - this residual includes contributions from all tails
 * of all scans preceeding itself
 * \param[in,out] rF_tail tails of each scan of given dimension
 * \param[in] rF_deps residuals to be added to each of the tails
 * \param[in] split_info tiling metadata of this dimension
 */
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

/**
 * Compute the residuals due to one dimension on residuals of another dimension;
 * this involves taking all the tails of the previous dimension and applying all
 * the scans of the current dimension and another other dimension in between -
 * number of functions generated is number of scans in prev dimension times
 * number of scans in current dimension; all these functions are lumped
 * together as a tuple to allow simultaneous computation.
 *
 * \param[in] rF_intra intra tile term that computes intra tile scans
 * \param[in,out] rF_tail  tails for each scan of current dimension, residuals
 * are added to these functions
 * \param[in] rF_tail_prev tails for each scan of prev dimension
 * \param[in] split_info      tiling metadata for current dimension
 * \param[in] split_info_prev tiling metadata for previous dimension
 * \returns function that computes the cross dimension residuals and
 * and another function that reindexes it (useful for computing in locally)
 */
static vector<RecFilterFunc> add_prev_dimension_residual_to_tails(
        RecFilterFunc         rF_intra,
        vector<RecFilterFunc> rF_tail,
        vector<RecFilterFunc> rF_tail_prev,
        SplitInfo split_info,
        SplitInfo split_info_prev)
{
    Var x    = split_info.var;
    Var xi   = split_info.inner_var;
    Var xo   = split_info.outer_var;
    int tile = split_info.tile_width;
    int num_tiles = split_info.num_tiles;

    Var  y    = split_info_prev.var;
    Var  yi   = split_info_prev.inner_var;
    Var  yo   = split_info_prev.outer_var;
    RDom ryi  = split_info_prev.inner_rdom;
    RDom ryt  = split_info_prev.tail_rdom;
    int  num_tiles_prev = split_info_prev.num_tiles;
    int  filter_dim_prev = split_info_prev.filter_dim;

    Function F_intra = rF_intra.func;
    vector<Function> F_tail;
    vector<Function> F_tail_prev;
    for (int i=0; i<rF_tail.size(); i++) {
        F_tail.push_back(rF_tail[i].func);
    }
    for (int i=0; i<rF_tail_prev.size(); i++) {
        F_tail_prev.push_back(rF_tail_prev[i].func);
    }

    // this stage will generate lots of functions that perform
    // scans within tiles and a reindexing function for each of
    // these scanning functions, store them separately
    vector<RecFilterFunc> intra_tile_funcs;
    vector<RecFilterFunc> reindex_funcs;

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

            // add all the scans from the intra tile term only for the scans
            // numbered between first_scan and last_scan, for other scans
            // create LHS = RHS like update defs
            for (int i=0; i<F_intra.updates().size(); i++) {
                map<string, VarTag> uvar_category = rF_intra.update_var_category[i];
                vector<Expr> args = F_intra.updates()[i].args;
                vector<Expr> values = F_intra.updates()[i].values;
                for (int v=0; v<ryi.dimensions(); v++) {
                    for (int u=0; u<args.size(); u++) {
                        args[u] = substitute(ryi[v].name(), ryt[v], args[u]);
                    }
                    if (uvar_category.find(ryi[v].name()) != uvar_category.end()) {
                        VarTag vc = uvar_category[ryi[v].name()];
                        if (v == filter_dim_prev) {
                            uvar_category.erase(ryi[v].name());
                            uvar_category.insert(make_pair(ryt[v].name(), TAIL));
                        } else {
                            uvar_category.erase(ryi[v].name());
                            uvar_category.insert(make_pair(ryt[v].name(), vc));
                        }
                    }
                }
                if (i>=first_scan && i<=last_scan) {
                    for (int v=0; v<ryi.dimensions(); v++) {
                        for (int u=0; u<values.size(); u++) {
                            values[u] = substitute_func_call(F_intra.name(), F_tail_prev_scanned_sub, values[u]);
                            values[u] = substitute(ryi[v].name(), ryt[v], values[u]);
                        }
                    }
                } else {
                    for (int u=0; u<values.size(); u++) {
                        values[u] = Call::make(F_tail_prev_scanned_sub, args, u);
                    }
                }
                F_tail_prev_scanned_sub.define_update(args, values);

                rF_tail_prev_scanned_sub.func_category = INTRA_1;
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

                rF_tail_prev_scanned.func = F_tail_prev_scanned;
            }

            // weight matrix for accumulating completed tail elements
            // of scan after applying all subsequent scans
            Image<float> weight  = tail_weights(split_info_prev, k, 0);

            // weight matrix for accumulating completed tail elements
            // of scan after applying all subsequent scans
            // for a tile that is clamped on all borders
            Image<float> c_weight = weight;
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
            rF_tail_prev_scanned.func_category     = REINDEX;
            rF_tail_prev_scanned.pure_var_category = rF_tail_prev_scanned_sub.pure_var_category;
            rF_tail_prev_scanned.callee_func       = rF_tail_prev_scanned_sub.func.name();

            intra_tile_funcs.push_back(rF_tail_prev_scanned_sub);
            reindex_funcs   .push_back(rF_tail_prev_scanned);
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

    // the list of intra tile scans functions and reindexing functions read the same
    // data and compute similar things, interleave them into a single function
    // also create a function to reindex the interleaved function
    RecFilterFunc rF;
    RecFilterFunc rF_reidx;
    {
        // interleave the functions by adding an extra dimension
        Var c("c");

        // interleaved result and a reindexing function
        Function F(rF_intra.func.name() + DASH + y.name()+ DASH + x.name() + DASH + SUB);
        Function F_reidx(rF_intra.func.name() + DASH + y.name()+ DASH + x.name());

        // copy scheduling tags from any of the functions
        rF.func          = F;
        rF.func_category = INTRA_1;
        rF.pure_var_category   = intra_tile_funcs[0].pure_var_category;
        rF.update_var_category = intra_tile_funcs[0].update_var_category;

        // add the extra dimension in the scheduling args
        rF.pure_var_category.insert(make_pair(c.name(), VarTag(INNER,0)));
        for (int i=0; i<rF.update_var_category.size(); i++) {
            rF.update_var_category[i].insert(make_pair(c.name(), VarTag(INNER,0)));
        }

        // scheduling tags for the reindexing function
        rF_reidx.func          = F_reidx;
        rF_reidx.func_category = REINDEX;
        rF_reidx.pure_var_category = rF.pure_var_category;
        rF_reidx.callee_func   = F.name();

        vector<string> args = intra_tile_funcs[0].func.args();
        vector<Expr> values = intra_tile_funcs[0].func.values();
        vector<UpdateDefinition> updates = intra_tile_funcs[0].func.updates();;

        // interleaving dimension
        int dimension = -1;
        for (int j=0; j<args.size(); j++) {
            if (args[j] == yi.name()) {
                dimension = j;
            }
        }
        if (dimension<0) {
            cerr << "Could not interleave functions because interleaving dimension not found" << endl;
            assert(false);
        }

        // create a dummy definition for the interleaved function its defintion becomes available
        // and other substitutions can be made
        F.define(args,values);

        // change the pure values to read from all the functions to be interleaved
        for (int i=0; i<intra_tile_funcs.size(); i++) {
            Function f = intra_tile_funcs[i].func;
            for (int j=0; j<values.size(); j++) {
                values[j] = simplify(select(c==i, f.values()[j], values[j]));
            }
        }

        // replace all calls to the functions to be interleaved by the interleaved function
        for (int i=0; i<intra_tile_funcs.size(); i++) {
            Function f = intra_tile_funcs[i].func;
            for (int k=0; k<updates.size(); k++) {
                for (int j=0; j<updates[k].values.size(); j++) {
                    Expr v1  = substitute_func_call(f.name(), F, updates[k].values[j]);
                    Expr v2  = substitute_func_call(f.name(), F, f.updates()[k].values[j]);
                    updates[k].values[j] = simplify(select(c==i, v2, v1));
                }
            }
        }

        // add the extra dimension and redefine the function
        args.insert(args.begin()+dimension, c.name());
        F.clear_all_definitions();
        F.define(args,values);
        for (int k=0; k<updates.size(); k++) {
            updates[k].args.insert(updates[k].args.begin()+dimension, c);
            for (int j=0; j<updates[k].values.size(); j++) {
                Expr v = insert_arg_in_func_call(F.name(), dimension, c, updates[k].values[j]);
                updates[k].values[j] = v;
            }
            F.define_update(updates[k].args, updates[k].values);
        }

        // create the reindexing function
        {
            vector<Expr> call_args;
            vector<Expr> values;
            for (int j=0; j<F.args().size(); j++) {
                call_args.push_back(Var(F.args()[j]));
            }
            for (int j=0; j<F.values().size(); j++) {
                values.push_back(Call::make(F,call_args,j));
            }
            F_reidx.define(F.args(), values);
        }

        // change all functions to index into the interleaved function
        // and mark them as inline
        for (int i=0; i<intra_tile_funcs.size(); i++) {
            RecFilterFunc& rf = intra_tile_funcs[i];
            Function f        = rf.func;

            vector<string> args = f.args();
            vector<Expr> call_args;
            vector<Expr> values;

            for (int j=0; j<args.size(); j++) {
                if (args[j] == yi.name()) {
                    call_args.push_back(Var(yi));
                } else {
                    call_args.push_back(Var(args[j]));
                }
            }
            call_args.insert(call_args.begin()+dimension, i);
            for (int j=0; j<f.values().size(); j++) {
                values.push_back(Call::make(F_reidx, call_args, j));
            }
            f.clear_all_definitions();
            f.define(args, values);
            rf.func_category = INLINE;
            rf.pure_var_category.clear();
            rf.update_var_category.clear();
            rf.callee_func.clear();
            rf.caller_func.clear();
        }

        // convert all all reindexing functions to inline
        for (int i=0; i<reindex_funcs.size(); i++) {
            reindex_funcs[i].func_category = INLINE;
            reindex_funcs[i].pure_var_category.clear();
            reindex_funcs[i].update_var_category.clear();
            reindex_funcs[i].callee_func.clear();
            reindex_funcs[i].caller_func.clear();
        }

        // remove all redundant update defs where LHS = RHS
        {
            vector<string> args = F.args();
            vector<Expr> values = F.values();
            vector<UpdateDefinition> updates = F.updates();

            vector< map<string,VarTag> > uvar_category = rF.update_var_category;

            F.clear_all_definitions();
            F.define(args, values);
            rF.update_var_category.clear();

            for (int j=0; j<updates.size(); j++) {
                vector<Expr> u_args = updates[j].args;
                vector<Expr> u_vals = updates[j].values;
                bool lhs_equals_rhs = true;
                for (int k=0; k<u_vals.size(); k++) {
                    Expr lhs = Call::make(F, u_args, k);
                    Expr rhs = u_vals[k];
                    lhs_equals_rhs &= equal(lhs,rhs);
                }
                if (!lhs_equals_rhs) {
                    F.define_update(u_args, u_vals);
                    rF.update_var_category.push_back(uvar_category[j]);
                }
            }
        }

        // never expose the extra dimension to the user, fuse it with the
        // interleaving dimension
        // Func(F).fuse(yi,c,yi);
        // Func(F_reidx).fuse(yi,c,yi);
    }

    return {rF, rF_reidx};
}

// -----------------------------------------------------------------------------

/**
 * Add the residuals to the final result: the residual from each scan is available
 * is a separate Func and the final result applies all the scans within tiles;
 * modify this such that the residuals of each scan are added to the first k
 * elements of the tile before applying the scan, the scan itself begins from
 * element k+1 and propagates the effect of residuals to the whole tile.
 * \param[in,out] rF final term that computes intra tile scans without residuals
 * \param[in] rF_deps list of residual functions for each scan of each dimension
 * \param[in] split_info tiling metadata
 */
static void add_residuals_to_final_result(
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
    map<int, pair< vector<Expr>, vector<Expr> > > new_updates;
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
                if (new_update_var_category[curr_scan].find(rxi[k].name()) != new_update_var_category[curr_scan].end()) {
                    VarTag v = new_update_var_category[curr_scan][rxi[k].name()];
                    new_update_var_category[curr_scan].erase(rxi[k].name());
                    new_update_var_category[curr_scan].insert(make_pair(rxt[k].name(), v));
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
                if (update_var_category[curr_scan].find(rxi[k].name()) != update_var_category[curr_scan].end()) {
                    VarTag vc = update_var_category[curr_scan][rxi[k].name()];
                    update_var_category[curr_scan].erase(rxi[k].name());
                    update_var_category[curr_scan].insert(make_pair(rxf[k].name(), vc));
                }
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

    // add a final update def to resolve bank conflicts
    // add one pixel to the innermost split dimension
    {
        int innermost_tiled_dim_id = -1;
        int innermost_tiled_dim_buffer_width = 0;

        // find the innermost tiled dimension
        for (int i=0; i<F.args().size(); i++) {
            string arg = F.args()[i];
            for (int j=0; innermost_tiled_dim_id<0 && j<split_info.size(); j++) {
                if (arg==split_info[j].inner_var.name()) {
                    innermost_tiled_dim_id = j;
                    innermost_tiled_dim_buffer_width = split_info[j].tile_width;
                }
            }
        }
        assert(innermost_tiled_dim_id>=0);

        vector<Expr> args;
        vector<Expr> values;
        for (int i=0; i<F.args().size(); i++) {
            if (i==innermost_tiled_dim_id) {
                args.push_back(innermost_tiled_dim_buffer_width);
            } else {
                args.push_back(Var(F.args()[i]));
            }
        }
        args[innermost_tiled_dim_id] = innermost_tiled_dim_buffer_width;
        for (int i=0; i<F.outputs(); i++) {
            values.push_back(undef(F.output_types()[i]));
        }
        F.define_update(args, values);

        // add scheduling tags for the last update def
        map<string,VarTag> uvar_category = rF.pure_var_category;
        uvar_category.erase(F.args()[innermost_tiled_dim_id]);
        rF.update_var_category.push_back(uvar_category);
    }
}

// -----------------------------------------------------------------------------

static vector< vector<RecFilterFunc> > split_scans(
        RecFilterFunc F_intra,
        vector< vector<RecFilterFunc> > F_tail,
        string final_result_func,
        vector<SplitInfo> &split_info)
{
    vector< vector<RecFilterFunc> > F_ctail_list;
    vector< vector<RecFilterFunc> > F_deps_list;

    for (int i=0; i<split_info.size(); i++) {
        string x = split_info[i].var.name();

        string s1 = F_intra.func.name() + DASH + INTER_TILE_TAIL_SUM   + DASH + x;
        string s2 = F_intra.func.name() + DASH + COMPLETE_TAIL_RESIDUAL+ DASH + x;
        string s3 = F_intra.func.name() + DASH + FINAL_RESULT_RESIDUAL + DASH + x;

        // all these recfilter func have already been added to the global list,
        // no need to add again
        vector<RecFilterFunc> F_ctail  = create_complete_tail_term (F_tail[i],split_info[i], s1);
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
            vector<RecFilterFunc> g = add_prev_dimension_residual_to_tails(
                    F_intra, F_ctail, F_ctail_list[j], split_info[i], split_info[j]);
            for (int k=0; k<g.size(); k++) {
                recfilter_func_list.insert(make_pair(g[k].func.name(), g[k]));
            }
        }

        // add all the generated functions to the global list
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

        // store the complete tail terms, used in next dimension
        F_ctail_list.push_back(F_ctailw);

        // store the residuals to the added to the final result
        F_deps_list.push_back(F_deps);
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
            found = true;

            // no need to add a split if tile width is not same as image width
            assert(contents.ptr->filter_info[j].tile_width == tile_width);
            if (contents.ptr->filter_info[j].image_width == tile_width) {
                continue;
            }

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
        }
        if (!found) {
            cerr << "Variable " << x << " does not correspond to any "
                << "dimension of the recursive filter " << contents.ptr->name << endl;
            assert(false);
        }
    }

    // return if there are no splits to apply
    if (recfilter_split_info.empty()) {
        return;
    }

    // apply the actual splitting
    RecFilterFunc rF_final;
    {
        // compute the intra tile result
        RecFilterFunc rF_intra = create_intra_tile_term(rF, recfilter_split_info);

        // create a term for the tail of each intra tile scan
        vector< vector<RecFilterFunc> > rF_tail = create_intra_tail_term(
                rF_intra, recfilter_split_info);

        // create a function will hold the final result, copy of the intra tile term
        rF_final = create_copy(rF_intra, F.name() + DASH + FINAL_TERM);

        // compute the residuals from splits in each dimension
        vector< vector<RecFilterFunc> > rF_deps = split_scans(rF_intra, rF_tail,
                rF_final.func.name(), recfilter_split_info);

        // transfer the tail of each scan to another buffer
        RecFilterFunc rF_intra_tail = extract_tails_from_each_scan(rF_intra, rF_tail, recfilter_split_info);

        // add all the residuals to the final term
        add_residuals_to_final_result(rF_final, rF_deps, recfilter_split_info);

        // add the intra, final and tail terms to the list functions
        recfilter_func_list.insert(make_pair(rF_intra.func.name(), rF_intra));
        recfilter_func_list.insert(make_pair(rF_final.func.name(), rF_final));
        recfilter_func_list.insert(make_pair(rF_intra_tail.func.name(), rF_intra_tail));
        for (int i=0; i<rF_tail.size(); i++) {
            for (int j=0; j<rF_tail[i].size(); j++) {
                recfilter_func_list.insert(make_pair(rF_tail[i][j].func.name(), rF_tail[i][j]));
            }
        }
    }

    // change the original function to index into the final term
    {
        Function F_final = rF_final.func;
        vector<string> args = F.args();
        vector<Expr> values;
        vector<Expr> call_args;
        for (int i=0; i<F_final.args().size(); i++) {
            string arg = F_final.args()[i];
            call_args.push_back(Var(arg));

            for (int j=0; j<recfilter_split_info.size(); j++) {
                Var var        = recfilter_split_info[j].var;
                Var inner_var  = recfilter_split_info[j].inner_var;
                Var outer_var  = recfilter_split_info[j].outer_var;
                int tile_width = recfilter_split_info[j].tile_width;
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

        // remove the scheduling tags of the update defs
        rF.func_category = REINDEX;
        rF.callee_func = F_final.name();
        rF.update_var_category.clear();

        // split the tiled vars of the final term
        for (int i=0; i<recfilter_split_info.size(); i++) {
            Var var        = recfilter_split_info[i].var;
            Var inner_var  = recfilter_split_info[i].inner_var;
            Var outer_var  = recfilter_split_info[i].outer_var;
            int tile_width = recfilter_split_info[i].tile_width;

            Func(F).split(var, outer_var, inner_var, tile_width);

            string s = "split(Var(\"" + var.name() + "\"), Var(\"" + outer_var.name()
                + "\"), Var(\"" + inner_var.name() + "\"), " + int_to_string(tile_width) + ")";

            rF.pure_var_category.erase(var.name());
            rF.pure_var_category.insert(make_pair(inner_var.name(), VarTag(INNER,i)));
            rF.pure_var_category.insert(make_pair(outer_var.name(), VarTag(OUTER,i)));
            rF.pure_schedule.push_back(s);
        }

        recfilter_func_list.insert(make_pair(rF.func.name(), rF));
    }

    // add all the generated RecFilterFuncs
    contents.ptr->func.insert(recfilter_func_list.begin(), recfilter_func_list.end());

    // apply bounds on all dimensions of all functions
    for (int i=0; i<recfilter_split_info.size(); i++) {
        string x = recfilter_split_info[i].var.name();
        string xi= recfilter_split_info[i].inner_var.name();
        string xo= recfilter_split_info[i].outer_var.name();
        int    w = recfilter_split_info[i].image_width;
        int    tw= recfilter_split_info[i].tile_width;
        int    nt= w/tw;

        map<string,RecFilterFunc>::iterator fit;
        for (fit=contents.ptr->func.begin(); fit!=contents.ptr->func.end(); fit++) {
            if (fit->second.func_category == INLINE) {
                continue;
            }
            Func F(fit->second.func);
            for (int j=0; j<F.args().size(); j++) {
                string v = F.args()[j].name();
                VarTag vt= fit->second.pure_var_category[v];
                if (v==x) { //if (vt.check(FULL)) {
                    F.bound(v,0,w);
                }
                //else if (v==xo) { //else if (vt.check(OUTER)) {
                //    F.bound(v,0,nt);
                //}
            }
        }
    }

    contents.ptr->tiled = true;

    // perform generic and target dependent optimizations
    finalize();

    recfilter_func_list.clear();
    recfilter_split_info.clear();
}

void RecFilter::split_all_dimensions(int tx) {
    map<string,int> dim_tile;
    for (int i=0; i<contents.ptr->filter_info.size(); i++) {
        if (contents.ptr->filter_info[i].num_scans) {
            dim_tile.insert(make_pair(contents.ptr->filter_info[i].var.name(),tx));
        }
    }
    split(dim_tile);
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
        // reassign var tag counts for all functions
        for (fit=contents.ptr->func.begin(); fit!=contents.ptr->func.end(); fit++) {
            reassign_vartag_counts(fit->second.pure_var_category);
            for (int i=0; i<fit->second.update_var_category.size(); i++) {
                reassign_vartag_counts(fit->second.update_var_category[i]);
            }
        }

        // inline all functions not required any more
        for (fit=contents.ptr->func.begin(); fit!=contents.ptr->func.end(); fit++) {
            if (fit->second.func_category == INLINE) {
                inline_func(fit->second.func.name());
                fit = contents.ptr->func.begin();            // list changed, start all over again
            }
        }

        // check that all intra tile scans are reindexed by a unique function
        for (fit=contents.ptr->func.begin(); fit!=contents.ptr->func.end(); fit++) {
            RecFilterFunc rF = fit->second;
            if (rF.func_category==INTRA_1 || rF.func_category==INTRA_N) {
                int num_reindexing_funcs = 0;
                map<string,RecFilterFunc>::iterator git;
                for (git=contents.ptr->func.begin(); git!=contents.ptr->func.end(); git++) {
                    if (git->second.func_category==REINDEX && git->second.callee_func==rF.func.name()) {
                        num_reindexing_funcs++;
                    }
                }
                if (num_reindexing_funcs>1) {
                    cerr << rF.func.name() << " is reindexed by multiple functions. "
                        << "This will make compute_locally schedules ambiguous" << endl;
                    assert(false);
                }
            }
        }

        // move initialization to update def in intra tile computation stages
        for (fit=contents.ptr->func.begin(); fit!=contents.ptr->func.end(); fit++) {
            RecFilterFunc& rF = fit->second;
            if (rF.func_category==INTRA_N) {
                convert_pure_def_into_first_update_def(rF, recfilter_split_info);
            }
        }
    }

    contents.ptr->finalized = true;
}
