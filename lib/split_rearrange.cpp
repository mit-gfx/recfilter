#include "split.h"
#include "split_macros.h"
#include "split_utils.h"

#include <algorithm>

using namespace Halide;
using namespace Halide::Internal;

using std::vector;

// -----------------------------------------------------------------------------

vector<SplitInfo> group_scans_by_dimension(Function F, vector<SplitInfo> split_info) {
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

void extract_tails_from_each_scan(Function F_intra, vector<SplitInfo> split_info) {
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
