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

void fix_intra_tile_scan_stages(Function F_intra, vector<SplitInfo> split_info) {
    vector<string> pure_args = F_intra.args();
    vector<Expr> pure_values = F_intra.values();
    vector<ReductionDefinition> reductions = F_intra.reductions();

    // pure definitions remain unchanged
    F_intra.clear_all_definitions();
    F_intra.define(pure_args, pure_values);

    // new reductions to be added for all the split reductions
    map<int, std::pair< vector<Expr>, vector<Expr> > > new_reductions;

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

    // add extra update steps to extract tail of each scan
    // which copies the tail to another buffer
    for (int i=0; i<reductions.size(); i++) {
        F_intra.define_reduction(reductions[i].args, reductions[i].values);
        if (new_reductions.find(i) != new_reductions.end()) {
            vector<Expr> args   = new_reductions[i].first;
            vector<Expr> values = new_reductions[i].second;
            F_intra.define_reduction(args, values);
        }
    }
}
