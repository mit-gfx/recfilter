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

    // assign scan stages to each scan: increment scan stage if a scan
    // is preceeded by a scan in same dimension
    for (int i=0; i<new_split_info.size(); i++) {
        for (int j=0; j<new_split_info[i].num_splits; j++) {
            int curr = new_split_info[i].num_splits-1-j;
            new_split_info[i].scan_stage[curr] = j;
        }
    }
    return new_split_info;
}

// -----------------------------------------------------------------------------

void add_intra_tile_scan_stages(Function F_intra, vector<SplitInfo> split_info) {
    vector<string> pure_args   = F_intra.args();
    vector<Expr>   pure_values = F_intra.values();
    vector<ReductionDefinition> reductions = F_intra.reductions();

    Var xs(SCAN_STAGE_ARG);

    int scan_stage_var_index = pure_args.size();

    // add the scan stage as an arg to the pure defintion
    pure_args.insert(pure_args.begin()+scan_stage_var_index, xs.name());

    // change the RHS to read the input only on
    for (int i=0; i<pure_values.size(); i++) {
        pure_values[i] = select(xs==0, pure_values[i], make_zero(pure_values[i].type()));
    }

    F_intra.clear_all_definitions();
    F_intra.define(pure_args, pure_values);

    // add actual value of scan stage for each reduction definition
    // on the LHS and RHS
    for (int u=0; u<split_info.size(); u++) {
        for (int v=0; v<split_info[u].num_splits; v++) {
            int scan_id    = split_info[u].scan_id[v];
            int scan_stage = split_info[u].scan_stage[v];
            reductions[scan_id].args.insert(reductions[scan_id].args.begin()+scan_stage_var_index,
                    scan_stage);
            for (int j=0; j<reductions[scan_id].values.size(); j++) {
                Expr val = reductions[scan_id].values[j];
                val = insert_arg_in_func_call(F_intra.name(), scan_stage_var_index, scan_stage, val);
                reductions[scan_id].values[j] = val;
            }
        }
    }

    for (int i=0; i<reductions.size(); i++) {
        F_intra.define_reduction(reductions[i].args, reductions[i].values);
    }
}

// -----------------------------------------------------------------------------

void fix_intra_tile_scan_stages(Function F_intra) {
    vector<string> pure_args = F_intra.args();
    vector<Expr> pure_values = F_intra.values();
    vector<ReductionDefinition> reductions = F_intra.reductions();

    // pure definitions remain unchanged
    F_intra.clear_all_definitions();
    F_intra.define(pure_args, pure_values);

    int scan_stage_var_index = -1;
    for (int i=0; i<pure_args.size(); i++) {
        if (pure_args[i] == SCAN_STAGE_ARG) {
            scan_stage_var_index = i;
        }
    }

    // add extra update steps to transfer data from one scan stage to another
    for (int i=0; i<reductions.size(); i++) {
        if (i>0) {
            // scan stage of previous and current scan
            Expr curr_scan_stage = reductions[i]  .args[scan_stage_var_index];
            Expr prev_scan_stage = reductions[i-1].args[scan_stage_var_index];

            // LHS and RHS differ only by scan stage value
            vector<Expr> args      = reductions[i].args;
            vector<Expr> call_args = reductions[i].args;
            call_args[scan_stage_var_index] = prev_scan_stage;

            // add the extra scan stage in between if consecutive scans
            // have different stages
            if (!equal(prev_scan_stage, curr_scan_stage)) {
                vector<Expr> values;
                for (int j=0; j<reductions[i].values.size(); j++) {
                    values.push_back(Call::make(F_intra, call_args, j));
                }
                F_intra.define_reduction(args, values);
            }
        }
        F_intra.define_reduction(reductions[i].args, reductions[i].values);
    }
}
