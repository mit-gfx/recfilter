#include "split.h"
#include "split_utils.h"

#include <algorithm>

#define SPLIT_HELPER_NAME '-'

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
