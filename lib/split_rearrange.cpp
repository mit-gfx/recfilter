#include "split.h"
#include "split_utils.h"

#include <algorithm>

#define SPLIT_HELPER_NAME '-'

using namespace Halide;
using namespace Halide::Internal;

using std::vector;

// -----------------------------------------------------------------------------

vector<SplitInfo> group_scans_by_stage(Function F, vector<SplitInfo> split_info) {
//    vector<string> args = F.args();
//    vector<Expr>  values = F.values();
//    vector<ReductionDefinition> reductions = F.reductions();
//
//    // extract the max scan stage value across all the reductions
//    int max_scan_stage = -1;
//    for (int i=0; i<split_info.size(); i++) {
//        max_scan_stage = std::max(max_scan_stage, split_info[i].scan_stage);
//    }
//    assert(max_scan_stage >= 0);
//
//    // nothing to change if there is only one scan stage
//    if (max_scan_stage == 0)
//        return split_info;
//
//    // list of scans with scan stages 0, 1, 2 ...
//    vector<vector<int> > scan_stages(max_scan_stage+1);
//    for (int i=0; i<split_info.size(); i++) {
//        scan_stages[ split_info[i].scan_stage ].push_back(split_info[i].scan_id);
//    }
//
//    vector<ReductionDefinition> new_reductions;
    vector<SplitInfo> new_split_info;
//
//    // use all scans with stage 0 at first, then 1 and so on
//    std::set<int> new_order;
//    for (int i=0; i<scan_stages.size(); i++) {
//        std::sort(scan_stages[i].begin(), scan_stages[i].end());
//        for (int j=0; j<scan_stages[i].size(); j++) {
//            assert(new_order.find(scan_stages[i][j]) == new_order.end());
//            new_reductions.push_back(reductions[ scan_stages[i][j] ]);
//            new_order.insert(scan_stages[i][j]);
//
//            // find the split_info struct for this scan and update its scan_id
//            for (int k=0; k<split_info.size(); k++) {
//                if (split_info[k].scan_id == scan_stages[i][j]) {
//                    SplitInfo s = split_info[k];
//                    s.scan_id = new_reductions.size()-1;
//                    new_split_info.insert(new_split_info.begin(), s);
//                }
//            }
//        }
//    }
//    assert(new_reductions.size() == reductions.size());
//
//
//    // reorder the reduction definitions as per the new order
//    F.clear_all_definitions();
//    F.define(args, values);
//    for (int i=0; i<new_reductions.size(); i++) {
//        F.define_reduction(new_reductions[i].args, new_reductions[i].values);
//    }
//
    assert(split_info.size() == new_split_info.size());
    return new_split_info;
}

vector<SplitInfo> group_scans_by_dimension(Function F, vector<SplitInfo> split_info) {
    vector<string> args = F.args();
    vector<Expr>  values = F.values();
    vector<ReductionDefinition> reductions = F.reductions();

    // dimension -> (reduction defintion id, split_info id)
    vector< vector<std::pair<int,int> > > dim_to_scan(args.size());

    for (int i=0; i<split_info.size(); i++) {
        int curr = split_info.size()-1-i;
        int dimension = split_info[curr].filter_dim;
        int scan_id   = split_info[curr].scan_id;
        dim_to_scan[dimension].push_back(std::make_pair(scan_id,curr));
    }

    vector<ReductionDefinition> new_reductions;
    vector<SplitInfo>           new_split_info;

    // use all scans with dimension 0 first, then 1 and so on
    // splits must be reverse order of scans
    for (int i=0; i<dim_to_scan.size(); i++) {
        for (int j=0; j<dim_to_scan[i].size(); j++) {
            int scan  = dim_to_scan[i][j].first;
            int split = dim_to_scan[i][j].second;

            ReductionDefinition r = reductions[scan];
            SplitInfo           s = split_info[split];

            s.scan_id = new_reductions.size();
            new_reductions.push_back(r);
            new_split_info.insert(new_split_info.begin(), s);
        }
    }
    assert(new_reductions.size() == reductions.size());
    assert(split_info.size() == new_split_info.size());

    // reorder the reduction definitions as per the new order
    F.clear_all_definitions();
    F.define(args, values);
    for (int i=0; i<new_reductions.size(); i++) {
        F.define_reduction(new_reductions[i].args, new_reductions[i].values);
    }

    // assign scan stages to each scan: increment scan stage if a scan
    // is preceeded by a scan in same dimension
    for (int i=0; i<new_split_info.size(); i++) {
        int curr = new_split_info.size()-1-i;
        int prev = curr+1;
        if (curr == new_split_info.size()-1) {
            new_split_info[curr].scan_stage = 0;
        }
        else {
            bool scan_causal     = new_split_info[curr].scan_causal;
            int  dimension       = new_split_info[curr].filter_dim;
            bool prev_scan_causal= new_split_info[prev].scan_causal;
            int  prev_dimension  = new_split_info[prev].filter_dim;
            int  prev_scan_stage = new_split_info[prev].scan_stage;

            //if (dimension==prev_dimension && scan_causal==prev_scan_causal) {
            //    new_split_info[curr].scan_stage = prev_scan_stage+1;
            //} else {
            //    new_split_info[curr].scan_stage = prev_scan_stage;
            //}

            if (dimension == prev_dimension) {
                new_split_info[curr].scan_stage = prev_scan_stage+1;
            } else {
                new_split_info[curr].scan_stage = prev_scan_stage;
            }
        }
    }
    return new_split_info;
}
