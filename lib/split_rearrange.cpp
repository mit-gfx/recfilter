#include "split.h"
#include "split_utils.h"

#include <algorithm>

#define SPLIT_HELPER_NAME '-'

using namespace Halide;
using namespace Halide::Internal;

using std::string;
using std::cerr;
using std::endl;
using std::vector;
using std::set;

// -----------------------------------------------------------------------------

vector<SplitInfo> group_scans_by_stage(Function F, vector<SplitInfo> split_info) {
    vector<string> args = F.args();
    vector<Expr>  values = F.values();
    vector<ReductionDefinition> reductions = F.reductions();

    // extract the max scan stage value across all the reductions
    int max_scan_stage = -1;
    for (int i=0; i<split_info.size(); i++) {
        max_scan_stage = std::max(max_scan_stage, split_info[i].scan_stage);
    }
    assert(max_scan_stage >= 0);

    // nothing to change if there is only one scan stage
    if (max_scan_stage == 0)
        return split_info;

    // list of scans with scan stages 0, 1, 2 ...
    vector<vector<int> > scan_stages(max_scan_stage+1);
    for (int i=0; i<split_info.size(); i++) {
        scan_stages[ split_info[i].scan_stage ].push_back(split_info[i].scan_id);
    }

    vector<ReductionDefinition> new_reductions;
    vector<SplitInfo> new_split_info;

    // use all scans with stage 0 at first, then 1 and so on
    set<int> new_order;
    for (int i=0; i<scan_stages.size(); i++) {
        std::sort(scan_stages[i].begin(), scan_stages[i].end());
        for (int j=0; j<scan_stages[i].size(); j++) {
            assert(new_order.find(scan_stages[i][j]) == new_order.end());
            new_reductions.push_back(reductions[ scan_stages[i][j] ]);
            new_order.insert(scan_stages[i][j]);

            // find the split_info struct for this scan and update its scan_id
            for (int k=0; k<split_info.size(); k++) {
                if (split_info[k].scan_id == scan_stages[i][j]) {
                    SplitInfo s = split_info[k];
                    s.scan_id = new_reductions.size()-1;
                    new_split_info.insert(new_split_info.begin(), s);
                }
            }
        }
    }
    assert(new_reductions.size() == reductions.size());


    // reorder the reduction definitions as per the new order
    F.clear_all_definitions();
    F.define(args, values);
    for (int i=0; i<new_reductions.size(); i++) {
        F.define_reduction(new_reductions[i].args, new_reductions[i].values);
    }

    assert(split_info.size() == new_split_info.size());
    return new_split_info;
}

vector<SplitInfo> group_scans_by_dimension(Function F, vector<SplitInfo> split_info) {
    vector<SplitInfo> new_split_info;

    new_split_info = split_info;

    assert(split_info.size() == new_split_info.size());
    return new_split_info;
}
