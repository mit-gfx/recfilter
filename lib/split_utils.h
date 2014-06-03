#ifndef _SPLIT_UTILS_H_
#define _SPLIT_UTILS_H_

#include <vector>
#include <string>

#include <Halide.h>

struct SplitInfo {
    int filter_order;
    int filter_dim;
    int num_splits;

    Halide::Expr tile_width;
    Halide::Expr image_width;
    Halide::Expr num_tiles;

    Halide::Var var;
    Halide::Var inner_var;
    Halide::Var outer_var;

    Halide::Image<float> filter_weights;
    Halide::Internal::Function intra_tile_scan;

    vector<bool> scan_causal;
    vector<int> scan_stage;
    vector<int> scan_id;

    vector<Halide::RDom> rdom;
    vector<Halide::RDom> split_rdom;
    vector<Halide::RDom> outer_rdom;
    vector<Halide::RDom> inner_rdom;

    vector<Halide::Image<float> > complete_tail_weight;
    vector<Halide::Image<float> > complete_result_weight;
};


// -----------------------------------------------------------------------------

// Checks

bool check_causal_scan(
        Halide::Internal::Function f,
        Halide::RVar rx,
        int scan_id,
        int dimension);

void check_split_feasible(
        Halide::Func& func,
        vector<int>  dimension,
        vector<Halide::Var>  var,
        vector<Halide::Var>  inner_var,
        vector<Halide::Var>  outer_var,
        vector<Halide::RDom> rdom,
        vector<Halide::RDom> inner_rdom,
        vector<int>  order);

bool check_for_pure_split(
        Halide::Internal::Function F,
        SplitInfo split_info);

// -----------------------------------------------------------------------------

// Weight matrix computation

Halide::Image<float> tail_weights(SplitInfo s, int split_id1);
Halide::Image<float> tail_weights(SplitInfo s, int split_id1, int split_id2);

// -----------------------------------------------------------------------------

std::vector<SplitInfo> group_scans_by_dimension(
        Halide::Internal::Function F,
        vector<SplitInfo> split_info);

// -----------------------------------------------------------------------------

#endif // SPLIT_UTILS_H_
