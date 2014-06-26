#ifndef _SPLIT_UTILS_H_
#define _SPLIT_UTILS_H_

#include <vector>
#include <string>

#include <Halide.h>

struct SplitInfo {
    bool clamp_border;

    int filter_order;
    int filter_dim;
    int num_splits;

    Halide::Expr tile_width;
    Halide::Expr image_width;
    Halide::Expr num_tiles;

    Halide::Var var;
    Halide::Var inner_var;
    Halide::Var outer_var;

    vector<bool> scan_causal;
    vector<int> scan_stage;
    vector<int> scan_id;

    vector<Halide::RDom> rdom;
    vector<Halide::RDom> outer_rdom;
    vector<Halide::RDom> inner_rdom;

    Halide::Image<float> feedfwd_coeff;
    Halide::Image<float> feedback_coeff;
};


// -----------------------------------------------------------------------------

// Checks

bool check_causal_scan(
        Halide::Internal::Function f,
        Halide::RVar rx,
        int scan_id,
        int dimension);

void check_split_feasible(
        Halide::Func func,
        vector<int>  dimension,
        vector<Halide::Var>  var,
        vector<Halide::Var>  inner_var,
        vector<Halide::Var>  outer_var,
        vector<Halide::RDom> rdom,
        vector<Halide::RDom> inner_rdom,
        vector<int>  order);

// -----------------------------------------------------------------------------

// Weight matrix computation

Halide::Image<float> tail_weights(SplitInfo s, int s1, bool clamp_border=false);
Halide::Image<float> tail_weights(SplitInfo s, int s1, int s2, bool clamp_border=false);

// -----------------------------------------------------------------------------

// Rearrangements for intra tile computation

std::vector<SplitInfo> group_scans_by_dimension(
        Halide::Internal::Function F,
        vector<SplitInfo> split_info);

void fix_intra_tile_scan_stages(Halide::Internal::Function F_intra);

void add_intra_tile_scan_stages(
        Halide::Internal::Function F_intra,
        std::vector<SplitInfo> split_info);

// -----------------------------------------------------------------------------

#endif // SPLIT_UTILS_H_
