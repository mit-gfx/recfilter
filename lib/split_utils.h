#ifndef _SPLIT_UTILS_H_
#define _SPLIT_UTILS_H_

#include <vector>
#include <string>

#include <Halide.h>

struct SplitInfo {
    bool scan_causal;
    int scan_stage;
    int filter_order;
    int filter_dim;
    int scan_id;

    Halide::Var var;
    Halide::Var inner_var;
    Halide::Var outer_var;

    Halide::RDom rdom;
    Halide::RDom split_rdom;
    Halide::RDom outer_rdom;
    Halide::RDom inner_rdom;

    Halide::Expr tile_width;
    Halide::Expr image_width;
    Halide::Expr num_tiles;

    Halide::Image<float> complete_tail_weight;
    Halide::Image<float> complete_result_weight;

    Halide::Internal::Function intra_tile_scan;
    Halide::Internal::Function incomplete_tail;
    Halide::Internal::Function complete_tail;
    Halide::Internal::Function dependencies;
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

Halide::Image<float> weight_matrix_A_FB(
        Halide::Image<float> filter_weights,
        int scan_id,
        int tile_width);

Halide::Image<float> weight_matrix_A_FP(
        Halide::Image<float> filter_weights,
        int scan_id,
        int tile_width);

Halide::Image<float> weight_matrix_transpose(
        Halide::Image<float> A);

Halide::Image<float> weight_matrix_mult(
        Halide::Image<float> A,
        Halide::Image<float> B);

// -----------------------------------------------------------------------------

std::vector<SplitInfo> group_scans_by_dimension(
        Halide::Internal::Function F,
        vector<SplitInfo> split_info);

std::vector<SplitInfo> group_scans_by_stage(
        Halide::Internal::Function F,
        vector<SplitInfo> split_info);

// -----------------------------------------------------------------------------

#endif // SPLIT_UTILS_H_
