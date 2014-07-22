#include "recfilter.h"
#include "recfilter_utils.h"

using namespace Halide;
using namespace Halide::Internal;

using std::string;
using std::cerr;
using std::endl;
using std::vector;
using std::queue;
using std::map;

RecFilter::RecFilter(string n) {
}

void RecFilter::setArgs(vector<Var> args, vector<Expr> width) {
}

void RecFilter::define(Expr pure_def) {
}

void RecFilter::addScan(bool causal, vector<float> feedback) {
}

Func RecFilter::func(std::string func_name) {
}

void RecFilter::split(vector<Var> dims, vector<Expr> tile) {
    Function F = func.function();

    int num_splits = var.size();

    vector<Expr> tile_width;
    vector<RDom> outer_rdom;
    vector<RDom> tail_rdom;
    vector<Expr> image_width;
    vector<Expr> num_tiles;

    for (int i=0; i<num_splits; i++) {
        // individual tile boundaries
        Expr inner_rdom_extent = simplify(inner_rdom[i].x.extent());

        tile_width .push_back(inner_rdom_extent);
        image_width.push_back(rdom[i].x.extent());
        num_tiles  .push_back(image_width[i]/tile_width[i]);

        // tile width for splitting multiple scans in same dimension must be same
        for (int j=0; j<i-1; j++) {
            if (dimension[j] == dimension[i] && !equal(tile_width[i], tile_width[j])) {
                cerr << "Different tile widths specified for splitting same dimension" << endl;
                assert(false);
            }
        }

        // extent of reduction along dimensions to be split
        assert(extract_params_in_expr(rdom[i].x.extent()).size()==1 &&
                "RDom extent must have a single image parameter");

        // outer_rdom.x: over tail elems of prev tile to compute tail of current tile
        // outer_rdom.y: over all tail elements of current tile
        // outer_rdom.z: over all tiles
        outer_rdom.push_back(RDom(
                    0,order[i],
                    0,order[i],
                    1, simplify(num_tiles[i]-1),
                    "r"+var[i].name()+"o"));

        // tail_rdom.x: over all tail elems of current tile
        tail_rdom.push_back(RDom(0,order[i], "r"+var[i].name()+"t"));
    }

    // list of structs with all the info about split in each dimension
    vector<SplitInfo> split_info(F.args().size());

    // loop over all reduction definitions in reverse order and
    // populate the split_info struct with info on splitting each reduction
    for (int i=F.reductions().size()-1; i>=0; i--) {
        assert(F.reductions()[i].domain.defined() &&
                "Reduction definition has no reduction domain");

        int index = -1;

        // extract the RDom in the reduction definition and
        // compare with the reduction domain to be split
        for (int j=0; j<num_splits; j++) {
            if (F.reductions()[i].domain.same_as(rdom[j].domain())) {
                index = j;
            }
        }

        if (index < 0) {
            cerr << "Could not find a split for scan " << i << endl;
            assert(false);
        }

        SplitInfo s = split_info[ dimension[index] ];

        s.clamp_border = false;
        s.filter_order = order[index];
        s.filter_dim   = dimension[index];
        s.var          = var[index];
        s.inner_var    = inner_var[index];
        s.outer_var    = outer_var[index];
        s.image_width  = image_width[index];
        s.tile_width   = tile_width[index];
        s.num_tiles    = num_tiles[index];
        s.feedfwd_coeff  = feedfwd_coeff;
        s.feedback_coeff = feedback_coeff;

        s.scan_id    .push_back(i);
        s.rdom       .push_back(rdom[index]);
        s.inner_rdom .push_back(inner_rdom[index]);
        s.outer_rdom .push_back(outer_rdom[index]);
        s.tail_rdom  .push_back(tail_rdom[index]);
        s.scan_causal.push_back(check_causal_scan(F, rdom[index], i, s.filter_dim));
        s.num_splits++;

        split_info[ dimension[index] ] = s;
    }

    // group scans in same dimension together
    // change the order of splits accordingly
    split_info = group_scans_by_dimension(F, split_info);

    // remove split_info structs for dimensions which are not split
    for (int i=0; i<split_info.size(); i++) {
        if (split_info[i].num_splits == 0) {
            split_info.erase(split_info.begin()+i);
            i--;
        }
    }

    // compute the intra tile result
    Function F_intra = create_intra_tile_term(F, split_info);

    // apply clamped boundary on border tiles and zero boundary on
    // inner tiles if explicit border clamping is requested
    if (clamp_borders) {
        for (int j=0; j<split_info.size(); j++) {
            split_info[j].clamp_border = true;
        }
        apply_zero_boundary_on_inner_tiles(F_intra, split_info);
    }

    // create a function will hold the final result,
    // just a copy of the intra tile computation for now
    Function F_final = create_copy(F_intra, F.name() + DELIMITER + FINAL_TERM);

    // compute the residuals from splits in each dimension
    vector< vector<Function> > F_deps = split_scans(F_intra, split_info);

    // transfer the tail of each scan to another buffer
    extract_tails_from_each_scan(F_intra, split_info);

    // add all the residuals to the final term
    add_all_residuals_to_final_result(F_final, F_deps, split_info);

    // change the original function to index into the final term computed here
    {
        vector<string> args = F.args();
        vector<Expr> values;
        vector<Expr> call_args;
        for (int i=0; i<F_final.args().size(); i++) {
            string arg = F_final.args()[i];
            call_args.push_back(Var(arg));
            for (int j=0; j<var.size(); j++) {
                if (arg == inner_var[j].name()) {
                    call_args[i] = substitute(arg, var[j]%tile_width[j], call_args[i]);
                } else if (arg == outer_var[j].name()) {
                    call_args[i] = substitute(arg, var[j]/tile_width[j], call_args[i]);
                }
            }
        }
        for (int i=0; i<F.outputs(); i++) {
            Expr val = Call::make(F_final, call_args, i);
            values.push_back(val);
        }
        F.clear_all_definitions();
        F.define(args, values);
    }

    func = Func(F);

    inline_function(func, F_final.name());
}
