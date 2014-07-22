#include "split_macros.h"
#include "split_utils.h"

#include <algorithm>

using namespace Halide;
using namespace Halide::Internal;

using std::string;
using std::cerr;
using std::endl;
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

// -----------------------------------------------------------------------------

bool check_causal_scan(Function f, RVar rx, int scan_id, int dimension) {
    assert(scan_id < f.reductions().size());

    ReductionDefinition reduction = f.reductions()[scan_id];
    Expr               arg       = reduction.args[dimension];

    // check if reduction arg increases on increasing the RVar
    // causal scan if yes, else anticausal
    Expr a = substitute(rx.name(), 0, arg);
    Expr b = substitute(rx.name(), 1, arg);
    Expr c = simplify(a<b);

    if (equal(c, make_bool(true))) {
        return true;
    } else if (equal(c, make_bool(false))) {
        return false;
    } else {
        cerr << "Could not deduce causal or anticausal scan for reduction definition "
            << scan_id << " of " << f.name() << endl;
        assert(false);
    }
}

// -----------------------------------------------------------------------------

void check_split_feasible(
        Func         func,
        vector<int>  dimension,
        vector<Var>  var,
        vector<Var>  inner_var,
        vector<Var>  outer_var,
        vector<RDom> rdom,
        vector<RDom> inner_rdom,
        vector<int>  order)
{

    if (!func.is_reduction()) {
        cerr << "Use Halide::Func::split() to split pure Func "  << func.name() << endl;
        assert(false);
    }

    int num_splits = var.size();

    assert(num_splits == dimension.size()  && "Each split must have a mapped function dimension");
    assert(num_splits == rdom.size()       && "Each split must have a mapped RDom");
    assert(num_splits == inner_var.size()  && "Each split must have a mapped inner Var");
    assert(num_splits == outer_var.size()  && "Each split must have a mapped outer Var");
    assert(num_splits == inner_rdom.size() && "Each split must have a mapped inner RDom");
    assert(num_splits == order.size()      && "Each split must have a mapped filter order");

    Function F = func.function();

    assert(F.has_pure_definition() &&  "Func to be split must be defined");
    assert(!F.is_pure() && "Use Halide::Func::split for pure Funcs");


    // check variables
    for (int k=0; k<num_splits; k++) {
        int dim = dimension[k];

        // repeated scans in the same dimension must have the filter order and tile width
        for (int i=0; i<k-1; i++) {
            if (dimension[k]==dimension[i] && order[i]!=order[k]) {
                cerr << "Different filter orders specified for two scans "
                    << "in the same dimension" << endl;
                assert(false);
            }
        }

        // RDom to be split must be 1D, each reduction definition should be 1D reduction
        if (rdom[k].dimensions() != 1) {
            cerr << "RDom to split must be 1D, each reduction "
                << "definition should use a unique be 1D RDom";
            assert(false);
        }

        // given inner RDom must be 1D, intra tile scans are 1D as full scan is 1D
        if (inner_rdom[k].dimensions() != 1) {
            cerr << "Inner RDom must be 1D, as splitting a 1D reduction"
                << "definition produces 1D intra-tile reductions";
            assert(false);
        }

        // variable at given dimension must match the one to be split
        if (F.args()[dim] != var[k].name()) {
            cerr << "Variable at dimension " << dim << " must match the one "
                << "specified for splitting"   << endl;
            assert(false);
        }

        // RDom to be split must not appear at any dimension other than the one specified
        for (int i=0; i<F.reductions().size(); i++) {
            string rdom_name = rdom[k].x.name();
            bool reduction_involves_rdom = false;
            for (int j=0; j<F.reductions()[i].args.size(); j++) {
                bool arg_contains_rdom = expr_depends_on_var(F.reductions()[i].args[j], rdom_name);
                if (j!=dim && arg_contains_rdom) {
                    cerr << "RDom " << rdom_name  << " to be split must appear only at the "
                         << "specified dimension " << dim << ", found in others" << endl;
                    assert(false);
                }
            }
        }

        // RDom to be split must appear in exactly one reduction definition
        int num_reductions_involving_rdom = 0;
        for (int i=0; i<F.reductions().size(); i++) {
            string rdom_name = rdom[k].x.name();
            bool reduction_involves_rdom = false;
            for (int j=0; j<F.reductions()[i].values.size(); j++) {
                reduction_involves_rdom |= expr_depends_on_var(F.reductions()[i].values[j], rdom_name);
            }
            if (reduction_involves_rdom) {
                if (!expr_depends_on_var(F.reductions()[i].args[dim], rdom_name)) {
                    cerr << "RDom " << rdom_name  << " to be split does not appear at the "
                        << "specified dimension " << dim << endl;
                    assert(false);
                }
                num_reductions_involving_rdom++;
            }
        }
        if (num_reductions_involving_rdom < 1) {
            cerr << "RDom to be split must appear in one reduction definition, found in none";
            assert(false);
        }
        if (num_reductions_involving_rdom > 1) {
            cerr << "RDom to be split must appear in only one reduction definition, found in multiple";
            assert(false);
        }
    }
}

// -----------------------------------------------------------------------------

// Matrices for computing feedback and feedforward coeff

static Image<float> matrix_B(
        Image<float> feedfwd_coeff,
        Image<float> feedback_coeff,
        int scan_id,
        int tile_width,
        bool clamp_border)
{
    int filter_order = feedback_coeff.height();

    float feedfwd = feedfwd_coeff(scan_id);

    vector<float> feedback(filter_order);
    for (int i=0; i<filter_order; i++) {
        feedback[i] = feedback_coeff(scan_id, i);
    }

    Image<float> C(tile_width, tile_width);

    // initialize
    for (int x=0; x<tile_width; x++) {
        for (int y=0; y<tile_width; y++) {
            C(x,y) = (x==y ? feedfwd : 0.0f);
        }
    }

    // update one row at a time from bottom to up
    for (int y=0; y<tile_width; y++) {
        for (int x=0; x<tile_width; x++) {
            for (int j=0; j<filter_order; j++) {
                float a = 0.0f;
                if (clamp_border) {
                    a = (y-j-1>=0 ? C(x,y-j-1)*feedback[j] : (x==0 ? feedback[j] : 0.0f));
                } else {
                    a = (y-j-1>=0 ? C(x,y-j-1)*feedback[j] : 0.0f);
                }
                C(x,y) += a;
            }
        }
    }

    return C;
}

static Image<float> matrix_R(
        Image<float> feedback_coeff,
        int scan_id,
        int tile_width)
{
    int filter_order = feedback_coeff.height();

    vector<float> weights(filter_order);
    for (int i=0; i<filter_order; i++) {
        weights[i] = feedback_coeff(scan_id, i);
    }

    Image<float> C(filter_order, tile_width);

    for (int x=0; x<filter_order; x++) {
        for (int y=0; y<tile_width; y++) {
            C(x,y) = 0.0f;
        }
    }

    for (int y=0; y<tile_width; y++) {
        for (int x=0; x<filter_order; x++) {
            if (y<filter_order) {
                C(x,y) = (x+y<filter_order ? weights[x+y] : 0.0f);
            }
            for (int j=0; y-j-1>=0 && j<filter_order; j++) {
                C(x,y) += C(x,y-j-1) * weights[j];
            }
        }
    }

    return C;
}

static Image<float> matrix_transpose(Image<float> A) {
    Image<float> B(A.height(),A.width());
    for (int y=0; y<B.height(); y++) {
        for (int x=0; x<B.width(); x++) {
            B(x,y) = A(y,x);
        }
    }
    return B;
}

static Image<float> matrix_mult(Image<float> A, Image<float> B) {
    assert(A.width() == B.height());

    int num_rows = A.height();
    int num_cols = B.width();
    int num_common = A.width();

    Image<float> C(num_cols, num_rows);

    for (int i=0; i<C.width(); i++) {
        for (int j=0; j<C.height(); j++) {
            C(i,j) = 0.0f;
        }
    }
    for (int i=0; i<C.height(); i++) {
        for (int j=0; j<C.width(); j++) {
            for (int k=0; k<num_common; k++) {
                C(j,i) += A(k,i) * B(j,k);
            }
        }
    }
    return C;
}

static Image<float> matrix_antidiagonal(int size) {
    Image<float> C(size, size);

    for (int i=0; i<C.width(); i++) {
        for (int j=0; j<C.height(); j++) {
            C(i,j) = (i==size-1-j ? 1.0f : 0.0f);
        }
    }
    return C;
}


/** Weight coefficients (tail_size x tile_width) for
 * applying scan's corresponding to split indices split_id1 to
 * split_id2 in the SplitInfo struct.
 * It is meaningful to apply subsequent scans on the tail of any scan
 * as it undergoes other scans only if they happen after the first
 * scan. The SpliInfo object stores the scans in reverse order, hence indices
 * into the SplitInfo object split_id1 and split_id2 must be decreasing
 */
Image<float> tail_weights(SplitInfo s, int split_id1, int split_id2, bool clamp_border) {
    assert(split_id1 >= split_id2);

    const int* tile_width_ptr = as_const_int(s.tile_width);
    assert(tile_width_ptr &&
            "Could not convert tile width expression to integer");

    int  tile_width  = *tile_width_ptr;
    int  scan_id     = s.scan_id[split_id1];
    bool scan_causal = s.scan_causal[split_id1];

    Image<float> R = matrix_R(s.feedback_coeff, scan_id, tile_width);

    // accummulate weight coefficients because of all subsequent scans
    // traversal is backwards because SplitInfo contains scans in the
    // reverse order
    for (int j=split_id1-1; j>=split_id2; j--) {
        if (scan_causal != s.scan_causal[j]) {
            Image<float> B = matrix_B(s.feedfwd_coeff,
                    s.feedback_coeff, s.scan_id[j], tile_width, clamp_border);
            Image<float> I = matrix_antidiagonal(R.height());
            R = matrix_mult(I, R);
            R = matrix_mult(B, R);
            R = matrix_mult(I, R);
        }
        else {
            Image<float> B = matrix_B(s.feedfwd_coeff,
                    s.feedback_coeff, s.scan_id[j], tile_width, false);
            R = matrix_mult(B, R);
        }
    }

    return matrix_transpose(R);
}

/** Weight coefficients (tail_size x tile_width) for
 * applying scan's corresponding to split indices split_id1
 */
Image<float> tail_weights(SplitInfo s, int split_id1, bool clamp_border) {
    return tail_weights(s, split_id1, split_id1, clamp_border);
}
