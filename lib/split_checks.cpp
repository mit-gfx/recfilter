#include "split.h"
#include "split_utils.h"

using namespace Halide;
using namespace Halide::Internal;

using std::string;
using std::cerr;
using std::endl;
using std::vector;

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
