#include "recfilter.h"
#include "recfilter_internals.h"
#include "modifiers.h"
#include "iir_coeff.h"

using namespace Halide;
using namespace Halide::Internal;

using std::string;
using std::cerr;
using std::endl;
using std::vector;
using std::map;

// -----------------------------------------------------------------------------

template<typename T>
vector<T> extract_row(Image<T> img, int i) {
    vector<T> res;
    for (int j=0; j<img.height(); j++) {
        res.push_back(img(i,j));
    }
    return res;
}

// -----------------------------------------------------------------------------

vector<RecFilter> RecFilter::cascade(vector<vector<int> > scans) {
    if (contents.ptr->tiled || contents.ptr->compiled || contents.ptr->finalized) {
        cerr << "Cascading directive cascade() cannot be used after "
            << "the filter is already tiled, compiled or realized" << endl;
        assert(false);
    }

    // check that the order does not violate
    {
        map<int, bool> scan_causal;
        map<int, bool> scan_dimension;
        map<int, int>  scan_occurance;
        vector<int> reordered_scans;
        for (int i=0; i<scans.size(); i++) {
            for (int j=0; j<scans[i].size(); j++) {
                reordered_scans.push_back(scans[i][j]);
            }
        }
        for (int i=0; i<contents.ptr->filter_info.size(); i++) {
            int scan_dim = contents.ptr->filter_info[i].filter_dim;
            for (int j=0; j<contents.ptr->filter_info[i].num_scans; j++) {
                int scan_id = contents.ptr->filter_info[i].scan_id[j];
                bool causal = contents.ptr->filter_info[i].scan_causal[j];
                scan_causal[scan_id] = causal;
                scan_dimension[scan_id] = scan_dim;
            }
        }
        // order is violated only if the relative order of two scans in
        // same dimension and opposite causality changes
        for (int u=0; u<reordered_scans.size(); u++) {
            int scan_a = reordered_scans[u];
            for (int v=u+1; v<reordered_scans.size(); v++) {
                int scan_b = reordered_scans[v];
                if (scan_dimension.find(scan_a) == scan_dimension.end()) {
                    cerr << "Scan " << scan_a << " not found in recursive filter " << endl;
                    assert(false);
                }
                if (scan_dimension.find(scan_b) == scan_dimension.end()) {
                    cerr << "Scan " << scan_b << " not found in recursive filter " << endl;
                    assert(false);
                }
                int dim_a  = scan_dimension[scan_a];
                int dim_b  = scan_dimension[scan_b];
                bool causal_a = scan_causal[scan_a];
                bool causal_b = scan_causal[scan_b];
                if (dim_a==dim_b && causal_a!=causal_b && scan_b<scan_a) {
                    cerr << "Scans " << scan_a << " " << scan_b << " cannot be reordered"
                        << " during cascading because they have opposite causality" << endl;
                    assert(false);
                }
            }

            // scan_a occurs once more in the list of args
            scan_occurance[scan_a] += 1;
        }

        // check that each scan has occured exactly once
        map<int,int>::iterator so = scan_occurance.begin();
        for (; so!=scan_occurance.end(); so++) {
            if (so->second == 0) {
                cerr << "Scan " << so->first << " does not appear in the list "
                    << "of scans for cascading" << endl;
                assert(false);
            }
            if (so->second > 1) {
                cerr << "Scan " << so->first << " appears multiple times in the list "
                    << "of scans for cascading" << endl;
                assert(false);
            }
        }
    }

    // create the cascaded recursive filters
    vector<RecFilterDim> args;
    for (int i=0; i<contents.ptr->filter_info.size(); i++) {
        args.push_back(RecFilterDim(
                    contents.ptr->filter_info[i].var.name(),
                    contents.ptr->filter_info[i].image_width));
    }

    vector<RecFilter> recfilters;

    for (int i=0; i<scans.size(); i++) {
        RecFilter rf(contents.ptr->name + "_" + int_to_string(i));

        // set the image border conditions
        if (contents.ptr->clamped_border) {
            rf.set_clamped_image_border();
        }

        // same pure def as original filter for the first
        // subsequent filters call the result of prev recfilter
        if (i == 0) {
            rf(args) = as_func().values();
        } else {
            vector<Expr> call_args;
            vector<Expr> pure_values;
            Function f_prev = recfilters[i-1].as_func().function();
            for (int j=0; j<args.size(); j++) {
                call_args.push_back(args[j]);
            }
            for (int j=0; j<f_prev.outputs(); j++) {
                pure_values.push_back(Call::make(f_prev, call_args, j));
            }
            rf(args) = pure_values;
        }

        // extract the scans from the filter and
        // add them to the new filter
        for (int j=0; j<scans[i].size(); j++) {
            int scan_id = scans[i][j];

            // find the split info struct that corresponds
            // to this scan
            int dim = -1;
            int idx = -1;
            for (int u=0; dim<0 && u<contents.ptr->filter_info.size(); u++) {
                for (int v=0; idx<0 && v<contents.ptr->filter_info[u].num_scans; v++) {
                    if (scan_id == contents.ptr->filter_info[u].scan_id[v]) {
                        dim = u;
                        idx = v;
                    }
                }
            }

            if (dim<0 || idx<0) {
                cerr << "Scan " << scan_id << " not found in recursive filter " << endl;
                assert(false);
            }

            RecFilterDim x(contents.ptr->filter_info[dim].var.name(),
                    contents.ptr->filter_info[dim].image_width);
            bool causal = contents.ptr->filter_info[dim].scan_causal[idx];
            int order   = contents.ptr->filter_info[dim].filter_order;

            vector<float> coeff;
            coeff.push_back(contents.ptr->feedfwd_coeff(scan_id));
            for (int u=0; u<order; u++) {
                coeff.push_back(contents.ptr->feedback_coeff(scan_id,u));
            }

            rf.add_filter((causal ? +x : -x), coeff);
        }

        recfilters.push_back(rf);
    }

    return recfilters;
}

vector<RecFilter> RecFilter::cascade(vector<int> a, vector<int> b) {
    if (contents.ptr->tiled || contents.ptr->compiled || contents.ptr->finalized) {
        cerr << "Cascading directive cascade() cannot be used after "
             << "the filter is already tiled, compiled or realized" << endl;
        assert(false);
    }

    return cascade({a,b});
}

vector<RecFilter> RecFilter::cascade_by_causality(void) {
    if (contents.ptr->tiled || contents.ptr->compiled || contents.ptr->finalized) {
        cerr << "Cascading directive cascade_by_causality() cannot be used after "
             << "the filter is already tiled, compiled or realized" << endl;
        assert(false);
    }

    vector<int> causal_scans;
    vector<int> anticausal_scans;
    for (int i=0; i<contents.ptr->filter_info.size(); i++) {
        for (int j=contents.ptr->filter_info[i].num_scans-1; j>=0; j--) {
            int id = contents.ptr->filter_info[i].scan_id[j];
            bool c = contents.ptr->filter_info[i].scan_causal[j];
            if (c) {
                causal_scans.push_back(id);
            } else {
                anticausal_scans.push_back(id);
            }
        }
    }
    return cascade({causal_scans, anticausal_scans});
}

vector<RecFilter> RecFilter::cascade_by_dimension(void) {
    if (contents.ptr->tiled || contents.ptr->compiled || contents.ptr->finalized) {
        cerr << "Cascading directive cascade() cannot be used after "
             << "the filter is already tiled, compiled or realized" << endl;
        assert(false);
    }

    vector< vector<int> > scans;
    for (int i=0; i<contents.ptr->filter_info.size(); i++) {
        vector<int> dim_scans;
        for (int j=contents.ptr->filter_info[i].num_scans-1; j>=0; j--) {
            dim_scans.push_back(contents.ptr->filter_info[i].scan_id[j]);
        }
        if (!dim_scans.empty()) {
            scans.push_back(dim_scans);
        }
    }
    return cascade(scans);
}

RecFilter RecFilter::overlap_to_higher_order_filter(RecFilter fB, string overlap_name) {
    if (contents.ptr->tiled || contents.ptr->compiled || contents.ptr->finalized) {
        cerr << "Overlapping directive overlap() cannot be used after "
             << "the filter is already tiled, compiled or realized" << endl;
        assert(false);
    }

    Function A = as_func().function();
    Function B = fB.as_func().function();

    // extract the input of B
    vector<Expr> input_B = B.values();

    // create a call to the result of A
    vector<Expr> result_A;
    {
        vector<Expr> call_args;
        for (int i=0; i<A.args().size(); i++) {
            call_args.push_back(Var(A.args()[i]));
        }
        for (int i=0; i<A.outputs(); i++) {
            result_A.push_back(Call::make(A, call_args, i));
        }
    }

    // check that result of A if same as input of B
    if (input_B.size() == result_A.size()) {
        for (int i=0; i<result_A.size(); i++) {
            if (!equal(input_B[i], result_A[i])) {
                cerr << "Filters cannot be overlapped because the input to second "
                    << "does not match the output of the first" << endl;
                assert(false);
            }
        }
    } else {
        cerr << "Filters cannot be overlapped because the number of inputs to second "
            << "does not match the number of outputs of the first" << endl;
        assert(false);
    }

    // check that each scans of A matches the corresponding scan of B
    vector<RecFilterDim> filter_dim;
    for (int i=0; i<contents.ptr->filter_info.size(); i++) {
        string var= contents.ptr->filter_info[i].var.name();
        int width = contents.ptr->filter_info[i].image_width;
        filter_dim.push_back(RecFilterDim(var, width));
    }

    // check that both filters have same border clamping
    bool border_clamp_a = contents.ptr->clamped_border;
    bool border_clamp_b = fB.contents.ptr->clamped_border;
    if (border_clamp_a != border_clamp_b) {
        cerr << "Filters cannot be overlapped because one clamps image border while the other does not" << endl;
        assert(false);
    }

    // check that both filters have same type
    if (contents.ptr->type != fB.contents.ptr->type) {
        cerr << "Filters cannot be overlapped because they have different types" << endl;
        assert(false);
    }

    // define the overlapped filter with the input of A
    RecFilter AB(overlap_name);
    AB.define(filter_dim, A.values());

    // clamp borders if needed
    if (border_clamp_a) {
        AB.set_clamped_image_border();
    }

    // feedback and feedforward coeff of both scans of both filters
    Image<float> feedfwd_a = contents.ptr->feedfwd_coeff;
    Image<float> feedfwd_b = fB.contents.ptr->feedfwd_coeff;
    Image<float> feedback_a= contents.ptr->feedback_coeff;
    Image<float> feedback_b= fB.contents.ptr->feedback_coeff;

    // add the scans
    for (int i=0; i<contents.ptr->filter_info.size(); i++) {
        int  filter_dim_a   = contents.ptr->filter_info[i].filter_dim;
        int  num_scans_a    = contents.ptr->filter_info[i].num_scans;
        int  image_width_a  = contents.ptr->filter_info[i].image_width;
        int  tile_width_a   = contents.ptr->filter_info[i].tile_width;
        Var  var_a          = contents.ptr->filter_info[i].var;
        RDom rdom_a         = contents.ptr->filter_info[i].rdom;

        int  filter_dim_b   = fB.contents.ptr->filter_info[i].filter_dim;
        int  num_scans_b    = fB.contents.ptr->filter_info[i].num_scans;
        int  image_width_b  = fB.contents.ptr->filter_info[i].image_width;
        int  tile_width_b   = fB.contents.ptr->filter_info[i].tile_width;
        Var  var_b          = fB.contents.ptr->filter_info[i].var;
        RDom rdom_b         = fB.contents.ptr->filter_info[i].rdom;

        // check each dimension is identical
        if (num_scans_a != num_scans_b) {
            cerr << "Filters cannot be overlapped because they have different num scans in dimension " << i << endl;
            assert(false);
        }
        if (image_width_a != image_width_b ) {
            cerr << "Filters cannot be overlapped because they have different image width in dimension " << i << endl;
            assert(false);
        }
        if (tile_width_a != tile_width_b) {
            cerr << "Filters cannot be overlapped because they have different tile width in dimension " << i << endl;
            assert(false);
        }
        if (!equal(rdom_a.x.min(),rdom_b.x.min()) && equal(rdom_a.x.extent(),rdom_b.x.extent())) {
            cerr << "Filters cannot be overlapped because they have different scan domain in dimension " << i << endl;
            assert(false);
        }
        // no need to ensure that Var are same because only the image width matters

        RecFilterDim x(var_a.name(), image_width_a);

        for (int j=0; j<num_scans_a; j++) {
            bool scan_causal_a = contents.ptr->filter_info[i].scan_causal[j];
            int  scan_id_a     = contents.ptr->filter_info[i].scan_id[j];
            bool scan_causal_b = fB.contents.ptr->filter_info[i].scan_causal[j];
            int  scan_id_b     = fB.contents.ptr->filter_info[i].scan_id[j];

            // check each scan is identical
            if (scan_causal_a != scan_causal_b) {
                cerr << "Filters cannot be overlapped because they have different causality in scan "
                    << j << " of dimension " << i << endl;
                assert(false);
            }
            if (scan_id_a != scan_id_b) {
                cerr << "Filters cannot be overlapped because they have different scan indices in scan "
                    << j << " of dimension " << i << endl;
                assert(false);
            }

            // feedforward coeff
            float ff = feedfwd_a(scan_id_a) * feedfwd_b(scan_id_b);

            // feedback coeff of the two scans
            vector<float> fb_a = extract_row<float>(feedback_a, scan_id_a);
            vector<float> fb_b = extract_row<float>(feedback_b, scan_id_b);
            vector<float> coeff = overlap_feedback_coeff(fb_a, fb_b);

            // add the feedfowrd coeff to the list of coeff
            coeff.insert(coeff.begin(), ff);
            if (scan_causal_a) {
                AB.add_filter(x, coeff);
            } else {
                AB.add_filter(-x, coeff);
            }
        }
    }
    return AB;
}
