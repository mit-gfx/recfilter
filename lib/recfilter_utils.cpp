#include "recfilter.h"
#include "recfilter_func.h"
#include "recfilter_internals.h"

using std::string;
using std::vector;
using std::cerr;
using std::endl;
using std::ostream;
using std::stringstream;
using std::runtime_error;

using namespace Halide;
using namespace Halide::Internal;

// -----------------------------------------------------------------------------

/** Remove $r from variable names which is attached by Halide automatically to all RVar names */
static string clean_var_names(string str) {
    int start_pos = 0;
    string replace_str = "$r";
    string null_str = "";
    while ((start_pos = str.find(replace_str, start_pos)) != string::npos) {
        str.replace(start_pos, replace_str.length(), null_str);
    }
    return str;
}

// -----------------------------------------------------------------------------

Arguments::Arguments(int argc, char** argv) :
    width  (4096),
    block  (32),
    iterations(1),
    threads(192),
    nocheck(false)
{
    string app_name = argv[0];
    string desc = "\nUsage\n " + app_name + " ";
    desc.append(string(
                "[-width|-w w] [-tile|-block|-b|-t b] [-thread n] [-iter i] [-nocheck] [-help]\n\n"
                "\twidth    width of input image [default = 4096]\n"
                "\ttile     tile width for splitting each dimension image [default = 32]\n"
                "\tthread   maximum threads per tile [default = 192]\n"
                "\tnocheck  do not check against reference solution [default = false]\n"
                "\titer     number of profiling iterations [default = 1]\n"
                "\thelp     show help message\n"
                )
            );

    try {
        for (int i=1; i<argc; i++) {
            string option = argv[i];

            if (!option.compare("-help") || !option.compare("--help")) {
                throw runtime_error("Showing help message");
            }

            else if (!option.compare("-nocheck") || !option.compare("--nocheck")) {
                nocheck = true;
            }

            else if (!option.compare("-iter") || !option.compare("--iter")) {
                if ((i+1) < argc)
                    iterations = atoi(argv[++i]);
                else
                    throw runtime_error("-iter requires a int value");
            }

            else if (!option.compare("-thread") || !option.compare("--thread")) {
                if ((i+1) < argc)
                    threads = atoi(argv[++i]);
                else
                    throw runtime_error("-thread requires a int value");
            }

            else if (!option.compare("-w") || !option.compare("--w") || !option.compare("-width") || !option.compare("--width")) {
                if ((i+1) < argc) {
                    width = atoi(argv[++i]);
                }
                else
                    throw runtime_error("-width requires an integer value");
            }

            else if (!option.compare("-b") || !option.compare("--b") || !option.compare("-block") || !option.compare("--block") ||
                    !option.compare("-t") || !option.compare("--t") || !option.compare("-tile") || !option.compare("--tile"))
            {
                if ((i+1) < argc)
                    block = atoi(argv[++i]);
                else
                    throw runtime_error("-block requires an integer value");
            }

            else {
                throw runtime_error("Bad command line option " + option);
            }
        }

        if (width%block)
            throw runtime_error("Width should be a multiple of block size");

    } catch (runtime_error & e) {
        cerr << endl << e.what() << endl << desc << endl;
        exit(EXIT_FAILURE);
    }
}

// -----------------------------------------------------------------------------

ostream &operator<<(ostream &s, const CheckResult& v) {
    Image<float> ref = v.ref;
    Image<float> out = v.out;

    assert(ref.width() == out.width());
    assert(ref.height() == out.height());
    assert(ref.channels() == out.channels());

    int width = ref.width();
    int height = ref.height();
    int channels = ref.channels();

    Image<double> diff(width, height, channels);

    double re      = 0.0;
    double max_re  = 0.0;
    double mean_re = 0.0;

    for (int z=0; z<channels; z++) {
        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {
                diff(x,y,z) = ref(x,y,z) - out(x,y,z);
                re       = std::abs(diff(x,y,z)) / ref(x,y,z);
                mean_re += re;
                max_re   = std::max(re, max_re);
            }
        }
    }
    mean_re /= double(width*height*channels);

    s << "Max  relative error = " << 100.0*max_re << " % \n";
    s << "Mean relative error = " << 100.0*mean_re << " % \n\n";

    return s;
}

ostream &operator<<(ostream &s, const CheckResultVerbose &v) {
    Image<float> ref = v.ref;
    Image<float> out = v.out;

    assert(ref.width() == out.width());
    assert(ref.height() == out.height());
    assert(ref.channels() == out.channels());

    int width = ref.width();
    int height = ref.height();
    int channels = ref.channels();

    Image<double> diff(width, height, channels);

    double re      = 0.0;
    double max_re  = 0.0;
    double mean_re = 0.0;

    for (int z=0; z<channels; z++) {
        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {
                diff(x,y,z) = (ref(x,y,z) - out(x,y,z));
                re       = std::abs(diff(x,y,z)) / ref(x,y,z);
                mean_re += re;
                max_re   = std::max(re, max_re);
            }
        }
    }
    mean_re /= double(width*height*channels);

    s << "Reference" << "\n" << ref << "\n";
    s << "Halide output" << "\n" << out << "\n";
    s << "Difference " << "\n" << diff << "\n";
    s << "Max  relative error = " << 100.0*max_re << " % \n";
    s << "Mean relative error = " << 100.0*mean_re << " % \n\n";

    return s;
}

static ostream &operator<<(ostream &s, const RecFilterFunc::FuncCategory &f) {
    if (f ==RecFilterFunc::INLINE           ) { s << "INLINE "           ; }
    if (f & RecFilterFunc::FULL_RESULT_SCAN ) { s << "FULL_RESULT_SCAN " ; }
    if (f & RecFilterFunc::FULL_RESULT_PURE ) { s << "FULL_RESULT_PURE " ; }
    if (f & RecFilterFunc::INTRA_TILE_SCAN  ) { s << "INTRA_TILE_SCAN "  ; }
    if (f & RecFilterFunc::INTER_TILE_SCAN  ) { s << "INTER_TILE_SCAN "  ; }
    if (f & RecFilterFunc::REINDEX_FOR_WRITE) { s << "REINDEX_FOR_WRITE "; }
    if (f & RecFilterFunc::REINDEX_FOR_READ ) { s << "REINDEX_FOR_READ " ; }
    return s;
}

static ostream &operator<<(ostream &s, const RecFilterFunc::VarCategory &v) {
    if (v & RecFilterFunc::INNER_PURE_VAR) { s << "INNER_PURE_VAR "; }
    if (v & RecFilterFunc::INNER_SCAN_VAR) { s << "INNER_SCAN_VAR "; }
    if (v & RecFilterFunc::OUTER_PURE_VAR) { s << "OUTER_PURE_VAR "; }
    if (v & RecFilterFunc::OUTER_SCAN_VAR) { s << "OUTER_SCAN_VAR "; }
    if (v & RecFilterFunc::TAIL_DIMENSION) { s << "TAIL_DIMENSION "; }
    if (v & RecFilterFunc::PURE_DIMENSION) { s << "PURE_DIMENSION "; }
    if (v & RecFilterFunc::SCAN_DIMENSION) { s << "SCAN_DIMENSION "; }
    return s;
}

ostream &operator<<(ostream &s, const Func &f) {
    s << f.function();
    return s;
}

ostream &operator<<(ostream &os, const Internal::Function &f) {
    stringstream s;

    if (f.has_pure_definition()) {
        s << "{\n";
        s << "Func " << f.name() << "(\"" << f.name() << "\");\n";

        // print the vars
        if (!f.args().empty()) {
            s << "Var ";
            for (int i=0; i<f.args().size(); i++) {
                s << f.args()[i] << "(\"" << f.args()[i] << "\")";
                if (i<f.args().size()-1) {
                    s << ", ";
                }
            }
            s << ";\n";
        }

        // collect RDoms from all update defs and print them
        vector<ReductionDomain> rdom;
        for (int j=0; j<f.updates().size(); j++) {
            if (f.updates()[j].domain.defined()) {
                bool already_added = false;
                for (int i=0; i<rdom.size(); i++) {
                    already_added = f.updates()[j].domain.same_as(rdom[i]);
                }
                if (!already_added) {
                    rdom.push_back(f.updates()[j].domain);
                }
            }
        }
        if (!rdom.empty()) {
            for (int i=0; i<rdom.size(); i++) {
                s << "Internal::ReductionDomain ";
                for (int k=0; k<rdom[i].domain().size(); k++) {
                    string r = rdom[i].domain()[k].var;
                    s << r << "("
                        << rdom[i].domain()[k].min    << ","
                        << rdom[i].domain()[k].extent << ") ";
                }
                s << ";\n";
            }
            s << "\n";
        }

        // print the pure def
        for (int v=0; v<f.values().size(); v++) {
            vector<string> args = f.args();
            s << f.name() << "(";
            for (int i=0; i<args.size(); i++) {
                s << args[i];
                if (i<args.size()-1) {
                    s << ", ";
                }
            }
            s << ") = ";
            if (f.outputs()>1) {
                s << "Tuple(";
                for (int v=0; v<f.values().size(); v++) {
                    s << f.values()[v];
                    if (v<f.values().size()-1) {
                        s << ", ";
                    }
                }
                s << ");\n";
            } else {
                s << f.values()[0] << ";\n";
            }
        }

        // print the update defs
        for (int j=0; j<f.updates().size(); j++) {
            vector<Expr> update_value = f.updates()[j].values;
            vector<Expr> args = f.updates()[j].args;
            s << f.name() << "(";
            for (int i=0; i<args.size(); i++) {
                s << args[i];
                if (i<args.size()-1) {
                    s << ", ";
                }
            }
            s << ") = ";
            if (update_value.size()>1) {
                s << "Tuple(";
                for (int v=0; v<update_value.size(); v++) {
                    s << update_value[v];
                    if (v<update_value.size()-1) {
                        s << ", ";
                    }
                }
                s << ");\n";
            } else {
                s << update_value[0] << ";\n";
            }
        }
        s << "}\n";
    }

    os << clean_var_names(s.str());
    return os;
}

ostream &operator<<(ostream &s, const RecFilter &r) {
    r.generate_hl_code(s);
    return s;
}

ostream &operator<<(std::ostream &os, const RecFilterFunc &f) {
    stringstream s;

    s << "// Func " << f.func.name() << " synopsis\n";
    s << "// Function tag: " << f.func_category;
    if (f.func_category & RecFilterFunc::REINDEX_FOR_WRITE) {
        s << " (calls " << f.callee_func <<  ")";
    }
    if (f.func_category & RecFilterFunc::REINDEX_FOR_READ) {
        s << " (called by " << f.caller_func <<  ")";
    }
    s << "\n";

    if (f.func_category != RecFilterFunc::INLINE) {
        map<string, RecFilterFunc::VarCategory>::const_iterator it;
        s << "// \n";
        s << "// Pure def tags \n";
        for (it=f.pure_var_category.begin(); it!=f.pure_var_category.end(); it++) {
            s << "//\t" << it->first << "\t: " << it->second << "\n";
        }
        if (!f.update_var_category.empty()) {
            s << "// \n";
            for (int i=0; i<f.update_var_category.size(); i++) {
                s << "// Update def " << i << " tags \n";
                for (it=f.update_var_category[i].begin(); it!=f.update_var_category[i].end(); it++) {
                    s << "//\t" << it->first << "\t: " << it->second << "\n";
                }
            }
        }
    }

    os << clean_var_names(s.str());
    return os;
}
