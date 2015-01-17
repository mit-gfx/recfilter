#include "recfilter.h"
#include "recfilter_internals.h"

using std::string;
using std::map;
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
    width    (4096),
    min_width(4096),
    max_width(4096),
    block     (32),
    iterations(1),
    nocheck(false)
{
    string app_name = argv[0];
    string desc = "\nUsage\n " + app_name + " ";
    desc.append(string(
        "[-width|-w w] [-tile|-t b] [-filter n] [-iter i] [-nocheck] [-help]\n\n"
        "\twidth    image width, set 0 to run all image widths and force --nocheck [default = 4096]\n"
        "\ttile     tile width for splitting each dimension image [default = 32]\n"
        "\tfilter   number of repeated filter applications on same input [default = 1]\n"
        "\tnocheck  do not check against reference solution, forced to true if width=0 or iterations>1 [default = false]\n"
        "\titer     number of profiling iterations [default = 1]\n"
        "\thelp     show help message\n"
        ));

    try {
        for (int i=1; i<argc; i++) {
            string option = argv[i];

            if (!option.compare("-help") || !option.compare("--help")) {
                throw runtime_error("Showing help message");
            }

            else if (!option.compare("-nocheck") || !option.compare("--nocheck")) {
                nocheck = true;
            }

            else if (!option.compare("-filter") || !option.compare("--filter")) {
                if ((i+1) < argc)
                    filter_reps = atoi(argv[++i]);
                else
                    throw runtime_error("-filter requires an integer value");
            }

            else if (!option.compare("-iter") || !option.compare("--iter")) {
                if ((i+1) < argc)
                    iterations = atoi(argv[++i]);
                else
                    throw runtime_error("-iter requires an integer value");
            }

            else if (!option.compare("-w") || !option.compare("--w") || !option.compare("-width") || !option.compare("--width")) {
                if ((i+1) < argc) {
                    width = atoi(argv[++i]);
                }
                else
                    throw runtime_error("-width requires an integer value");
            }

            else if (!option.compare("-t") || !option.compare("--t") || !option.compare("-tile") || !option.compare("--tile"))
            {
                if ((i+1) < argc)
                    block = atoi(argv[++i]);
                else
                    throw runtime_error("-tile requires an integer value");
            }

            else {
                throw runtime_error("Bad command line option " + option);
            }
        }

        if (width%block) {
            throw runtime_error("Width should be a multiple of block size");
        }

        if (width) {
            max_width = width;
            min_width = width;
        } else {
            min_width = 2*block;
            max_width = (4096/block)*block;
            nocheck   = true;
        }

        if (iterations>1) {
            nocheck = true;
        }

    } catch (runtime_error & e) {
        cerr << endl << e.what() << endl << desc << endl;
        exit(EXIT_FAILURE);
    }
}

// -----------------------------------------------------------------------------

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
        {
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
    s << r.print_hl_code();
    return s;
}

ostream &operator<<(std::ostream &os, const RecFilterFunc &f) {
    stringstream s;

    s << "// Func " << f.func.name() << " synopsis\n";
    s << "// Function tag: " << f.func_category;
    if (!f.callee_func.empty()) {
        s << " (calls " << f.callee_func <<  ")";
    }
    if (!f.caller_func.empty()) {
        s << " (called by " << f.caller_func <<  ")";
    }
    s << "\n";

    if (f.func_category != INLINE) {
        map<string, VarTag>::const_iterator it;
        s << "// \n";
        s << "// Pure def tags \n";
        for (it=f.pure_var_category.begin(); it!=f.pure_var_category.end(); it++) {
            s << "//\t" << it->first << "\t: " << it->second << "\n";
        }
        if (!f.update_var_category.empty()) {
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

ostream &operator<<(ostream &s, const RecFilterDim &v) {
    s << v.var();
    return s;
}

// ----------------------------------------------------------------------------

RecFilterDimAndCausality operator+(RecFilterDim x) { return RecFilterDimAndCausality(x,true); }
RecFilterDimAndCausality operator-(RecFilterDim x) { return RecFilterDimAndCausality(x,false);}

// -----------------------------------------------------------------------------

VarTag::VarTag(void) : tag(INVALID) {}

VarTag::VarTag(const VarTag &t) : tag(t.tag) {}

VarTag::VarTag(const VariableTag &t) : tag(t) {}

VarTag::VarTag(const VarTag &t, int i) : VarTag(t.tag,i) {}

VarTag::VarTag(const VariableTag &t, int i) {
   if      (i<0)  { tag = t; }
   else if (i==0) { tag = VarTag(t | __1).tag; }
   else if (i==1) { tag = VarTag(t | __2).tag; }
   else if (i==2) { tag = VarTag(t | __3).tag; }
   else if (i==3) { tag = VarTag(t | __4).tag; }
   else {
       std::cerr << "Cannot convert integer to VarTag count" << std::endl;
       assert(false);
    }
}

VarTag::VarTag(int i) : tag(static_cast<VariableTag>(i)) {}

VarTag& VarTag::operator=(const VarTag &t) {
    tag = t.tag;
    return *this;
}

VarTag& VarTag::operator=(const VariableTag &t) {
    tag = t;
    return *this;
}

int VarTag::as_integer(void) const {
    return static_cast<int>(tag);
}

VarTag VarTag::split_var(void) const {
    return VarTag(tag|SPLIT);
}

bool VarTag::same_except_count(const VarTag &t) const {
    int a = as_integer()  & (~(__1 | __2 | __3 | __4));
    int b = t.as_integer()& (~(__1 | __2 | __3 | __4));
    return (a==b);
}

int VarTag::check(const VariableTag &t) const {
    return (as_integer() & VarTag(t).as_integer());
}

bool VarTag::has_count(void) const {
    VariableTag t_count = static_cast<VariableTag>(static_cast<int>(tag) & 0x0000000f);
    return (t_count!=INVALID);
}

int VarTag::count(void) const {
    VariableTag t_count = static_cast<VariableTag>(static_cast<int>(tag) & 0x0000000f);
    int t_int;
    switch (t_count) {
        case __1: t_int = 0; break;
        case __2: t_int = 1; break;
        case __3: t_int = 2; break;
        case __4: t_int = 3; break;
        default: std::cerr << "VarTag does not have a count" << std::endl; assert(false);
    }
    return t_int;
}

// -----------------------------------------------------------------------------

VariableTag operator|(const VariableTag &a, const VariableTag &b) {
    return static_cast<VariableTag>(static_cast<int>(a) | static_cast<int>(b));
}

VariableTag operator&(const VariableTag &a, const VariableTag &b) {
    return static_cast<VariableTag>(static_cast<int>(a) & static_cast<int>(b));
}

VarTag operator|(const VarTag &a, const VarTag &b) { return VarTag(a.as_integer() | b.as_integer()); }
VarTag operator&(const VarTag &a, const VarTag &b) { return VarTag(a.as_integer() & b.as_integer()); }

bool operator==(const VarTag &a, const VarTag &b)      { return (a.as_integer() == b.as_integer()); }
bool operator==(const FuncTag &a,const FuncTag &b)     { return (a.as_integer() == b.as_integer()); }
bool operator!=(const VarTag &a, const VarTag &b)      { return !(a == b); }
bool operator!=(const FuncTag &a,const FuncTag &b)     { return !(a == b); }
bool operator==(const VarTag &a, const VariableTag &b) { return (a == VarTag(b));  }
bool operator==(const FuncTag &a,const FunctionTag &b) { return (a == FuncTag(b)); }

ostream &operator<<(ostream &s, const FunctionTag &f) { s << FuncTag(f); return s; }
ostream &operator<<(ostream &s, const VariableTag &v) { s << VarTag(v);  return s; }

ostream &operator<<(ostream &s, const FuncTag &f) {
    if (f==INLINE ) { s << "INLINE" ; }
    if (f==INTRA_1) { s << "INTRA_1"; }
    if (f==INTRA_N) { s << "INTRA_N"; }
    if (f==INTER  ) { s << "INTER"  ; }
    if (f==REINDEX) { s << "REINDEX"; }
    return s;
}

ostream &operator<<(ostream &s, const VarTag &v) {
    if (v.check(FULL )) { s << "FULL ";  }
    if (v.check(INNER)) { s << "INNER "; }
    if (v.check(OUTER)) { s << "OUTER "; }
    if (v.check(SCAN )) { s << "SCAN ";  }
    if (v.check(TAIL )) { s << "TAIL ";  }
    if (v.check(__1))   { s << "1 ";     }
    if (v.check(__2))   { s << "2 ";     }
    if (v.check(__3))   { s << "3 ";     }
    if (v.check(__4))   { s << "4 ";     }
    if (v.check(SPLIT)) { s << "SPLIT ";}
    return s;
}
