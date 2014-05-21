#include "split.h"

std::ostream &operator<<(std::ostream &s, Halide::Func f) {
    s << f.function();
    return s;
}

std::ostream &operator<<(std::ostream &s, Halide::Internal::Function f) {
    if (f.has_pure_definition()) {
        std::vector<Halide::Expr> pure_value = f.values();
        s << "Func " << f.name() << ";\n";
        for (size_t v=0; v<pure_value.size(); v++) {
            std::vector<std::string> args = f.args();
            s << f.name() << "(";
            for (size_t i=0; i<args.size(); i++) {
                s << args[i];
                if (i<args.size()-1)
                    s << ",";
            }
            if (pure_value.size()>1)
                s << ")[" << v << "]";
            else
                s << ")";
            s << " = " << pure_value[v] << "\n";
        }

        // reduction definitions
        for (size_t j=0; j<f.reductions().size(); j++) {
            std::vector<Halide::Expr> reduction_value = f.reductions()[j].values;
            for (size_t v=0; v<reduction_value.size(); v++) {
                std::vector<Halide::Expr> args = f.reductions()[j].args;
                s << f.name() << "(";
                for (size_t i=0; i<args.size(); i++) {
                    s << args[i];
                    if (i<args.size()-1)
                        s << ",";
                }
                if (reduction_value.size()>1)
                    s << ")[" << v << "]";
                else
                    s << ")";
                s << " = " << reduction_value[v];
                if (f.reductions()[j].domain.defined()) {
                    s << " with  ";
                    for (size_t k=0; k<f.reductions()[j].domain.domain().size(); k++) {
                        string r = f.reductions()[j].domain.domain()[k].var;
                        s << r << "("
                            << f.reductions()[j].domain.domain()[k].min   << ","
                            << f.reductions()[j].domain.domain()[k].extent<< ") ";
                    }
                }
                s << "\n";
            }
        }
    }
    return s;
}
