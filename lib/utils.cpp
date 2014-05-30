#include "split.h"

std::ostream &operator<<(std::ostream &s, CheckResult v) {
    Halide::Image<float> ref = v.ref;
    Halide::Image<float> out = v.out;

    assert(ref.width() == out.width());
    assert(ref.height() == out.height());

    int width = ref.width();
    int height = ref.height();

    Halide::Image<float> diff(width, height);

    float diff_sum = 0.0f;
    float max_val  = 0.0f;

    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            diff(x,y) = float(ref(x,y)) - float(out(x,y));
            diff_sum += std::abs(diff(x,y) * diff(x,y));
            max_val = std::max(ref(x,y), max_val);
        }
    }
    float mse  = diff_sum/float(width*height);

    s << "Mean sq error = " << mse << "\n\n";

    return s;
}

std::ostream &operator<<(std::ostream &s, CheckResultVerbose v) {
    Halide::Image<float> ref = v.ref;
    Halide::Image<float> out = v.out;

    assert(ref.width() == out.width());
    assert(ref.height() == out.height());
    assert(ref.channels() == out.channels());

    int width = ref.width();
    int height = ref.height();

    Halide::Image<float> diff(width, height);

    float diff_sum = 0.0f;
    float max_val  = 0.0f;

    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            diff(x,y) = float(ref(x,y)) - float(out(x,y));
            diff_sum += std::abs(diff(x,y) * diff(x,y));
            max_val = std::max(ref(x,y), max_val);
        }
    }
    float mse  = diff_sum/float(width*height);

    s << "Reference" << "\n" << ref << "\n";
    s << "Halide output" << "\n" << out << "\n";
    s << "Difference " << "\n" << diff << "\n";
    s << "Mean sq error = " << mse << "\n\n";

    return s;
}

std::ostream &operator<<(std::ostream &s, Halide::Func f) {
    s << f.function();
    return s;
}

std::ostream &operator<<(std::ostream &s, Halide::Internal::Function f) {
    if (f.has_pure_definition()) {
        std::vector<Halide::Expr> pure_value = f.values();
        s << "Func " << f.name() << ";\n";
        for (int v=0; v<pure_value.size(); v++) {
            std::vector<std::string> args = f.args();
            s << f.name() << "(";
            for (int i=0; i<args.size(); i++) {
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
        for (int j=0; j<f.reductions().size(); j++) {
            std::vector<Halide::Expr> reduction_value = f.reductions()[j].values;
            for (int v=0; v<reduction_value.size(); v++) {
                std::vector<Halide::Expr> args = f.reductions()[j].args;
                s << f.name() << "(";
                for (int i=0; i<args.size(); i++) {
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
                    for (int k=0; k<f.reductions()[j].domain.domain().size(); k++) {
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



