#include "modifiers.h"

using namespace Halide;
using namespace Halide::Internal;

using std::string;
using std::vector;
using std::set;
using std::map;
using std::make_pair;

template<typename T>
void mutate_binary_operator(IRMutator *mutator, const T *op, Expr *expr) {
    Expr a = mutator->mutate(op->a);
    Expr b = mutator->mutate(op->b);
    if (a.same_as(op->a) &&
        b.same_as(op->b)) {
        *expr = op;
    } else {
        *expr = T::make(a, b);
    }
}

// -----------------------------------------------------------------------------

// Does an expr depend on a particular variable?
class ExprDependsOnVar : public IRVisitor {
private:
    using IRVisitor::visit;
    void visit(const Variable *op) {
        if (op->name==var)
            result = true;
    }
    string var;
public:
    bool result;
    ExprDependsOnVar(string v) :
        result(false), var(v) {}
};

// -----------------------------------------------------------------------------

// Does an expr depend on calls to a particular function?
class ExprDependsOnFunc : public IRVisitor {
private:
    using IRVisitor::visit;
    void visit(const Call *op) {
        if (op->call_type==Call::Halide && op->name==func_name)
            result |= true;

        for (size_t i=0; i<op->args.size(); i++) {
            op->args[i].accept(this);
        }

        // Consider extern call args
        Function f = op->func;
        if (op->call_type == Call::Halide && f.has_extern_definition()) {
            for (size_t i=0; i<f.extern_arguments().size(); i++) {
                ExternFuncArgument arg = f.extern_arguments()[i];
                if (arg.is_expr()) {
                    arg.expr.accept(this);
                }
            }
        }
    }

    string func_name;

public:
    bool result;
    ExprDependsOnFunc(string f) :
        result(false), func_name(f) {}
};

// -----------------------------------------------------------------------------

// Extract variables referenced in Expr
class ExtractVarsInExpr : public IRVisitor {
private:
    using IRVisitor::visit;
    void visit(const Variable *op) {
        if (op->reduction_domain.defined()) {
            rvar_list.push_back(op->name);
            var_or_rvar_list.push_back(op->name);
        }
        else if (op->param.defined()) {
            param_list.push_back(op->name);
        }
        else {
            var_list.push_back(op->name);
            var_or_rvar_list.push_back(op->name);
        }
    }
public:
    vector<string> var_list;
    vector<string> rvar_list;
    vector<string> var_or_rvar_list;
    vector<string> param_list;
    ExtractVarsInExpr(void) {}
};

// -----------------------------------------------------------------------------

// Replace all
// - all Func calls to func_name with 0 if matching flag is set
// - all Func calls other than func_name with 0 if matching flag is not set
class RemoveFunctionCall : public IRMutator {
private:
    using IRMutator::visit;
    void visit(const Call *op) {
        vector<Expr> new_args(op->args.size());
        bool changed = false;

        // Mutate the args
        for (size_t i=0; i<op->args.size(); i++) {
            Expr old_arg = op->args[i];
            Expr new_arg = mutate(old_arg);
            if (!new_arg.same_as(old_arg)) changed = true;
            new_args[i] = new_arg;
        }

        if (changed) {
            expr = Call::make(op->type, op->name, new_args, op->call_type,
                    op->func, op->value_index, op->image, op->param);
        } else {
            expr = op;
        }

        if (op->call_type==Call::Halide &&
                (( matching && op->name==func_name) ||
                 (!matching && op->name!=func_name)))
        {
            expr = make_zero(op->func.output_types()[op->value_index]);
        }
    }

    bool   matching;
    string func_name;

public:
    vector<Expr> removed_expr;
    RemoveFunctionCall(string f, bool m) : func_name(f), matching(m) {}
};

// -----------------------------------------------------------------------------

// Add an Expr to Func call
class AugmentFunctionCall : public IRMutator {
private:
    using IRMutator::visit;
    void visit(const Call *op) {
        vector<Expr> new_args(op->args.size());
        bool changed = false;

        // Mutate the args
        for (size_t i=0; i<op->args.size(); i++) {
            Expr old_arg = op->args[i];
            Expr new_arg = mutate(old_arg);
            if (!new_arg.same_as(old_arg)) changed = true;
            new_args[i] = new_arg;
        }

        if (changed) {
            expr = Call::make(op->type, op->name, new_args, op->call_type,
                    op->func, op->value_index, op->image, op->param);
        } else {
            expr = op;
        }

        // add the extra term, use the extra term corresponding to the
        // value_index of the call to func_name and replace all args of
        // func with calling args
        if (op->call_type==Call::Halide && op->name==func_name) {
            Expr e = extra_term[op->value_index];
            for (size_t i=0; i<func_args.size(); i++) {
                e = substitute(func_args[i], new_args[i], e);
            }
            expr += e;
        }
    }

    string func_name;
    vector<string> func_args;
    vector<Expr>   extra_term;

public:
    AugmentFunctionCall(string f, vector<string> a, vector<Expr> e) :
        func_name(f), func_args(a), extra_term(e) {}
};

// -----------------------------------------------------------------------------


// Substitute Func call with another Func call
class SubstituteFunctionCall : public IRMutator {
private:
    using IRMutator::visit;
    void visit(const Call *op) {
        vector<Expr> new_args(op->args.size());
        bool changed = false;

        // Mutate the args
        for (size_t i=0; i<op->args.size(); i++) {
            Expr old_arg = op->args[i];
            Expr new_arg = mutate(old_arg);
            if (!new_arg.same_as(old_arg)) changed = true;
            new_args[i] = new_arg;
        }

        Function replacement_func = op->func;
        if (op->call_type==Call::Halide && op->name==func_name) {
            replacement_func = new_func;
            changed = true;
        }

        if (changed) {
            expr = Call::make(op->type, replacement_func.name(), new_args, op->call_type,
                    replacement_func, op->value_index, op->image, op->param);
        } else {
            expr = op;
        }
    }

    string   func_name;
    Function new_func;

public:
    SubstituteFunctionCall(string f, Function r) :
        func_name(f), new_func(r) {}
};

// -----------------------------------------------------------------------------

// Add argument to Func call
class InsertArgToFunctionCall : public IRMutator {
private:
    using IRMutator::visit;
    void visit(const Call *op) {
        vector<Expr> new_args(op->args.size());
        bool changed = false;

        // Mutate the args
        for (size_t i=0; i<op->args.size(); i++) {
            Expr old_arg = op->args[i];
            Expr new_arg = mutate(old_arg);
            if (!new_arg.same_as(old_arg)) changed = true;
            new_args[i] = new_arg;
        }

        if (op->call_type==Call::Halide && op->name==func_name) {
            new_args.insert(new_args.begin()+new_pos, new_arg);
            changed = true;
        }

        if (changed) {
            expr = Call::make(op->type, op->name, new_args, op->call_type,
                    op->func, op->value_index, op->image, op->param);
        } else {
            expr = op;
        }
    }

    string func_name;
    size_t new_pos;
    Expr   new_arg;

public:
    InsertArgToFunctionCall(string f, size_t p, Expr a) :
        func_name(f), new_pos(p), new_arg(a) {}
};

// -----------------------------------------------------------------------------

// Remove argument from Func call
class RemoveArgFromFunctionCall : public IRMutator {
private:
    using IRMutator::visit;
    void visit(const Call *op) {
        vector<Expr> new_args(op->args.size());
        bool changed = false;

        // Mutate the args
        for (size_t i=0; i<op->args.size(); i++) {
            Expr old_arg = op->args[i];
            Expr new_arg = mutate(old_arg);
            if (!new_arg.same_as(old_arg)) changed = true;
            new_args[i] = new_arg;
        }

        if (op->call_type==Call::Halide && op->name==func_name) {
            assert(arg_pos < new_args.size());
            new_args.erase(new_args.begin()+arg_pos);
            changed = true;
        }

        if (changed) {
            expr = Call::make(op->type, op->name, new_args, op->call_type,
                    op->func, op->value_index, op->image, op->param);
        } else {
            expr = op;
        }
    }

    string func_name;
    size_t arg_pos;

public:
    RemoveArgFromFunctionCall(string f, size_t p) :
        func_name(f), arg_pos(p) {}
};

// -----------------------------------------------------------------------------

// Substitute a calling arg in feed forward recursive calls
// to a Func; feedforward calls are those where all args in the
// Function call are identical to args in Function definition
class SubstituteArgInFeedforwardFuncCall : public IRMutator {
private:
    using IRMutator::visit;
    void visit(const Call *op) {
        vector<Expr> new_args(op->args.size());
        bool changed = false;

        // Mutate the args
        for (size_t i=0; i<op->args.size(); i++) {
            Expr old_arg = op->args[i];
            Expr new_arg = mutate(old_arg);
            if (!new_arg.same_as(old_arg)) changed = true;
            new_args[i] = new_arg;
        }

        if (op->call_type==Call::Halide && op->name==func_name) {
            assert(pos < new_args.size());
            bool feedforward = true;
            for (size_t i=0; i<new_args.size(); i++) {
                feedforward &= equal(new_args[i], def_args[i]);
            }
            if (feedforward) {
                new_args[pos] = new_arg;
                changed = true;
            }
        }

        if (changed) {
            expr = Call::make(op->type, op->name, new_args, op->call_type,
                    op->func, op->value_index, op->image, op->param);
        } else {
            expr = op;
        }
    }

    string func_name;
    vector<Expr> def_args;
    size_t pos;
    Expr   new_arg;

public:
    SubstituteArgInFeedforwardFuncCall(string f, vector<Expr> d, size_t p, Expr a) :
        func_name(f), def_args(d), pos(p), new_arg(a) {}
};

// -----------------------------------------------------------------------------

// Remove all min/max clamping in an expression
class RemoveMinMax : public IRMutator {
private:
    using IRMutator::visit;
    void visit(const Max *op) { expr = (is_const(op->a) ? op->b : op->a); }
    void visit(const Min *op) { expr = (is_const(op->a) ? op->b : op->a); }
public:
    RemoveMinMax(void) {}
};

// -----------------------------------------------------------------------------

// Apply zero boundary condition on all tiles except tiles that touch image
// borders; this pads tiles by zeros if the intra tile index is out of range -
// - less than zero or greater than tile width.
// Usually, if the feedforward coeff is 1.0 zero padding is required for in
// the Function definition itself - which automatically pads all tiles by zeros.
// But in Gaussian filtering feedforward coeff is not 1.0; so Function definition
// cannot pad the image by zeros; in such case it is important to forcibly pad
// all inner tiles by zeros.
class ApplyZeroBoundaryInFuncCall : public IRMutator {
private:
    using IRMutator::visit;
    void visit(const Call *op) {
        bool change = false;
        Expr condition;

        if (op->call_type==Call::Halide && op->name==func_name) {
            assert(dim < op->args.size());
            Expr call_arg = op->args[dim];

            // remove any clamping
            RemoveMinMax r;
            call_arg = r.mutate(simplify(call_arg));

            // not a feedback coeff if calling arg is same as defintion arg
            // check more than 0 for causal, less than width for anticausal
            if (!equal(call_arg, def_arg)) {
                bool causal = is_zero(boundary);
                condition = border_tile_condition ||
                    (causal ? call_arg>=0 : call_arg<boundary);
                change = true;
            }
        }

        if (change) {
            expr = select(condition, op, make_zero(op->type));
        } else {
            expr = op;
        }
    }

    string func_name;
    size_t dim;
    Expr   def_arg;
    Expr   boundary;
    Expr   border_tile_condition;

public:
    ApplyZeroBoundaryInFuncCall(string f, size_t d, Expr a, Expr b, Expr c) :
        func_name(f), dim(d), def_arg(a), boundary(b), border_tile_condition(c) {}
};

// -----------------------------------------------------------------------------

// Swap two calling args in Func call
class SwapCallArgsInFunctionCall : public IRMutator {
private:
    using IRMutator::visit;
    void visit(const Call *op) {
        vector<Expr> new_args(op->args.size());
        bool changed = false;

        // Mutate the args
        for (size_t i=0; i<op->args.size(); i++) {
            Expr old_arg = op->args[i];
            Expr new_arg = mutate(old_arg);
            if (!new_arg.same_as(old_arg)) changed = true;
            new_args[i] = new_arg;
        }

        if (op->call_type==Call::Halide && op->name==func_name) {
            Expr temp    = new_args[va];
            new_args[va] = new_args[vb];
            new_args[vb] = temp;
            changed = true;
        }

        if (changed) {
            expr = Call::make(op->type, op->name, new_args, op->call_type,
                    op->func, op->value_index, op->image, op->param);
        } else {
            expr = op;
        }
    }

    string func_name;
    size_t va;
    size_t vb;

public:
    SwapCallArgsInFunctionCall(string f, size_t a, size_t b) :
        func_name(f), va(a), vb(b) {}
};

// -----------------------------------------------------------------------------

// Substitute in Func call
class SubstituteInFunctionCall : public IRMutator {
private:
    using IRMutator::visit;
    void visit(const Call *op) {
        vector<Expr> new_args(op->args.size());
        bool changed = false;

        // Mutate the args and perform the substitution
        for (size_t i=0; i<op->args.size(); i++) {
            Expr old_arg = op->args[i];
            Expr new_arg = substitute(var, replace, mutate(old_arg));
            if (!new_arg.same_as(old_arg)) changed = true;
            new_args[i] = new_arg;
        }

        if (changed) {
            expr = Call::make(op->type, op->name, new_args, op->call_type,
                    op->func, op->value_index, op->image, op->param);
        } else {
            expr = op;
        }
    }

    string func_name;
    string var;
    Expr   replace;

public:
    SubstituteInFunctionCall(string f, string v, Expr r) :
        func_name(f), var(v), replace(r) {}
};

// -----------------------------------------------------------------------------

// Increment/decrement value index in Func call
class IncrementValueIndexInFunctionCall : public IRMutator {
private:
    using IRMutator::visit;
    void visit(const Call *op) {
        vector<Expr> new_args(op->args.size());
        bool changed = false;

        // Mutate the args
        for (size_t i=0; i<op->args.size(); i++) {
            Expr old_arg = op->args[i];
            Expr new_arg = mutate(old_arg);
            if (!new_arg.same_as(old_arg)) changed = true;
            new_args[i] = new_arg;
        }

        int value_index = op->value_index;
        if (op->call_type==Call::Halide && op->name==func_name) {
            changed = true;
            value_index += increment;
        }

        if (changed) {
            expr = Call::make(op->type, op->name, new_args, op->call_type,
                    op->func, value_index, op->image, op->param);
        } else {
            expr = op;
        }
    }

    string func_name;
    int increment;

public:
    IncrementValueIndexInFunctionCall(string f, int i) :
        func_name(f), increment(i) {}
};

// -----------------------------------------------------------------------------

//// Replace all calls to a pure function by its actual value
//class InlineAllFuncCalls : public IRMutator {
//private:
//    using IRMutator::visit;
//    void visit(const Call *op) {
//        if (op->call_type==Call::Halide && op->name==f.name()) {
//            vector<Expr>   call_args = op->args;
//            vector<string> args = f.args();
//            vector<string> temp_args = f.args();
//            Expr val = f.values()[op->value_index];
//            for (size_t i=0; i<call_args.size(); i++) {  // mutate calling args
//                call_args[i] = mutate(call_args[i]);     // of f
//            }
//
//            // replace all args with a tagged version of arg
//            for (size_t i=0; i<args.size(); i++) {
//                temp_args[i] = args[i] + int_to_string(rand());
//                val = simplify(substitute(args[i], Var(temp_args[i]), val));
//            }
//
//            // replace all the 'tagged' args with calling args, this is avoid
//            // situations when some arg is the same as some other calling arg
//            for (size_t i=0; i<temp_args.size(); i++) {
//                val = simplify(substitute(temp_args[i], call_args[i], val));
//            }
//            expr = val;
//        }
//        else {
//            // standard mutation routine
//            vector<Expr> new_args(op->args.size());
//            bool changed = false;
//
//            // Mutate the args
//            for (size_t i=0; i<op->args.size(); i++) {
//                Expr old_arg = op->args[i];
//                Expr new_arg = mutate(old_arg);
//                if (!new_arg.same_as(old_arg)) changed = true;
//                new_args[i] = new_arg;
//            }
//
//            if (changed) {
//                expr = Call::make(op->type, op->name, new_args, op->call_type,
//                        op->func, op->value_index, op->image, op->param);
//            } else {
//                expr = op;
//            }
//        }
//    }
//
//    Function f;
//
//public:
//    InlineAllFuncCalls(Function func) : f(func) {}
//};
//
//// -----------------------------------------------------------------------------

bool expr_depends_on_var(Expr expr, string var) {
    ExprDependsOnVar depends(var);
    expr.accept(&depends);
    return depends.result;
}

bool expr_depends_on_func(Expr expr, string func_name) {
    ExprDependsOnFunc depends(func_name);
    expr.accept(&depends);
    return depends.result;
}

Expr substitute_func_call(string func_name, Function new_func, Expr original) {
    SubstituteFunctionCall s(func_name, new_func);
    return s.mutate(original);
}

Expr remove_func_calls(string func_name, bool matching, Expr original) {
    RemoveFunctionCall s(func_name, matching);
    return simplify(s.mutate(original));
}

Expr augment_func_call(string func_name, vector<string> func_args, vector<Expr> extra, Expr original) {
    AugmentFunctionCall s(func_name, func_args, extra);
    return s.mutate(original);
}

Expr insert_arg_in_func_call(string func_name, size_t pos, Expr arg, Expr original) {
    InsertArgToFunctionCall s(func_name, pos, arg);
    return s.mutate(original);
}

Expr remove_arg_from_func_call(string func_name, size_t pos, Expr original) {
    RemoveArgFromFunctionCall s(func_name, pos);
    return s.mutate(original);
}

Expr swap_args_in_func_call(string func_name, size_t a, size_t b, Expr original) {
    SwapCallArgsInFunctionCall s(func_name, a, b);
    return s.mutate(original);
}

Expr substitute_arg_in_feedforward_func_call(string func_name, vector<Expr> def_args, size_t pos, Expr new_arg, Expr original) {
    SubstituteArgInFeedforwardFuncCall s(func_name, def_args, pos, new_arg);
    return s.mutate(remove_lets(original));
}

Expr apply_zero_boundary_in_func_call(string func_name, size_t dim, Expr arg, Expr boundary, Expr cond, Expr original) {
    ApplyZeroBoundaryInFuncCall s(func_name, dim, arg, boundary, cond);
    return s.mutate(remove_lets(original));
}

Expr substitute_in_func_call(string func_name, string var, Expr replace, Expr original) {
    SubstituteInFunctionCall s(func_name, var, replace);
    return s.mutate(original);
}

Expr increment_value_index_in_func_call(string func_name, int increment, Expr original) {
    IncrementValueIndexInFunctionCall s(func_name, increment);
    return s.mutate(original);
}

//Expr inline_func_calls(Function func, Expr original) {
//    InlineAllFuncCalls s(func);
//    return s.mutate(original);
//}

Expr swap_vars_in_expr(string a, string b, Expr original) {
    Expr value = original;
    string t = "temp_var_" + int_to_string(rand());
    value = substitute(a, Var(t), value);
    value = substitute(b, Var(a), value);
    value = substitute(t, Var(b), value);
    return value;
}


vector<string> extract_vars_or_rvars_in_expr(Expr expr) {
    ExtractVarsInExpr extract;
    expr.accept(&extract);
    return extract.var_or_rvar_list;
}

vector<string> extract_rvars_in_expr(Expr expr) {
    ExtractVarsInExpr extract;
    expr.accept(&extract);
    return extract.rvar_list;
}

vector<string> extract_params_in_expr(Expr expr) {
    ExtractVarsInExpr extract;
    expr.accept(&extract);
    return extract.param_list;
}

map<string, Func> extract_func_calls(Func func) {
    map<string, Func> func_list;
    map<string, Function> func_map = find_transitive_calls(func.function());
    map<string, Function>::iterator f_it  = func_map.begin();
    map<string, Function>::iterator f_end = func_map.end();
    while (f_it != f_end) {
        func_list.insert(make_pair(f_it->first, Func(f_it->second)));
        f_it++;
    }
    return func_list;
}
