#include "split.h"

using namespace Halide;
using namespace Halide::Internal;

using std::string;
using std::vector;
using std::set;
using std::map;

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

// Extract all function calls in an Expr
class ExtractFuncCalls : public IRVisitor {
private:
    using IRVisitor::visit;
    void visit(const Call *op) {
        for (size_t i=0; i<op->args.size(); i++) {
            op->args[i].accept(this);
        }

        // Consider extern call args
        Function f = op->func;
        if (op->call_type==Call::Halide && f.has_extern_definition()) {
            for (size_t i=0; i<f.extern_arguments().size(); i++) {
                ExternFuncArgument arg = f.extern_arguments()[i];
                if (arg.is_expr()) {
                    arg.expr.accept(this);
                }
            }
        }

        if (op->call_type==Call::Halide && op->func.has_pure_definition()) {
            extract_func_calls(Func(op->func), func_list); // recursively extract all calls
        }
    }
public:
    vector<Func> func_list;
    ExtractFuncCalls(void) {}
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
    int va;
    int vb;

public:
    SwapCallArgsInFunctionCall(string f, int a, int b) :
        func_name(f), va(a), vb(b) {}
};

// -----------------------------------------------------------------------------

// Replace all calls to a pure function by its actual value
class InlineAllFuncCalls : public IRMutator {
private:
    using IRMutator::visit;
    void visit(const Call *op) {
        if (op->call_type==Call::Halide && op->name==f.name()) {
            vector<Expr>   call_args = op->args;
            vector<string> args = f.args();
            vector<string> temp_args = f.args();
            Expr val = f.values()[op->value_index];
            for (size_t i=0; i<call_args.size(); i++) {  // mutate calling args
                call_args[i] = mutate(call_args[i]);     // of f
            }

            // replace all args with a tagged version of arg
            for (size_t i=0; i<args.size(); i++) {
                temp_args[i] = args[i] + int_to_string(rand());
                val = simplify(substitute(args[i], Var(temp_args[i]), val));
            }

            // replace all the 'tagged' args with calling args, this is avoid
            // situations when some arg is the same as some other calling arg
            for (size_t i=0; i<temp_args.size(); i++) {
                val = simplify(substitute(temp_args[i], call_args[i], val));
            }
            expr = val;
        }
        else {
            // standard mutation routine
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
        }
    }

    Function f;

public:
    InlineAllFuncCalls(Function func) : f(func) {}
};

// -----------------------------------------------------------------------------

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

Expr insert_arg_to_func_call(string func_name, size_t pos, Expr arg, Expr original) {
    InsertArgToFunctionCall s(func_name, pos, arg);
    return s.mutate(original);
}

Expr substitute_in_func_call(string func_name, string var, Expr replace, Expr original) {
    SubstituteInFunctionCall s(func_name, var, replace);
    return s.mutate(original);
}

Expr increment_value_index_in_func_call(string func_name, int increment, Expr original) {
    IncrementValueIndexInFunctionCall s(func_name, increment);
    return s.mutate(original);
}

Expr inline_func_calls(Function func, Expr original) {
    InlineAllFuncCalls s(func);
    return s.mutate(original);
}

Expr swap_vars_in_expr(string a, string b, Expr original) {
    Expr value = original;
    string t = "temp_var_" + int_to_string(rand());
    value = substitute(a, Var(t), value);
    value = substitute(b, Var(a), value);
    value = substitute(t, Var(b), value);
    return value;
}


Expr swap_callargs_in_func_call(string func_name, int a, int b, Expr original) {
    SwapCallArgsInFunctionCall s(func_name, a, b);
    return s.mutate(original);
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

void extract_func_calls(Func func, vector<Func>& func_list) {
    ExtractFuncCalls extract;
    for (int i=0; i<func.outputs(); i++) {
        if (func.name().find("NO_REVEAL") == string::npos) {
            Expr expr = func.values()[i];
            expr.accept(&extract);
            func_list.push_back(func);
            func_list.insert(func_list.end(),
                    extract.func_list.begin(), extract.func_list.end());
        }
    }

    // remove duplicates
    set<string> func_names;
    for (size_t i=0; i<func_list.size(); i++) {
        string name = func_list[i].name();
        if (func_names.find(name) == func_names.end()) {
            func_names.insert(name);
        } else {
            func_list.erase(func_list.begin()+i);
            i--;
        }
    }
}
