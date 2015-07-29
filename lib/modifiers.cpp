#include "recfilter.h"
#include "recfilter_internals.h"
#include "modifiers.h"

using namespace Halide;
using namespace Halide::Internal;

using std::string;
using std::vector;
using std::set;
using std::map;
using std::cerr;
using std::endl;
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
    void visit(const Variable *op) { result = (op->name==var); }
    string var;
public:
    bool result;
    ExprDependsOnVar(string v) : result(false), var(v) {}
};

// -----------------------------------------------------------------------------

// Does an expr depend on calls to a particular function?
class ExprDependsOnFunc : public IRVisitor {
private:
    using IRVisitor::visit;
    void visit(const Call *op) {
        result |= (op->call_type==Call::Halide && op->name==func_name);

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
    ExprDependsOnFunc(string f) : result(false), func_name(f) {}
};

// -----------------------------------------------------------------------------

// Extract buffer calls referenced in Expr
class ExtractBufferCallsInExpr : public IRVisitor {
private:
    using IRVisitor::visit;
    void visit(const Call *op) {
        if (op->image.defined()) {
            buff_list[op->image.name()] = op->image;
        }
        if (op->param.defined() && op->param.is_buffer()) {
            buff_list[op->param.name()] = op->param.get_buffer();
        }

        // recursive calls
        for (int i=0; i<op->args.size(); i++) {
            op->args[i].accept(this);
        }

        // Consider extern call args
        Function f = op->func;
        if (op->call_type == Call::Halide && f.has_extern_definition()) {
            for (int i=0; i<f.extern_arguments().size(); i++) {
                ExternFuncArgument arg = f.extern_arguments()[i];
                if (arg.is_expr()) {
                    arg.expr.accept(this);
                }
            }
        }
    }

public:
    map<string,Buffer> buff_list;

    ExtractBufferCallsInExpr(map<string,Buffer> b) : buff_list(b) {}
};

// -----------------------------------------------------------------------------

class RemoveLets : public IRMutator {
private:
    Expr canonicalize(Expr e) {
        set<Expr, IRDeepCompare>::iterator i = canonical.find(e);
        if (i != canonical.end()) {
            return *i;
        } else {
            canonical.insert(e);
            return e;
        }
    }

    using IRMutator::mutate;

    Expr find_replacement(Expr e) {
        for (size_t i = replacement.size(); i > 0; i--) {
            map<Expr, Expr, ExprCompare>::iterator iter = replacement[i-1].find(e);
            if (iter != replacement[i-1].end()) return iter->second;
        }
        return Expr();
    }

    void add_replacement(Expr key, Expr value) {
        replacement[replacement.size()-1][key] = value;
    }

    void enter_scope() {
        replacement.resize(replacement.size()+1);
    }

    void leave_scope() {
        replacement.pop_back();
    }

    using IRMutator::visit;

    void visit(const Let *let) {
        Expr var = canonicalize(Variable::make(let->value.type(), let->name));

        Expr new_value = mutate(let->value);
        enter_scope();
        add_replacement(var, new_value);
        expr = mutate(let->body);
        leave_scope();
    }

    void visit(const LetStmt *let) {
        Expr var = canonicalize(Variable::make(let->value.type(), let->name));
        Expr new_value = mutate(let->value);
        enter_scope();
        add_replacement(var, new_value);
        stmt = mutate(let->body);
        leave_scope();
    }

public:
    set<Expr, IRDeepCompare> canonical;
    vector<map<Expr, Expr, ExprCompare> > replacement;

    RemoveLets() {
        enter_scope();
    }

    Expr mutate(Expr e) {
        e = canonicalize(e);

        Expr r = find_replacement(e);
        if (r.defined()) {
            return r;
        } else {
            Expr new_expr = canonicalize(IRMutator::mutate(e));
            add_replacement(e, new_expr);
            return new_expr;
        }


    }

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

class SubstituteFunctionCallWithCallArgs : public IRMutator {
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

        if (op->call_type==Call::Halide && op->name==func_name) {
            bool identical_call_args = (new_args.size() == call_args.size());
            for (int i=0; identical_call_args && i<call_args.size(); i++) {
                identical_call_args &= equal(call_args[i], new_args[i]);
            }
            if (identical_call_args) {
                expr = replace;
            }
        }
    }

    string func_name;
    Expr   replace;
    vector<Expr> call_args;

public:
    SubstituteFunctionCallWithCallArgs(string f, vector<Expr> c, Expr r)
        : func_name(f), replace(r), call_args(c) {}
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

bool is_undef(Expr e) {
    return equal(e, undef(e.type()));
}

bool is_undef(Tuple t) {
    return is_undef(t.as_vector());
}

bool is_undef(vector<Expr> e) {
    bool result = true;
    for (int i=0; i<e.size(); i++) {
        result &= is_undef(e[i]);
    }
    return result;
}

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

Expr remove_lets(Expr e) {
    return RemoveLets().mutate(e);
}

Expr substitute_func_call(string func_name, Function new_func, Expr original) {
    SubstituteFunctionCall s(func_name, new_func);
    return s.mutate(original);
}

Expr substitute_func_call_with_args(string func_name, vector<Expr> call_args, Expr replace, Expr original) {
    SubstituteFunctionCallWithCallArgs s(func_name, call_args, replace);
    return simplify(s.mutate(original));
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

Expr swap_vars_in_expr(string a, string b, Expr original) {
    Expr value = original;
    string t = "temp_var_" + int_to_string(rand());
    value = substitute(a, Var(t), value);
    value = substitute(b, Var(a), value);
    value = substitute(t, Var(b), value);
    return value;
}

map<string, Buffer> extract_buffer_calls(Func func) {
    map<string, Buffer>   buff_list;
    map<string, Function> func_map = find_transitive_calls(func.function());
    map<string, Function>::iterator f_it;

    ExtractBufferCallsInExpr extract(buff_list);

    for (f_it=func_map.begin(); f_it!=func_map.end(); f_it++) {
        Function f = f_it->second;

        // all buffers in pure values
        for (int i=0; i<f.values().size(); i++) {
            f.values()[i].accept(&extract);
        }

        // all buffers in update args and values
        for (int j=0; j<f.updates().size(); j++) {
            for (int i=0; i<f.updates()[j].args.size(); i++) {
                f.updates()[j].args[i].accept(&extract);
            }
            for (int i=0; i<f.updates()[j].values.size(); i++) {
                f.updates()[j].values[i].accept(&extract);
            }
        }
    }

    return extract.buff_list;
}

// -----------------------------------------------------------------------------

void reassign_vartag_counts(
        map<string,VarTag>& var_tags,
        vector<string> args,
        map<string,string> var_splits)
{
    vector<Expr> args_expr;
    for (int i=0; i<args.size(); i++) {
        args_expr.push_back(Var(args[i]));
    }
    reassign_vartag_counts(var_tags, args_expr, var_splits);
}

void reassign_vartag_counts(
        map<string,VarTag>& var_tags,
        vector<Expr> args,
        map<string,string> var_splits)
{
    // repeat the process for INNER, OUTER and FULL variables
    // these are the only ones which have counts
    vector<VariableTag> ref_vartag = {INNER, OUTER, FULL};

    for (int u=0; u<ref_vartag.size(); u++) {
        VariableTag ref = ref_vartag[u];

        // list of vars is ordered by their dimension in the Func
        map<int, vector<string> > vartag_count;

        map<string,VarTag>::iterator vartag_it = var_tags.begin();
        for (; vartag_it!=var_tags.end(); vartag_it++) {
            string var = vartag_it->first;
            VarTag tag = vartag_it->second;

            // no need to touch the count if this is a SCAN or SPLIT var
            if (tag.check(ref) && !tag.check(SPLIT) && !tag.check(SCAN)) {
                bool processed = false;
                string original_var = var;

                // if this variable was created by RecFilterSchedule::split() then
                // use the original var; do recursively because the original var
                // might itself be created by RecFilterSchedule::split()
                while (var_splits.find(original_var) != var_splits.end()) {
                    original_var = var_splits[original_var];
                }

                // find the arg which depends upon this variable, use its index
                // to assign a count to this variable
                for (int i=0; !processed && i<args.size(); i++) {
                    if (expr_depends_on_var(args[i], original_var)) {
                        vartag_count[i].push_back(var);
                        processed = true;
                    }
                }

                if (!processed) {
                    cerr << "Variable " << var << " could not be reassigned a new VarTag count "
                        << "because it was not found in list of args as well as list of variables "
                        << "created by RecFilterSchedule::split()" << endl;
                    assert(false);
                }
            }
        }

        // extract the vars and put them in sorted order according to their count
        int next_count = 0;
        map<int, vector<string> >::iterator vartag_count_it = vartag_count.begin();
        for (; vartag_count_it!=vartag_count.end(); vartag_count_it++) {
            vector<string> var_list = vartag_count_it->second;
            for (int i=0; i<var_list.size(); i++) {
                string v = var_list[i];
                var_tags[v] = VarTag(ref,next_count);
                next_count++;
            }
        }
    }
}

// -----------------------------------------------------------------------------

void inline_function(Function f, vector<Func> func_list) {
    if (!f.is_pure()) {
        cerr << "Function " << f.name() << " to be inlined must be pure" << endl;
        assert(false);
    }

    // go to all other functions and inline calls to f
    for (int j=0; j<func_list.size(); j++) {
        Function g = func_list[j].function();

        // check if g not same as f and g calls f
        map<string,Function> called_funcs = find_direct_calls(g);
        if (g.name()==f.name() || called_funcs.find(f.name())==called_funcs.end()) {
            continue;
        }

        vector<string> args   = g.args();
        vector<Expr>   values = g.values();
        vector<UpdateDefinition> updates = g.updates();

        for (int k=0; k<values.size(); k++) {
            values[k] = inline_function(values[k], f);
        }
        g.clear_all_definitions();
        g.define(args, values);

        for (int k=0; k<updates.size(); k++) {
            vector<Expr> update_args   = updates[k].args;
            vector<Expr> update_values = updates[k].values;
            for (int u=0; u<update_args.size(); u++) {
                update_args[u] = inline_function(update_args[u], f);
            }
            for (int u=0; u<update_values.size(); u++) {
                update_values[u] = inline_function(update_values[u], f);
            }
            g.define_update(update_args, update_values);
        }
    }
}

