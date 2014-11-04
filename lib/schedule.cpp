#include "recfilter.h"
#include "recfilter_internals.h"

using std::cerr;
using std::endl;
using std::vector;
using std::map;
using std::stringstream;
using std::make_pair;

using namespace Halide;

#define PURE_DEF  (-1)

vector<string> RecFilter::internal_functions(FuncTag ftag) {
    vector<string> func_list;
    map<string,RecFilterFunc>::iterator f_it  = contents.ptr->func.begin();
    map<string,RecFilterFunc>::iterator f_end = contents.ptr->func.end();
    while (f_it != f_end) {
        if (f_it->second.func_category & ftag) {
            func_list.push_back(f_it->second.func.name());
        }
        f_it++;
    }
    if (func_list.empty()) {
        cerr << "No recursive filter has the given scheduling tag " << ftag << endl;
        assert(false);
    }
    return func_list;
}

map< int,vector<Var> > RecFilter::internal_func_vars(RecFilterFunc f, VarTag vtag) {
    map< int,vector<Var> > var_list;
    map<string,VarTag>::iterator vit;
    for (vit = f.pure_var_category.begin(); vit!=f.pure_var_category.end(); vit++) {
        if (vit->second & vtag) {
            var_list[PURE_DEF].push_back(Var(vit->first));
        }
    }
    for (int i=0; i<f.update_var_category.size(); i++) {
        for (vit=f.update_var_category[i].begin(); vit!=f.update_var_category[i].end(); vit++) {
            if (vit->second & vtag) {
                var_list[i].push_back(Var(vit->first));
            }
        }
    }
    if (var_list.empty()) {
        cerr << "No variables in function " << f.func.name() <<
            " have the given scheduling tag " << vtag << endl;
        assert(false);
    }
    return var_list;
}

RecFilter& RecFilter::compute_in_global(FuncTag ftag) {
    vector<string> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc& rF = internal_function(func_list[i]);
        Func            F = Func(rF.func);

        F.compute_root();

        string schedule_op_str = "compute_root()";
        rF.schedule[PURE_DEF].push_back(schedule_op_str);
    }
    return *this;
}

RecFilter& RecFilter::compute_in_shared(FuncTag ftag) {
    vector<string> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc& rF = internal_function(func_list[i]);
        Func            F = Func(rF.func);

        // find the functions that is REINDEX_FOR_WRITE and calls this function
        vector<Func> callee_funcs;
        map<string,RecFilterFunc>::iterator f;
        for (f=contents.ptr->func.begin(); f!=contents.ptr->func.end(); f++) {
            if ((f->second.func_category & REINDEX_FOR_WRITE) && f->second.callee_func==F.name()) {
                callee_funcs.push_back(Func(f->second.func));
            }
        }

        // merge them if there are multiple such functions
        if (callee_funcs.size()>1) {
            cerr << "TODO: Merging required" << endl;
            assert(false);
        } else {
            F.compute_at(callee_funcs[0], Var::gpu_blocks());
            string schedule_op_str = "compute_at(" + callee_funcs[0].name() +
                ", Var::gpu_blocks())";
            rF.schedule[PURE_DEF].push_back(schedule_op_str);
        }
    }
    return *this;
}

RecFilter& RecFilter::split(FuncTag ftag, VarTag old, Expr factor) {
    vector<string> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc& rF = internal_function(func_list[i]);
        Func            F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::parallel(FuncTag ftag, VarTag vtag) {
    vector<string> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc& rF = internal_function(func_list[i]);
        Func            F = Func(rF.func);

        map< int,vector<Var> > vars = internal_func_vars(rF, vtag);
        map< int,vector<Var> >::iterator vit;

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            vector<Var> var_list = vit->second;
            if (def==PURE_DEF) {
                for (int i=0; i<var_list.size(); i++) {
                    F.parallel(var_list[i]);
                    stringstream s;
                    s << "parallel(Var(\"" << var_list[i].name() << "\"))";
                    rF.schedule[PURE_DEF].push_back(s.str());
                }
            } else {
                for (int i=0; i<var_list.size(); i++) {
                    F.update(def).parallel(var_list[i]);
                    stringstream s;
                    s << "parallel(Var(\"" << var_list[i].name() << "\"))";
                    rF.schedule[def].push_back(s.str());
                }
            }
        }
    }
    return *this;
}

RecFilter& RecFilter::parallel(FuncTag ftag, VarTag vtag, Expr task_size) {
    vector<string> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc& rF = internal_function(func_list[i]);
        Func            F = Func(rF.func);

        map< int,vector<Var> > vars = internal_func_vars(rF, vtag);
        map< int,vector<Var> >::iterator vit;

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            vector<Var> var_list = vit->second;
            if (def==PURE_DEF) {
                for (int i=0; i<var_list.size(); i++) {
                    F.parallel(var_list[i], task_size);
                    stringstream s;
                    s << "parallel(Var(\"" << var_list[i].name() << "\")," << task_size << ")";
                    rF.schedule[def].push_back(s.str());
                }
            } else {
                for (int i=0; i<var_list.size(); i++) {
                    F.update(def).parallel(var_list[i], task_size);
                    stringstream s;
                    s << "parallel(Var(\"" << var_list[i].name() << "\")," << task_size << ")";
                    rF.schedule[def].push_back(s.str());
                }
            }
        }
    }
    return *this;
}

RecFilter& RecFilter::unroll(FuncTag ftag, VarTag vtag) {
    vector<string> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc& rF = internal_function(func_list[i]);
        Func            F = Func(rF.func);

        map< int,vector<Var> > vars = internal_func_vars(rF, vtag);
        map< int,vector<Var> >::iterator vit;

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            vector<Var> var_list = vit->second;
            if (def==PURE_DEF) {
                for (int i=0; i<var_list.size(); i++) {
                    F.unroll(var_list[i]);
                    stringstream s;
                    s << "unroll(Var(\"" << var_list[i].name() << "\"))";
                    rF.schedule[def].push_back(s.str());
                }
            } else {
                for (int i=0; i<var_list.size(); i++) {
                    F.update(def).unroll(var_list[i]);
                    stringstream s;
                    s << "unroll(Var(\"" << var_list[i].name() << "\"))";
                    rF.schedule[def].push_back(s.str());
                }
            }
        }}
    return *this;
}

RecFilter& RecFilter::unroll(FuncTag ftag, VarTag vtag, int factor) {
    vector<string> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc& rF = internal_function(func_list[i]);
        Func            F = Func(rF.func);

        map< int,vector<Var> > vars = internal_func_vars(rF, vtag);
        map< int,vector<Var> >::iterator vit;

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            vector<Var> var_list = vit->second;
            if (def==PURE_DEF) {
                for (int i=0; i<var_list.size(); i++) {
                    F.unroll(var_list[i], factor);
                    stringstream s;
                    s << "unroll(Var(\"" << var_list[i].name() << "\")," << factor << ")";
                    rF.schedule[def].push_back(s.str());
                }
            } else {
                for (int i=0; i<var_list.size(); i++) {
                    F.update(def).unroll(var_list[i], factor);
                    stringstream s;
                    s << "unroll(Var(\"" << var_list[i].name() << "\")," << factor << ")";
                    rF.schedule[def].push_back(s.str());
                }
            }
        }
    }
    return *this;
}

RecFilter& RecFilter::vectorize(FuncTag ftag, VarTag vtag) {
    vector<string> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc& rF = internal_function(func_list[i]);
        Func            F = Func(rF.func);

        map< int,vector<Var> > vars = internal_func_vars(rF, vtag);
        map< int,vector<Var> >::iterator vit;

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            vector<Var> var_list = vit->second;
            if (def==PURE_DEF) {
                for (int i=0; i<var_list.size(); i++) {
                    F.vectorize(var_list[i]);
                    stringstream s;
                    s << "vectorize(Var(\"" << var_list[i].name() << "\"))";
                    rF.schedule[def].push_back(s.str());
                }
            } else {
                for (int i=0; i<var_list.size(); i++) {
                    F.update(def).vectorize(var_list[i]);
                    stringstream s;
                    s << "vectorize(Var(\"" << var_list[i].name() << "\"))";
                    rF.schedule[def].push_back(s.str());
                }
            }
        }
    }
    return *this;
}

RecFilter& RecFilter::vectorize(FuncTag ftag, VarTag vtag, int factor) {
    vector<string> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc& rF = internal_function(func_list[i]);
        Func            F = Func(rF.func);

        map< int,vector<Var> > vars = internal_func_vars(rF, vtag);
        map< int,vector<Var> >::iterator vit;

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            vector<Var> var_list = vit->second;
            if (def==PURE_DEF) {
                for (int i=0; i<var_list.size(); i++) {
                    F.vectorize(var_list[i], factor);
                    stringstream s;
                    s << "vectorize(Var(\"" << var_list[i].name() << "\")," << factor << ")";
                    rF.schedule[def].push_back(s.str());
                }
            } else {
                for (int i=0; i<var_list.size(); i++) {
                    F.update(def).vectorize(var_list[i], factor);
                    stringstream s;
                    s << "vectorize(Var(\"" << var_list[i].name() << "\")," << factor << ")";
                    rF.schedule[def].push_back(s.str());
                }
            }
        }
    }
    return *this;
}

RecFilter& RecFilter::bound(FuncTag ftag, VarTag vtag, Expr min, Expr extent) {
    vector<string> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc& rF = internal_function(func_list[i]);
        Func            F = Func(rF.func);

        map< int,vector<Var> > vars = internal_func_vars(rF, vtag);
        map< int,vector<Var> >::iterator vit;

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            vector<Var> var_list = vit->second;
            if (def==PURE_DEF) {
                for (int i=0; i<var_list.size(); i++) {
                    F.bound(Var(var_list[i].name()), min, extent);
                    stringstream s;
                    s << "bound(Var(\"" << var_list[i].name() << "," << min << "," << extent << ")";
                    rF.schedule[def].push_back(s.str());
                }
            }
        }
    }
    return *this;
}

RecFilter& RecFilter::reorder(FuncTag ftag, vector<VarTag> x) {
    vector<string> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc& rF = internal_function(func_list[i]);
        Func            F = Func(rF.func);

        map< int,vector<Var> > vars;
        map< int,vector<Var> >::iterator vit;

        for (int i=0; i<x.size(); i++) {
            map< int,vector<Var> > vars_x = internal_func_vars(rF, x[i]);
            for (vit=vars_x.begin(); vit!=vars_x.end(); vit++) {
                for (int j=0; j<vit->second.size(); j++) {
                    vars[vit->first].push_back(vit->second[j]);
                }
            }
        }

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            vector<Var> var_list = vit->second;
            stringstream s;

            if (var_list.size() > 2) {                 // at least 2 to reorder
                s << "reorder(Internal::vec(";
                for (int i=0; i<var_list.size(); i++) {
                    if (i>0) {
                        s << ",";
                    }
                    s << "Var(\"" << var_list[i].name() << "\")";
                }
                s << "))";

                // only support up to 6 variables to reorder
                if (def==PURE_DEF) {
                    switch (var_list.size()) {
                        case 2: F.reorder(var_list[0], var_list[1]); break;
                        case 3: F.reorder(var_list[0], var_list[1], var_list[2]); break;
                        case 4: F.reorder(var_list[0], var_list[1], var_list[2], var_list[3]); break;
                        case 5: F.reorder(var_list[0], var_list[1], var_list[2], var_list[3], var_list[4]); break;
                        case 6: F.reorder(var_list[0], var_list[1], var_list[2], var_list[3], var_list[4], var_list[5]); break;
                        default:cerr << "Too many variables in reorder()" << endl; assert(false); break;
                    }
                } else {
                    switch (var_list.size()) {
                        case 2: F.update(def).reorder(var_list[0], var_list[1]); break;
                        case 3: F.update(def).reorder(var_list[0], var_list[1], var_list[2]); break;
                        case 4: F.update(def).reorder(var_list[0], var_list[1], var_list[2], var_list[3]); break;
                        case 5: F.update(def).reorder(var_list[0], var_list[1], var_list[2], var_list[3], var_list[4]); break;
                        case 6: F.update(def).reorder(var_list[0], var_list[1], var_list[2], var_list[3], var_list[4], var_list[5]); break;
                        default:cerr << "Too many variables in reorder()" << endl; assert(false); break;
                    }
                }
                rF.schedule[def].push_back(s.str());
            }
        }
    }
    return *this;
}

RecFilter& RecFilter::reorder_storage(FuncTag ftag, vector<VarTag> x) {
    vector<string> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc& rF = internal_function(func_list[i]);
        Func            F = Func(rF.func);

        map< int,vector<Var> > vars;
        map< int,vector<Var> >::iterator vit;

        for (int i=0; i<x.size(); i++) {
            map< int,vector<Var> > vars_x = internal_func_vars(rF, x[i]);
            for (vit=vars_x.begin(); vit!=vars_x.end(); vit++) {
                for (int j=0; j<vit->second.size(); j++) {
                    vars[vit->first].push_back(vit->second[j]);
                }
            }
        }

        vector<Var> var_list = vars[PURE_DEF];
        stringstream s;

        if (var_list.size() > 2) {                 // at least 2 to reorder
            s << "reorder_storage(Internal::vec(";
            for (int i=0; i<var_list.size(); i++) {
                if (i>0) {
                    s << ",";
                }
                s << "Var(\"" << var_list[i].name() << "\")";
            }
            s << "))";

            switch (var_list.size()) {
                case 2: F.reorder_storage(var_list[0], var_list[1]); break;
                case 3: F.reorder_storage(var_list[0], var_list[1], var_list[2]); break;
                case 4: F.reorder_storage(var_list[0], var_list[1], var_list[2], var_list[3]); break;
                case 5: F.reorder_storage(var_list[0], var_list[1], var_list[2], var_list[3], var_list[4]); break;
                default:cerr << "Too many variables in reorder_storage()" << endl; assert(false); break;
            }
            rF.schedule[PURE_DEF].push_back(s.str());
        }
    }
    return *this;
}

RecFilter& RecFilter::reorder(FuncTag ftag, VarTag x, VarTag y) {
    return reorder(ftag, Internal::vec(x,y));
}

RecFilter& RecFilter::reorder(FuncTag ftag, VarTag x, VarTag y, VarTag z) {
    return reorder(ftag, Internal::vec(x,y,z));
}

RecFilter& RecFilter::reorder(FuncTag ftag, VarTag x, VarTag y, VarTag z, VarTag w) {
    return reorder(ftag, Internal::vec(x,y,z,w));
}

RecFilter& RecFilter::reorder(FuncTag ftag, VarTag x, VarTag y, VarTag z, VarTag w, VarTag t) {
    return reorder(ftag, Internal::vec(x,y,z,w,t));
}

RecFilter& RecFilter::reorder_storage(FuncTag ftag, VarTag x, VarTag y) {
    return reorder_storage(ftag, Internal::vec(x,y));
}

RecFilter& RecFilter::reorder_storage(FuncTag ftag, VarTag x, VarTag y, VarTag z) {
    return reorder_storage(ftag, Internal::vec(x,y,z));
}

RecFilter& RecFilter::reorder_storage(FuncTag ftag, VarTag x, VarTag y, VarTag z, VarTag w) {
    return reorder(ftag, Internal::vec(x,y,z,w));
}

RecFilter& RecFilter::reorder_storage(FuncTag ftag, VarTag x, VarTag y, VarTag z, VarTag w, VarTag t) {
    return reorder_storage(ftag, Internal::vec(x,y,z,w,t));
}

RecFilter& RecFilter::gpu_threads(FuncTag ftag, VarTag thread_x) {
    vector<string> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc& rF = internal_function(func_list[i]);
        Func            F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::gpu_threads(FuncTag ftag, VarTag thread_x, VarTag thread_y) {
    vector<string> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc& rF = internal_function(func_list[i]);
        Func            F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::gpu_threads(FuncTag ftag, VarTag thread_x, VarTag thread_y, VarTag thread_z) {
    vector<string> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc& rF = internal_function(func_list[i]);
        Func            F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::gpu_blocks(FuncTag ftag, VarTag block_x) {
    vector<string> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc& rF = internal_function(func_list[i]);
        Func            F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::gpu_blocks(FuncTag ftag, VarTag block_x, VarTag block_y) {
    vector<string> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc& rF = internal_function(func_list[i]);
        Func            F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::gpu_tile(FuncTag ftag, VarTag x, int x_size) {
    vector<string> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc& rF = internal_function(func_list[i]);
        Func            F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::gpu_tile(FuncTag ftag, VarTag x, VarTag y, int x_size, int y_size) {
    vector<string> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc& rF = internal_function(func_list[i]);
        Func            F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::gpu_tile(FuncTag ftag, VarTag x, VarTag y, VarTag z, int x_size, int y_size, int z_size) {
    vector<string> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc& rF = internal_function(func_list[i]);
        Func            F = Func(rF.func);
    }
    return *this;
}
