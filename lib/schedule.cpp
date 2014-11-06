#include "recfilter.h"
#include "recfilter_internals.h"

using std::cerr;
using std::endl;
using std::vector;
using std::map;
using std::string;
using std::stringstream;
using std::pair;
using std::make_pair;

using namespace Halide;

#define PURE_DEF  (-1)

RecFilter& RecFilter::compute_in_global(FuncTag ftag) {
    vector<string> func_list = internal_functions(ftag);
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = internal_function(func_list[j]);
        Func            F = Func(rF.func);

        F.compute_root();

        string schedule_op_str = "compute_root()";
        rF.schedule[PURE_DEF].push_back(schedule_op_str);
    }
    return *this;
}

RecFilter& RecFilter::compute_in_shared(FuncTag ftag) {
    vector<string> func_list = internal_functions(ftag);
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = internal_function(func_list[j]);
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
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = internal_function(func_list[j]);
        Func            F = Func(rF.func);

        // TODO
    }
    return *this;
}

RecFilter& RecFilter::parallel(FuncTag ftag, VarTag vtag) {
    vector<string> func_list = internal_functions(ftag);
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = internal_function(func_list[j]);
        Func            F = Func(rF.func);

        map< int,vector<VarOrRVar> > vars = internal_func_vars(rF, vtag);
        map< int,vector<VarOrRVar> >::iterator vit;

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            vector<VarOrRVar> var_list = vit->second;
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
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = internal_function(func_list[j]);
        Func            F = Func(rF.func);

        map< int,vector<VarOrRVar> > vars = internal_func_vars(rF, vtag);
        map< int,vector<VarOrRVar> >::iterator vit;

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            vector<VarOrRVar> var_list = vit->second;
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
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = internal_function(func_list[j]);
        Func            F = Func(rF.func);

        map< int,vector<VarOrRVar> > vars = internal_func_vars(rF, vtag);
        map< int,vector<VarOrRVar> >::iterator vit;

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            vector<VarOrRVar> var_list = vit->second;
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
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = internal_function(func_list[j]);
        Func            F = Func(rF.func);

        map< int,vector<VarOrRVar> > vars = internal_func_vars(rF, vtag);
        map< int,vector<VarOrRVar> >::iterator vit;

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            vector<VarOrRVar> var_list = vit->second;
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
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = internal_function(func_list[j]);
        Func            F = Func(rF.func);

        map< int,vector<VarOrRVar> > vars = internal_func_vars(rF, vtag);
        map< int,vector<VarOrRVar> >::iterator vit;

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            vector<VarOrRVar> var_list = vit->second;
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
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = internal_function(func_list[j]);
        Func            F = Func(rF.func);

        map< int,vector<VarOrRVar> > vars = internal_func_vars(rF, vtag);
        map< int,vector<VarOrRVar> >::iterator vit;

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            vector<VarOrRVar> var_list = vit->second;
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
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = internal_function(func_list[j]);
        Func            F = Func(rF.func);

        map< int,vector<VarOrRVar> > vars = internal_func_vars(rF, vtag);
        map< int,vector<VarOrRVar> >::iterator vit;

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            vector<VarOrRVar> var_list = vit->second;
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

RecFilter& RecFilter::gpu_threads(FuncTag ftag, VarTag thread, Expr task_size) {
    vector<string> func_list = internal_functions(ftag);
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = internal_function(func_list[j]);
        Func            F = Func(rF.func);

        map< int,vector<VarOrRVar> > vars = internal_func_vars(rF, thread);
        map< int,vector<VarOrRVar> >::iterator vit;
        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            if (vit->second.size()>1) {
                cerr << "Multiple vars found with same tag " << thread
                    << " in function " << F.name()
                    << ", cannot assign all to same GPU thread index" << endl;
                assert(false);
            }
        }

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            VarOrRVar v = vit->second[0].var;
            stringstream s;

            Var t(v.name()+".split");

            s << "split(Var(\""   << v.name() << "\"), Var(\"" << v.name() << "\"), Var(\"" << t.name() << "\"), " << task_size << ").";
            s << "reorder(Var(\"" << t.name() << "\"), Var(\"" << v.name() << "\")).";
            s << "gpu_threads(Var(\"" << v.name() << "\")).";
            s << "unroll(Var(\""      << t.name() << "\"))";

            if (def==PURE_DEF) {
                F.split(v,v,t,task_size).reorder(t,v).gpu_threads(v).unroll(t);
            } else {
                F.update(def).split(v,v,t,task_size).reorder(t,v).gpu_threads(v).unroll(t);
            }
            rF.schedule[def].push_back(s.str());
        }
    }
    return *this;
}

RecFilter& RecFilter::gpu_threads(FuncTag ftag, VarTag thread_x) {
    return gpu_threads(ftag, Internal::vec(thread_x));
}

RecFilter& RecFilter::gpu_threads(FuncTag ftag, VarTag thread_x, VarTag thread_y) {
    return gpu_threads(ftag, Internal::vec(thread_x,thread_y));
}

RecFilter& RecFilter::gpu_threads(FuncTag ftag, VarTag thread_x, VarTag thread_y, VarTag thread_z) {
    return gpu_threads(ftag, Internal::vec(thread_x,thread_y,thread_z));
}

RecFilter& RecFilter::gpu_blocks(FuncTag ftag, VarTag block_x) {
    return gpu_blocks(ftag, Internal::vec(block_x));
}

RecFilter& RecFilter::gpu_blocks(FuncTag ftag, VarTag block_x, VarTag block_y) {
    return gpu_blocks(ftag, Internal::vec(block_x, block_y));
}

RecFilter& RecFilter::gpu_blocks(FuncTag ftag, VarTag block_x, VarTag block_y, VarTag block_z) {
    return gpu_blocks(ftag, Internal::vec(block_x, block_y, block_z));
}

RecFilter& RecFilter::gpu_tile(FuncTag ftag, VarTag x, int x_size) {
    return gpu_tile(ftag, Internal::vec(make_pair(x,x_size)));
}

RecFilter& RecFilter::gpu_tile(FuncTag ftag, VarTag x, VarTag y, int x_size, int y_size) {
    return gpu_tile(ftag, Internal::vec(make_pair(x,x_size), make_pair(y,y_size)));
}

RecFilter& RecFilter::gpu_tile(FuncTag ftag, VarTag x, VarTag y, VarTag z, int x_size, int y_size, int z_size) {
    return gpu_tile(ftag, Internal::vec(make_pair(x,x_size), make_pair(y,y_size), make_pair(z,z_size)));
}

RecFilter& RecFilter::reorder(FuncTag ftag, vector<VarTag> x) {
    vector<string> func_list = internal_functions(ftag);
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = internal_function(func_list[j]);
        Func            F = Func(rF.func);

        map< int,vector<VarOrRVar> > vars;
        map< int,vector<VarOrRVar> >::iterator vit;

        for (int i=0; i<x.size(); i++) {
            map< int,vector<VarOrRVar> > vars_x = internal_func_vars(rF, x[i]);
            for (vit=vars_x.begin(); vit!=vars_x.end(); vit++) {
                for (int k=0; k<vit->second.size(); k++) {
                    vars[vit->first].push_back(vit->second[k]);
                }
            }
        }

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            vector<VarOrRVar> var_list = vit->second;
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

                if (def==PURE_DEF) {
                    F.reorder(var_list); break;
                } else {
                    F.update(def).reorder(var_list); break;
                }
                rF.schedule[def].push_back(s.str());
            }
        }
    }
    return *this;
}

RecFilter& RecFilter::reorder_storage(FuncTag ftag, vector<VarTag> x) {
    vector<string> func_list = internal_functions(ftag);
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = internal_function(func_list[j]);
        Func            F = Func(rF.func);

        map< int,vector<VarOrRVar> > vars;
        map< int,vector<VarOrRVar> >::iterator vit;

        for (int i=0; i<x.size(); i++) {
            map< int,vector<VarOrRVar> > vars_x = internal_func_vars(rF, x[i]);
            for (vit=vars_x.begin(); vit!=vars_x.end(); vit++) {
                for (int k=0; k<vit->second.size(); k++) {
                    vars[vit->first].push_back(vit->second[k]);
                }
            }
        }

        vector<VarOrRVar> var_list = vars[PURE_DEF];
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
                case 2: F.reorder_storage(var_list[0].var,
                                          var_list[1].var); break;
                case 3: F.reorder_storage(var_list[0].var,
                                          var_list[1].var,
                                          var_list[2].var); break;
                case 4: F.reorder_storage(var_list[0].var,
                                          var_list[1].var,
                                          var_list[2].var,
                                          var_list[3].var); break;
                case 5: F.reorder_storage(var_list[0].var,
                                          var_list[1].var,
                                          var_list[2].var,
                                          var_list[3].var,
                                          var_list[4].var); break;
                default:cerr << "Too many variables in reorder_storage()" << endl; assert(false); break;
            }
            rF.schedule[PURE_DEF].push_back(s.str());
        }
    }
    return *this;
}

RecFilter& RecFilter::gpu_threads(FuncTag ftag, vector<VarTag> thread) {
    if (thread.empty() || thread.size()>3) {
        cerr << "GPU thread grid dimensions must be between 1 and 3" << endl;
        assert(false);
    }

    for (int i=0; i<thread.size(); i++) {
        for (int j=i+1; j<thread.size(); j++) {
            if (thread[i] == thread[j]) {
                cerr << "Same variable tag mapped to multiple GPU thread" << endl;
                assert(false);
            }
        }
    }

    vector<string> func_list = internal_functions(ftag);
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = internal_function(func_list[j]);
        Func            F = Func(rF.func);

        map< int,vector<VarOrRVar> > vars;
        map< int,vector<VarOrRVar> >::iterator vit;

        for (int i=0; i<thread.size(); i++) {
            map< int,vector<VarOrRVar> > vars_x = internal_func_vars(rF, thread[i]);
            for (vit=vars_x.begin(); vit!=vars_x.end(); vit++) {
                if (vit->second.size()>1) {
                    cerr << "Multiple vars found with same tag " << thread[i]
                         << " in function " << F.name()
                         << ", cannot assign all to same GPU thread index" << endl;
                    assert(false);
                }
                for (int k=0; k<vit->second.size(); j++) {
                    vars[vit->first].push_back(vit->second[k]);
                }
            }
        }

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            vector<VarOrRVar> var_list = vit->second;
            stringstream s;

            if (var_list.size()) {
                s << "gpu_threads(";
                for (int i=0; i<var_list.size(); i++) {
                    if (i>0) {
                        s << ",";
                    }
                    s << "Var(\"" << var_list[i].name() << "\")";
                }
                s << ")";

                // only support up to 3 variables
                if (def==PURE_DEF) {
                    switch (var_list.size()) {
                        case 1: F.gpu_threads(var_list[0]); break;
                        case 2: F.gpu_threads(var_list[0], var_list[1]); break;
                        case 3: F.gpu_threads(var_list[0], var_list[1], var_list[2]); break;
                        default:cerr << "Too many variables in gpu_threads()" << endl; assert(false); break;
                    }
                } else {
                    switch (var_list.size()) {
                        case 1: F.update(def).gpu_threads(var_list[0]); break;
                        case 2: F.update(def).gpu_threads(var_list[0], var_list[1]); break;
                        case 3: F.update(def).gpu_threads(var_list[0], var_list[1], var_list[2]); break;
                        default:cerr << "Too many variables in gpu_threads()" << endl; assert(false); break;
                    }
                }
                rF.schedule[def].push_back(s.str());
            }
        }
    }
    return *this;
}

RecFilter& RecFilter::gpu_blocks(FuncTag ftag, vector<VarTag> block) {
    if (block.empty() || block.size()>3) {
        cerr << "GPU block grid dimensions must be between 1 and 3" << endl;
        assert(false);
    }

    for (int i=0; i<block.size(); i++) {
        for (int j=i+1; j<block.size(); j++) {
            if (block[i] == block[j]) {
                cerr << "Same variable tag mapped to multiple GPU blocks" << endl;
                assert(false);
            }
        }
    }

    vector<string> func_list = internal_functions(ftag);
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = internal_function(func_list[j]);
        Func            F = Func(rF.func);

        map< int,vector<VarOrRVar> > vars;
        map< int,vector<VarOrRVar> >::iterator vit;

        for (int i=0; i<block.size(); i++) {
            map< int,vector<VarOrRVar> > vars_x = internal_func_vars(rF, block[i]);
            for (vit=vars_x.begin(); vit!=vars_x.end(); vit++) {
                if (vit->second.size()>1) {
                    cerr << "Multiple vars found with same tag " << block[i]
                         << " in function " << F.name()
                         << ", cannot assign all to same GPU block" << endl;
                    assert(false);
                }
                for (int k=0; k<vit->second.size(); k++) {
                    vars[vit->first].push_back(vit->second[k]);
                }
            }
        }

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            vector<VarOrRVar> var_list = vit->second;
            stringstream s;

            if (var_list.size()) {
                s << "gpu_blocks(";
                for (int i=0; i<var_list.size(); i++) {
                    if (i>0) {
                        s << ",";
                    }
                    s << "Var(\"" << var_list[i].name() << "\")";
                }
                s << ")";

                // only support up to 3 variables
                if (def==PURE_DEF) {
                    switch (var_list.size()) {
                        case 1: F.gpu_blocks(var_list[0]); break;
                        case 2: F.gpu_blocks(var_list[0], var_list[1]); break;
                        case 3: F.gpu_blocks(var_list[0], var_list[1], var_list[2]); break;
                        default:cerr << "Too many variables in gpu_blocks()" << endl; assert(false); break;
                    }
                } else {
                    switch (var_list.size()) {
                        case 1: F.update(def).gpu_blocks(var_list[0]); break;
                        case 2: F.update(def).gpu_blocks(var_list[0], var_list[1]); break;
                        case 3: F.update(def).gpu_blocks(var_list[0], var_list[1], var_list[2]); break;
                        default:cerr << "Too many variables in gpu_blocks()" << endl; assert(false); break;
                    }
                }
                rF.schedule[def].push_back(s.str());
            }
        }
    }
    return *this;
}

RecFilter& RecFilter::gpu_tile(FuncTag ftag, vector< pair<VarTag,int> > tile) {
    if (tile.empty() || tile.size()>3) {
        cerr << "GPU thread/block grid dimensions must be between 1 and 3" << endl;
        assert(false);
    }

    for (int i=0; i<tile.size(); i++) {
        for (int j=i+1; j<tile.size(); j++) {
            if (tile[i].first == tile[j].first) {
                cerr << "Same variable tag mapped to multiple GPU block/thread tiling factors" << endl;
                assert(false);
            }
        }
    }

    vector<string> func_list = internal_functions(ftag);
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = internal_function(func_list[j]);
        Func            F = Func(rF.func);

        map< int,vector<pair<VarOrRVar,int> > > vars;

        for (int i=0; i<tile.size(); i++) {
            map< int,vector<VarOrRVar> > vars_x = internal_func_vars(rF, tile[i].first);
            map< int,vector<VarOrRVar> >::iterator vit;

            for (vit=vars_x.begin(); vit!=vars_x.end(); vit++) {
                if (vit->second.size()>1) {
                    cerr << "Multiple vars found with same tag " << tile[i].first
                         << " in function " << F.name()
                         << ", cannot assign all to same GPU block/thread" << endl;
                    assert(false);
                }
                vars[vit->first].push_back(make_pair(vit->second[0], tile[i].second));
            }
        }

        map< int,vector< pair<VarOrRVar,int> > >::iterator vit;
        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            vector< pair<VarOrRVar,int> > var_list = vit->second;
            stringstream s;

            if (var_list.size()) {
                s << "gpu_tile(";
                for (int i=0; i<var_list.size(); i++) {
                    if (i>0) {
                        s << ",";
                    }
                    s << "Var(\"" << var_list[i].first.name() << "\")";
                }
                for (int i=0; i<var_list.size(); i++) {
                    s << var_list[i].second;
                }
                s << ")";

                // only support up to 3 variables
                if (def==PURE_DEF) {
                    switch (var_list.size()) {
                        case 1: F.gpu_tile(var_list[0].first,
                                           var_list[0].second); break;
                        case 2: F.gpu_tile(var_list[0].first,
                                           var_list[1].first,
                                           var_list[0].second,
                                           var_list[1].second); break;
                        case 3: F.gpu_tile(var_list[0].first,
                                           var_list[1].first,
                                           var_list[2].first,
                                           var_list[0].second,
                                           var_list[1].second,
                                           var_list[2].second); break;
                        default:cerr << "Too many variables in gpu_tile()" << endl; assert(false); break;
                    }
                } else {
                    switch (var_list.size()) {
                        case 1: F.update(def).gpu_tile(var_list[0].first,
                                                       var_list[0].second); break;
                        case 2: F.update(def).gpu_tile(var_list[0].first,
                                                       var_list[1].first,
                                                       var_list[0].second,
                                                       var_list[1].second); break;
                        case 3: F.update(def).gpu_tile(var_list[0].first,
                                                       var_list[1].first,
                                                       var_list[2].first,
                                                       var_list[0].second,
                                                       var_list[1].second,
                                                       var_list[2].second); break;
                        default:cerr << "Too many variables in gpu_tile()" << endl; assert(false); break;
                    }
                }
                rF.schedule[def].push_back(s.str());
            }
        }
    }
    return *this;
}

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

map< int,vector<VarOrRVar> > RecFilter::internal_func_vars(RecFilterFunc f, VarTag vtag) {
    map< int,vector<VarOrRVar> > var_list;
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


