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

static const VarOrRVar GPU_THREAD[] = {
    VarOrRVar("__thread_id_x",false),
    VarOrRVar("__thread_id_y",false),
    VarOrRVar("__thread_id_z",false),
};

static const VarOrRVar GPU_BLOCK[] = {
    VarOrRVar("__block_id_x",false),
    VarOrRVar("__block_id_y",false),
    VarOrRVar("__block_id_z",false),
};

// -----------------------------------------------------------------------------

RecFilter& RecFilter::compute_in_global(FuncTag ftag) {
    vector<string> func_list = internal_functions(ftag);
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = internal_function(func_list[j]);
        Func            F = Func(rF.func);

        F.compute_root();

        // remove the initializations of all scans which are scheduled as compute_root
        // to avoid extra kernel execution for initializing the output buffer
        if (F.has_update_definition()) {
            remove_pure_def(F.name());
        }

        rF.schedule[PURE_DEF].push_back("compute_root()");
    }
    return *this;
}

RecFilter& RecFilter::compute_in_shared(FuncTag ftag) {
    vector<string> func_list = internal_functions(ftag);
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = internal_function(func_list[j]);
        Func            F = Func(rF.func);

        // find the functions that are REINDEX_FOR_WRITE or REINDEX_FOR_READ
        // and associated with this this function
        Func         callee_func;
        vector<Func> caller_funcs;
        map<string,RecFilterFunc>::iterator f;
        for (f=contents.ptr->func.begin(); f!=contents.ptr->func.end(); f++) {
            if ((f->second.func_category & REINDEX_FOR_WRITE) && f->second.callee_func==F.name()) {
                if (callee_func.defined()) {
                    cerr << F.name() << " cannot be computed in shared mem "
                        << "because it is called by multiple functions" << endl;
                    assert(false);
                }
                callee_func = Func(f->second.func);
            }
            if ((f->second.func_category & REINDEX_FOR_READ) && f->second.caller_func==F.name()) {
                caller_funcs.push_back(Func(f->second.func));
            }
        }

        // function that calls this function must be computed in the global mem
        if (callee_func.defined()) {
            callee_func.compute_root();
            internal_function(callee_func.name()).
                schedule[PURE_DEF].push_back("compute_root()");
        } else {
            cerr << F.name() << " cannot be computed in shared mem "
                << "because it is not called by any function" << endl;
            assert(false);
        }

        // all the functions which are called in this function should also
        // be computed at same level
        for (int i=0; i<caller_funcs.size(); i++) {
            Func f = caller_funcs[i];
            if (f.defined()) {
                f.compute_at(callee_func, Var::gpu_blocks());
                stringstream s;
                s << "compute_at(" << callee_func.name() << ", Var::gpu_blocks())";
                internal_function(f.name()).schedule[PURE_DEF].push_back(s.str());
            }
        }

        F.compute_at(callee_func, Var::gpu_blocks());
        stringstream s;
        s << "compute_at(" << callee_func.name() << ", Var::gpu_blocks())";
        rF.schedule[PURE_DEF].push_back(s.str());
    }
    return *this;
}

RecFilter& RecFilter::parallel(FuncTag ftag, VarTag vtag, uint vidx) {
    if (vtag & SCAN_DIMENSION ||
            vtag & INNER_SCAN_VAR ||
            vtag & OUTER_SCAN_VAR) {
        cerr << "Cannot create parallel threads from scan variable" << endl;
        assert(false);
    }

    vector<string> func_list = internal_functions(ftag);
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = internal_function(func_list[j]);
        Func            F = Func(rF.func);

        map<int,VarOrRVar> vars = internal_func_vars(rF, vtag, vidx);
        map<int,VarOrRVar>::iterator vit;

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            VarOrRVar v = vit->second;
            if (def==PURE_DEF) {
                F.parallel(v);
                stringstream s;
                s << "parallel(Var(\"" << v.name() << "\"))";
                rF.schedule[PURE_DEF].push_back(s.str());
            } else {
                F.update(def).parallel(v);
                stringstream s;
                s << "parallel(Var(\"" << v.name() << "\"))";
                rF.schedule[def].push_back(s.str());
            }
        }
    }
    return *this;
}

RecFilter& RecFilter::parallel(FuncTag ftag, VarTag vtag, Expr task_size, uint vidx) {
    if (vtag & SCAN_DIMENSION ||
            vtag & INNER_SCAN_VAR ||
            vtag & OUTER_SCAN_VAR) {
        cerr << "Cannot create parallel threads from scan variable" << endl;
        assert(false);
    }

    vector<string> func_list = internal_functions(ftag);
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = internal_function(func_list[j]);
        Func            F = Func(rF.func);

        map<int,VarOrRVar> vars = internal_func_vars(rF, vtag, vidx);
        map<int,VarOrRVar>::iterator vit;

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            VarOrRVar v = vit->second;
            if (def==PURE_DEF) {
                F.parallel(v, task_size);
                stringstream s;
                s << "parallel(Var(\"" << v.name() << "\")," << task_size << ")";
                rF.schedule[def].push_back(s.str());
            } else {
                F.update(def).parallel(v, task_size);
                stringstream s;
                s << "parallel(Var(\"" << v.name() << "\")," << task_size << ")";
                rF.schedule[def].push_back(s.str());
            }
        }
    }
    return *this;
}

RecFilter& RecFilter::unroll(FuncTag ftag, VarTag vtag, uint vidx) {
    vector<string> func_list = internal_functions(ftag);
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = internal_function(func_list[j]);
        Func            F = Func(rF.func);

        map<int,VarOrRVar> vars = internal_func_vars(rF, vtag, vidx);
        map<int,VarOrRVar>::iterator vit;

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            VarOrRVar v = vit->second;
            if (def==PURE_DEF) {
                F.unroll(v);
                stringstream s;
                s << "unroll(Var(\"" << v.name() << "\"))";
                rF.schedule[def].push_back(s.str());
            } else {
                F.update(def).unroll(v);
                stringstream s;
                s << "unroll(Var(\"" << v.name() << "\"))";
                rF.schedule[def].push_back(s.str());
            }
        }
    }
    return *this;
}

RecFilter& RecFilter::unroll(FuncTag ftag, VarTag vtag, uint factor, uint vidx) {
    vector<string> func_list = internal_functions(ftag);
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = internal_function(func_list[j]);
        Func            F = Func(rF.func);

        map<int,VarOrRVar> vars = internal_func_vars(rF, vtag, vidx);
        map<int,VarOrRVar>::iterator vit;

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            VarOrRVar v = vit->second;
            if (def==PURE_DEF) {
                F.unroll(v, factor);
                stringstream s;
                s << "unroll(Var(\"" << v.name() << "\")," << factor << ")";
                rF.schedule[def].push_back(s.str());
            } else {
                F.update(def).unroll(v, factor);
                stringstream s;
                s << "unroll(Var(\"" << v.name() << "\")," << factor << ")";
                rF.schedule[def].push_back(s.str());
            }
        }
    }
    return *this;
}

RecFilter& RecFilter::vectorize(FuncTag ftag, VarTag vtag, uint vidx) {
    vector<string> func_list = internal_functions(ftag);
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = internal_function(func_list[j]);
        Func            F = Func(rF.func);

        map<int,VarOrRVar> vars = internal_func_vars(rF, vtag, vidx);
        map<int,VarOrRVar>::iterator vit;

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            VarOrRVar v = vit->second;
            if (def==PURE_DEF) {
                F.vectorize(v);
                stringstream s;
                s << "vectorize(Var(\"" << v.name() << "\"))";
                rF.schedule[def].push_back(s.str());
            } else {
                F.update(def).vectorize(v);
                stringstream s;
                s << "vectorize(Var(\"" << v.name() << "\"))";
                rF.schedule[def].push_back(s.str());
            }
        }
    }
    return *this;
}

RecFilter& RecFilter::vectorize(FuncTag ftag, VarTag vtag, uint factor, uint vidx) {
    vector<string> func_list = internal_functions(ftag);
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = internal_function(func_list[j]);
        Func            F = Func(rF.func);

        map<int,VarOrRVar> vars = internal_func_vars(rF, vtag, vidx);
        map<int,VarOrRVar>::iterator vit;

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            VarOrRVar v = vit->second;
            if (def==PURE_DEF) {
                F.vectorize(v, factor);
                stringstream s;
                s << "vectorize(Var(\"" << v.name() << "\")," << factor << ")";
                rF.schedule[def].push_back(s.str());
            } else {
                F.update(def).vectorize(v, factor);
                stringstream s;
                s << "vectorize(Var(\"" << v.name() << "\")," << factor << ")";
                rF.schedule[def].push_back(s.str());
            }
        }
    }
    return *this;
}

RecFilter& RecFilter::bound(Var v, Halide::Expr min, Halide::Expr extent) {
    vector<Func> func_list = funcs();
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = internal_function(func_list[j].name());
        Func            F = Func(rF.func);

        for (int i=0; i<F.args().size(); i++) {
            if (v.name() == F.args()[i].name()) {
                F.bound(v, min, extent);
                stringstream s;
                s << "bound(Var(\"" << v.name() << "\")," << min << "," << extent << ")";
                rF.schedule[PURE_DEF].push_back(s.str());
            }
        }
    }
    return *this;
}

RecFilter& RecFilter::bound(VarTag vtag, Expr min, Expr extent, uint vidx) {
    vector<Func> func_list = funcs();
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = internal_function(func_list[j].name());
        Func            F = Func(rF.func);

        map<int,VarOrRVar> vars = internal_func_vars(rF, vtag, vidx);
        map<int,VarOrRVar>::iterator vit;

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            VarOrRVar v = vit->second;
            if (def==PURE_DEF) {
                F.bound(Var(v.name()), min, extent);
                stringstream s;
                s << "bound(Var(\"" << v.name() << "\")," << min << "," << extent << ")";
                rF.schedule[def].push_back(s.str());
            }
        }
    }
    return *this;
}

RecFilter& RecFilter::gpu_threads(FuncTag ftag, VarTag vtag, uint t1) {
    vector<uint> task_size;
    task_size.push_back(t1);
    return gpu_threads(ftag, vtag, task_size);
}

RecFilter& RecFilter::gpu_threads(FuncTag ftag, VarTag vtag, uint t1, uint t2) {
    vector<uint> task_size;
    task_size.push_back(t1);
    task_size.push_back(t2);
    return gpu_threads(ftag, vtag, task_size);
}

RecFilter& RecFilter::gpu_threads(FuncTag ftag, VarTag vtag, uint t1, uint t2, uint t3) {
    vector<uint> task_size;
    task_size.push_back(t1);
    task_size.push_back(t2);
    task_size.push_back(t3);
    return gpu_threads(ftag, vtag, task_size);
}

RecFilter& RecFilter::gpu_threads(FuncTag ftag, VarTag vtag, vector<uint> task_size) {
    if (vtag & SCAN_DIMENSION || vtag & INNER_SCAN_VAR || vtag & OUTER_SCAN_VAR) {
        cerr << "Cannot map a scan variable to parallel threads" << endl;
        assert(false);
    }

    vector<string> func_list = internal_functions(ftag);
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = internal_function(func_list[j]);
        Func            F = Func(rF.func);

        map<int,VarOrRVar>::iterator vit;

        map< int,vector<VarOrRVar> > parallel_vars;

        for (int i=0; i<task_size.size(); i++) {
            int vidx  = i;
            int tsize = task_size[i];

            map<int,int> next_gpu_thread;

            map<int,VarOrRVar> vars = internal_func_vars(rF, vtag, vidx);
            for (vit=vars.begin(); vit!=vars.end(); vit++) {
                int def = vit->first;
                VarOrRVar v = vit->second.var;
                stringstream s;

                int thread_id_x = 0;
                if (next_gpu_thread.find(def) != next_gpu_thread.end()) {
                    thread_id_x = next_gpu_thread[def];
                    next_gpu_thread[def] = thread_id_x+1;
                }
                next_gpu_thread[def] = thread_id_x+1;

                if (thread_id_x<0 || thread_id_x>2) {
                    cerr << "Cannot map more than three vars to GPU threads" << endl;
                    assert(false);
                }

                if (tsize > 1) {
                    Var t(v.name()+".1");

                    s << "split(Var(\"" << v.name() << "\"), Var(\"" << v.name() << "\"), Var(\"" << t.name() << "\"), " << tsize << ").";
                    s << "reorder(Var(\""  << t.name() << "\"), Var(\"" << v.name() << "\"))";

                    if (def==PURE_DEF) {
                        F.split(v,v,t,tsize).reorder(t,v);
                        rF.pure_var_category.insert(make_pair(t.name(), vtag | SCHEDULE_INNER));
                    } else {
                        F.update(def).split(v,v,t,tsize).reorder(t,v);
                        rF.update_var_category[def].insert(make_pair(t.name(), vtag | SCHEDULE_INNER));
                    }
                    rF.schedule[def].push_back(s.str());
                }

                s << std::flush;
                s << "parallel(Var(\"" << v.name() << "\")).";
                s << "rename(Var(\""   << v.name() << "\"), Var(\"" << GPU_THREAD[thread_id_x].name() << "\"))";
                if (def==PURE_DEF) {
                    F.parallel(v).rename(v, GPU_THREAD[thread_id_x]);
                } else {
                    F.update(def).parallel(v).rename(v, GPU_THREAD[thread_id_x]);
                }

                rF.schedule[def].push_back(s.str());
            }
        }
    }
    return *this;
}

RecFilter& RecFilter::inner_split(FuncTag ftag, VarTag vtag, Expr factor, uint vidx) {
    return split(ftag, vtag, factor, vidx, true);
}

RecFilter& RecFilter::outer_split(FuncTag ftag, VarTag vtag, Expr factor, uint vidx) {
    return split(ftag, vtag, factor, vidx, false);
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

RecFilter& RecFilter::gpu_tile(FuncTag ftag, VarTag vtag_x, uint x) {
    return gpu_tile(ftag, Internal::vec(make_pair(vtag_x,x)));
}

RecFilter& RecFilter::gpu_tile(FuncTag ftag, VarTag vtag_x, VarTag vtag_y, uint x, uint y) {
    return gpu_tile(ftag, Internal::vec(make_pair(vtag_x,x), make_pair(vtag_y,y)));
}

RecFilter& RecFilter::gpu_tile(FuncTag ftag, VarTag vtag_x, VarTag vtag_y, VarTag vtag_z, uint x, uint y, uint z) {
    return gpu_tile(ftag, Internal::vec(make_pair(vtag_x,x), make_pair(vtag_y,y), make_pair(vtag_z,z)));
}

RecFilter& RecFilter::split(FuncTag ftag, VarTag vtag, Expr factor, uint vidx, bool do_reorder) {
    if (vtag & SCHEDULE_INNER) {
        cerr << "Cannot split a variable which was previously split by scheduling ops" << endl;
        assert(false);
    }

    vector<string> func_list = internal_functions(ftag);
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = internal_function(func_list[j]);
        Func            F = Func(rF.func);

        map<int,VarOrRVar> vars = internal_func_vars(rF, vtag, vidx);
        map<int,VarOrRVar>::iterator vit;

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            VarOrRVar v = vit->second.var;
            stringstream s;

            Var t(v.name()+".1");

            s << "split(Var(\"" << v.name() << "\"), Var(\"" << v.name()
                << "\"), Var(\"" << t.name() << "\"), " << factor << ")";
            if (do_reorder) {
                s << ".reorder(Var(\"" << t.name() << "\"), Var(\"" << v.name() << "\"))";
            }

            if (def==PURE_DEF) {
                F.split(v,v,t,factor);
                if (do_reorder) {
                    F.reorder(t,v);
                }
                rF.pure_var_category.insert(make_pair(t.name(), vtag | SCHEDULE_INNER));
            } else {
                F.update(def).split(v,v,t,factor);
                if (do_reorder) {
                    F.update(def).reorder(t,v);
                }
                rF.update_var_category[def].insert(make_pair(t.name(), vtag | SCHEDULE_INNER));
            }
            rF.schedule[def].push_back(s.str());
        }
    }
    return *this;
}

RecFilter& RecFilter::reorder(FuncTag ftag, vector<VarTag> vtag) {
    vector<string> func_list = internal_functions(ftag);
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = internal_function(func_list[j]);
        Func            F = Func(rF.func);

        map< int,vector<VarOrRVar> > vars;
        map< int,vector<VarOrRVar> >::iterator vit;

        for (int i=0; i<vtag.size(); i++) {
            map< int,vector<VarOrRVar> > vars_x = internal_func_vars(rF, vtag[i]);
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

RecFilter& RecFilter::reorder_storage(FuncTag ftag, vector<VarTag> vtag) {
    vector<string> func_list = internal_functions(ftag);
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = internal_function(func_list[j]);
        Func            F = Func(rF.func);

        // no need to reorder storage if the function is not compute
        if (!rF.func.schedule().compute_level().is_root()) {
            continue;
        }

        map< int,vector<VarOrRVar> > vars;
        map< int,vector<VarOrRVar> >::iterator vit;

        for (int i=0; i<vtag.size(); i++) {
            map< int,vector<VarOrRVar> > vars_x = internal_func_vars(rF, vtag[i]);
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

RecFilter& RecFilter::gpu_blocks(FuncTag ftag, VarTag vtag) {
    if (vtag & SCAN_DIMENSION || vtag & INNER_SCAN_VAR || vtag & OUTER_SCAN_VAR) {
        cerr << "Cannot create parallel GPU blocks from scan variable" << endl;
        assert(false);
    }

    vector<string> func_list = internal_functions(ftag);
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = internal_function(func_list[j]);
        Func            F = Func(rF.func);

        // do not apply gpu_blocks to functions which are not compute_root
        if (!rF.func.schedule().compute_level().is_root()) {
            continue;
        }

        map< int,vector<VarOrRVar> > vars = internal_func_vars(rF, vtag);
        map< int,vector<VarOrRVar> >::iterator vit;

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            vector<VarOrRVar> var_list = vit->second;
            stringstream s;

            if (var_list.size()>0 && var_list.size()<=3) {
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

RecFilter& RecFilter::gpu_tile(FuncTag ftag, vector< pair<VarTag,uint> > vtag) {
    if (vtag.empty() || vtag.size()>3) {
        cerr << "GPU thread/block grid dimensions must be between 1 and 3" << endl;
        assert(false);
    }

    for (int i=0; i<vtag.size(); i++) {
        if (vtag[i].first & SCAN_DIMENSION ||
                vtag[i].first & INNER_SCAN_VAR ||
                vtag[i].first & OUTER_SCAN_VAR) {
            cerr << "Cannot create parallel GPU tiles from a scan variable" << endl;
            assert(false);
        }

        for (int j=i+1; j<vtag.size(); j++) {
            if (vtag[i].first == vtag[j].first) {
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

        for (int i=0; i<vtag.size(); i++) {
            map<int, vector<VarOrRVar> > vars_x = internal_func_vars(rF, vtag[i].first);
            map<int, vector<VarOrRVar> >::iterator vit;
            for (vit=vars_x.begin(); vit!=vars_x.end(); vit++) {
                for (int k=0; k<vit->second.size(); k++) {
                    vars[vit->first].push_back(make_pair(vit->second[k], vtag[i].second));
                }
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
        if (vit->second == vtag) {
            var_list[PURE_DEF].push_back(Var(vit->first));
        }
    }
    for (int i=0; i<f.update_var_category.size(); i++) {
        for (vit=f.update_var_category[i].begin(); vit!=f.update_var_category[i].end(); vit++) {
            if (vit->second == vtag) {
                var_list[i].push_back(Var(vit->first));
            }
        }
    }
    return var_list;
}

map<int,VarOrRVar> RecFilter::internal_func_vars(RecFilterFunc f, VarTag vtag, uint vidx) {
    map< int,vector<VarOrRVar> > var_list;
    map<string,VarTag>::iterator vit;
    for (vit = f.pure_var_category.begin(); vit!=f.pure_var_category.end(); vit++) {
        if (vit->second == vtag) {
            var_list[PURE_DEF].push_back(Var(vit->first));
        }
    }
    for (int i=0; i<f.update_var_category.size(); i++) {
        for (vit=f.update_var_category[i].begin(); vit!=f.update_var_category[i].end(); vit++) {
            if (vit->second == vtag) {
                var_list[i].push_back(Var(vit->first));
            }
        }
    }

    map<int,VarOrRVar> vlist;
    map<int,vector<VarOrRVar> >::iterator var_list_it;
    for (var_list_it=var_list.begin(); var_list_it!=var_list.end(); var_list_it++) {
        int def             = var_list_it->first;
        vector<VarOrRVar> v = var_list_it->second;
        if (v.size()>vidx) {
            vlist.insert(make_pair(def,v[vidx]));
        }
    }
    return vlist;
}
