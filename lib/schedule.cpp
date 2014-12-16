#include "recfilter.h"
#include "recfilter_internals.h"
#include "modifiers.h"

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

RecFilterSchedule::RecFilterSchedule(RecFilter& r, vector<string> fl) :
    recfilter(r), func_list(fl) {}


// -----------------------------------------------------------------------------

map< int,vector<VarOrRVar> > RecFilterSchedule::var_list_by_tag(RecFilterFunc f, VarTag vtag) {
    map< int,vector<VarOrRVar> > var_list;
    map<string,VarTag>::iterator vit;
    for (vit = f.pure_var_category.begin(); vit!=f.pure_var_category.end(); vit++) {
        if (vit->second.check(vtag)) {
            var_list[PURE_DEF].push_back(Var(vit->first));
        }
    }
    for (int i=0; i<f.update_var_category.size(); i++) {
        for (vit=f.update_var_category[i].begin(); vit!=f.update_var_category[i].end(); vit++) {
            if (vit->second.check(vtag)) {
                var_list[i].push_back(Var(vit->first));
            }
        }
    }
    return var_list;
}

map<int,VarOrRVar> RecFilterSchedule::var_by_tag(RecFilterFunc f, VarTag vtag) {
    map<int,VarOrRVar> var_list;
    map<string,VarTag>::iterator vit;
    for (vit = f.pure_var_category.begin(); vit!=f.pure_var_category.end(); vit++) {
        if (vit->second == vtag) {
            if (var_list.find(PURE_DEF)==var_list.end()) {
                var_list.insert(make_pair(PURE_DEF, Var(vit->first)));
            } else {
                cerr << "Found multiple vars with the scheduling tag " << vtag << endl;
                assert(false);
            }
        }
    }
    for (int i=0; i<f.update_var_category.size(); i++) {
        for (vit=f.update_var_category[i].begin(); vit!=f.update_var_category[i].end(); vit++) {
            if (vit->second == vtag) {
                if (var_list.find(i)==var_list.end()) {
                    var_list.insert(make_pair(i, Var(vit->first)));
                } else {
                    cerr << "Found multiple vars with the scheduling tag " << vtag << endl;
                    assert(false);
                }
            }
        }
    }

    return var_list;
}

// -----------------------------------------------------------------------------

RecFilterSchedule& RecFilterSchedule::compute_globally() {
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = recfilter.internal_function(func_list[j]);
        Func            F = Func(rF.func);

        // remove the initializations of all scans which are scheduled as
        // compute_root to avoid extra kernel execution for initializing
        // the output buffer
        if (F.has_update_definition()) {
            move_pure_def_to_update(F.function());
        }

        F.compute_root();
        rF.pure_schedule.push_back("compute_root()");
    }
    return *this;
}

RecFilterSchedule& RecFilterSchedule::compute_locally() {
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = recfilter.internal_function(func_list[j]);
        Func            F = Func(rF.func);

        // ignore if this is a reindexing function
        if (rF.func_category==REINDEX) {
            continue;
        }

        // find the reindexing functions associated with this function
        // function that calls this function must be computed in the global mem
        Func callee_func;
        for (int i=0; i<func_list.size(); i++) {
            RecFilterFunc& rf = recfilter.internal_function(func_list[i]);
            if (rf.func_category == REINDEX && rf.callee_func==F.name()) {
                if (callee_func.defined()) {
                    cerr << F.name() << " cannot be computed in shared mem "
                        << "because it is called by multiple functions" << endl;
                    assert(false);
                }
                callee_func = Func(rf.func);
                callee_func.compute_root();
                rf.pure_schedule.push_back("compute_root()");
            }
        }

        if (!callee_func.defined()) {
            cerr << F.name() << " cannot be computed in shared mem "
                << "because it is not called by any function" << endl;
            assert(false);
        }

        // functions called in this function should be computed at same level
        for (int i=0; i<func_list.size(); i++) {
            RecFilterFunc& rf = recfilter.internal_function(func_list[i]);
            if (rf.func_category==REINDEX && rf.caller_func==F.name()) {
                Func(rf.func).compute_at(callee_func, Var::gpu_blocks());
                stringstream s;
                s << "compute_at(" << callee_func.name() << ", Var::gpu_blocks())";
                rf.pure_schedule.push_back(s.str());
            }
        }

        F.compute_at(callee_func, Var::gpu_blocks());
        stringstream s;
        s << "compute_at(" << callee_func.name() << ", Var::gpu_blocks())";
        rF.pure_schedule.push_back(s.str());
    }
    return *this;
}

RecFilterSchedule& RecFilterSchedule::parallel(VarTag vtag) {
    if (vtag.check(SCAN)) {
        cerr << "Cannot create parallel threads from scan variable" << endl;
        assert(false);
    }

    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = recfilter.internal_function(func_list[j]);
        Func            F = Func(rF.func);

        map<int,VarOrRVar> vars = var_by_tag(rF, vtag);
        map<int,VarOrRVar>::iterator vit;

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            VarOrRVar v = vit->second;

            // no need to add scheduling to pure def if the function has update defs
            // pure def is left undefined by construction
            if (def==PURE_DEF) {
                if (!F.has_update_definition()) {
                    F.parallel(v);
                    rF.pure_schedule.push_back("parallel(Var(\"" + v.name() + "\"))");
                }
            } else {
                F.update(def).parallel(v);
                stringstream s;
                rF.update_schedule[def].push_back("parallel(Var(\"" + v.name() + "\"))");
            }
        }
    }
    return *this;
}

RecFilterSchedule& RecFilterSchedule::unroll(VarTag vtag) {
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = recfilter.internal_function(func_list[j]);
        Func            F = Func(rF.func);

        map<int, vector<VarOrRVar> > vars = var_list_by_tag(rF, vtag);
        map<int, vector<VarOrRVar> >::iterator vit;

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            for (int i=0; i<vit->second.size(); i++) {
                VarOrRVar v = vit->second[i];

                // no need to add scheduling to pure def if the function has update defs
                // pure def is left undefined by construction
                if (def==PURE_DEF) {
                    if (!F.has_update_definition()) {
                        F.unroll(v);
                        rF.pure_schedule.push_back("unroll(Var(\"" + v.name() + "\"))");
                    }
                } else {
                    F.update(def).unroll(v);
                    rF.update_schedule[def].push_back("unroll(Var(\"" + v.name() + "\"))");
                }
            }
        }
    }
    return *this;
}

RecFilterSchedule& RecFilterSchedule::vectorize(VarTag vtag) {
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = recfilter.internal_function(func_list[j]);
        Func            F = Func(rF.func);

        map<int,VarOrRVar> vars = var_by_tag(rF, vtag);
        map<int,VarOrRVar>::iterator vit;

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            VarOrRVar v = vit->second;

            // no need to add scheduling to pure def if the function has update defs
            // pure def is left undefined by construction
            if (def==PURE_DEF) {
                if (!F.has_update_definition()) {
                    F.vectorize(v);
                    rF.pure_schedule.push_back("vectorize(Var(\"" + v.name() + "\"))");
                }
            } else {
                F.update(def).vectorize(v);
                stringstream s;
                s << "vectorize(Var(\"" << v.name() << "\"))";
                rF.update_schedule[def].push_back("vectorize(Var(\"" + v.name() + "\"))");
            }
        }
    }
    return *this;
}

RecFilterSchedule& RecFilterSchedule::gpu_blocks(VarTag v1) {
    return gpu_blocks(v1, VarTag(INVALID), VarTag(INVALID));
}

RecFilterSchedule& RecFilterSchedule::gpu_blocks(VarTag v1, VarTag v2) {
    return gpu_blocks(v1, v2, VarTag(INVALID));
}

RecFilterSchedule& RecFilterSchedule::gpu_blocks(VarTag v1, VarTag v2, VarTag v3) {
    if (v1.check(SCAN) || v2.check(SCAN) || v3.check(SCAN)) {
        cerr << "Cannot map a scan variable to parallel GPU blocks" << endl;
        assert(false);
    }

    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = recfilter.internal_function(func_list[j]);
        Func            F = Func(rF.func);

        // no need to set gpu_blocks if the function is not compute root
        if (!rF.func.schedule().compute_level().is_root()) {
            continue;
        }

        map<int,VarOrRVar>::iterator vit;
        map<int,vector<VarOrRVar> > parallel_vars;
        map<int,int> next_gpu_block;

        for (int i=0; i<3; i++) {
            VarTag vtag;

            switch (i) {
                case 0: vtag = v1; break;
                case 1: vtag = v2; break;
                case 2: vtag = v3; break;
                default: break;
            }

            if (vtag == INVALID) {
                continue;
            }

            map<int,VarOrRVar> vars = var_by_tag(rF, vtag);
            for (vit=vars.begin(); vit!=vars.end(); vit++) {
                int def = vit->first;
                VarOrRVar v = vit->second.var;
                stringstream s;

                int block_id_x = 0;
                if (next_gpu_block.find(def) != next_gpu_block.end()) {
                    block_id_x = next_gpu_block[def];
                    next_gpu_block[def] = block_id_x+1;
                }
                next_gpu_block[def] = block_id_x+1;

                if (block_id_x<0 || block_id_x>2) {
                    cerr << "Cannot map more than three vars to GPU threads" << endl;
                    assert(false);
                }

                s << "parallel(Var(\"" << v.name() << "\")).";
                s << "rename(Var(\""   << v.name() << "\"), Var(\"" << GPU_BLOCK[block_id_x].name() << "\"))";

                // no need to add scheduling to pure def if the function has update defs
                // pure def is left undefined by construction
                if (def==PURE_DEF) {
                    if (!F.has_update_definition()) {
                        F.parallel(v).rename(v, GPU_BLOCK[block_id_x]);
                        rF.pure_schedule.push_back(s.str());
                    }
                } else {
                    F.update(def).parallel(v).rename(v, GPU_BLOCK[block_id_x]);
                    rF.update_schedule[def].push_back(s.str());
                }
            }
        }
    }
    return *this;
}

RecFilterSchedule& RecFilterSchedule::gpu_threads(VarTag v1) {
    return gpu_threads(v1,VarTag(INVALID),VarTag(INVALID));
}

RecFilterSchedule& RecFilterSchedule::gpu_threads(VarTag v1, VarTag v2) {
    return gpu_threads(v1,v2,VarTag(INVALID));
}

RecFilterSchedule& RecFilterSchedule::gpu_threads(VarTag v1, VarTag v2, VarTag v3) {
    if (v1.check(SCAN) || v2.check(SCAN) || v3.check(SCAN)) {
        cerr << "Cannot map a scan variable to parallel threads" << endl;
        assert(false);
    }

    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = recfilter.internal_function(func_list[j]);
        Func            F = Func(rF.func);

        map<int,VarOrRVar>::iterator vit;
        map<int,vector<VarOrRVar> > parallel_vars;
        map<int,int> next_gpu_thread;

        for (int i=0; i<3; i++) {
            VarTag vtag;

            switch (i) {
                case 0: vtag = v1; break;
                case 1: vtag = v2; break;
                default:vtag = v3; break;
            }

            if (vtag == INVALID) {
                continue;
            }

            map<int,VarOrRVar> vars = var_by_tag(rF, vtag);
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

                s << "parallel(Var(\"" << v.name() << "\")).";
                s << "rename(Var(\""   << v.name() << "\"), Var(\"" << GPU_THREAD[thread_id_x].name() << "\"))";

                // no need to add scheduling to pure def if the function has update defs
                // pure def is left undefined by construction
                if (def==PURE_DEF) {
                    if (!F.has_update_definition()) {
                        F.parallel(v).rename(v, GPU_THREAD[thread_id_x]);
                        rF.pure_schedule.push_back(s.str());
                    }
                } else {
                    F.update(def).parallel(v).rename(v, GPU_THREAD[thread_id_x]);
                    rF.update_schedule[def].push_back(s.str());
                }
            }
        }
    }
    return *this;
}

RecFilterSchedule& RecFilterSchedule::reorder(VarTag x, VarTag y) {
    return reorder({x,y});
}

RecFilterSchedule& RecFilterSchedule::reorder(VarTag x, VarTag y, VarTag z) {
    return reorder({x,y,z});
}

RecFilterSchedule& RecFilterSchedule::reorder(VarTag x, VarTag y, VarTag z, VarTag w) {
    return reorder({x,y,z,w});
}

RecFilterSchedule& RecFilterSchedule::reorder(VarTag x, VarTag y, VarTag z, VarTag w, VarTag t) {
    return reorder({x,y,z,w,t});
}

RecFilterSchedule& RecFilterSchedule::reorder(VarTag x, VarTag y, VarTag z, VarTag w, VarTag s, VarTag t) {
    return reorder({x,y,z,s,t});
}

RecFilterSchedule& RecFilterSchedule::reorder(VarTag x, VarTag y, VarTag z, VarTag w, VarTag s, VarTag t, VarTag u) {
    return reorder({x,y,z,s,t,u});
}

RecFilterSchedule& RecFilterSchedule::reorder_storage(VarTag x, VarTag y) {
    return reorder_storage({x,y});
}

RecFilterSchedule& RecFilterSchedule::reorder_storage(VarTag x, VarTag y, VarTag z) {
    return reorder_storage({x,y,z});
}

RecFilterSchedule& RecFilterSchedule::reorder_storage(VarTag x, VarTag y, VarTag z, VarTag w) {
    return reorder({x,y,z,w});
}

RecFilterSchedule& RecFilterSchedule::reorder_storage(VarTag x, VarTag y, VarTag z, VarTag w, VarTag t) {
    return reorder_storage({x,y,z,w,t});
}

RecFilterSchedule& RecFilterSchedule::split(VarTag vtag, int factor) {
    if (vtag.check(SPLIT)) {
        cerr << "Cannot split a variable which was created by split scheduling ops" << endl;
        assert(false);
    }

    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = recfilter.internal_function(func_list[j]);
        Func            F = Func(rF.func);

        map<int,VarOrRVar> vars = var_by_tag(rF, vtag);
        map<int,VarOrRVar>::iterator vit;

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            VarOrRVar v = vit->second.var;
            stringstream s;

            Var t(v.name()+".1");

            s << "split(Var(\"" << v.name() << "\"), Var(\"" << v.name()
                << "\"), Var(\"" << t.name() << "\"), " << factor << ")";

            // no need to add scheduling to pure def if the function has update defs
            // pure def is left undefined by construction
            if (def==PURE_DEF) {
                if (!F.has_update_definition()) {
                    F.split(v,v,t,factor);
                    rF.pure_var_category.insert(make_pair(t.name(), vtag|SPLIT));
                    rF.pure_schedule.push_back(s.str());
                }
            } else {
                F.update(def).split(v,v,t,factor);
                rF.update_var_category[def].insert(make_pair(t.name(), vtag|SPLIT));
                rF.update_schedule[def].push_back(s.str());
            }
        }
    }
    return *this;
}

RecFilterSchedule& RecFilterSchedule::reorder(vector<VarTag> vtag) {
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = recfilter.internal_function(func_list[j]);
        Func            F = Func(rF.func);

        map< int,vector<VarOrRVar> > vars;
        map< int,vector<VarOrRVar> >::iterator vit;

        for (int i=0; i<vtag.size(); i++) {
            map< int,vector<VarOrRVar> > vars_x = var_list_by_tag(rF, vtag[i]);
            for (vit=vars_x.begin(); vit!=vars_x.end(); vit++) {
                for (int k=0; k<vit->second.size(); k++) {
                    // check if this var is already in the list
                    bool already_added = false;
                    for (int u=0; !already_added && u<vars[vit->first].size(); u++) {
                        if (vars[vit->first][u].name() == vit->second[k].name()) {
                            already_added = true;
                        }
                    }
                    if (!already_added) {
                        vars[vit->first].push_back(vit->second[k]);
                    }
                }
            }
        }

        for (vit=vars.begin(); vit!=vars.end(); vit++) {
            int def = vit->first;
            vector<VarOrRVar> var_list = vit->second;
            stringstream s;

            if (var_list.size()>1) {                 // at least 2 to reorder
                s << "reorder(Internal::vec(";
                for (int i=0; i<var_list.size(); i++) {
                    if (i>0) {
                        s << ",";
                    }
                    s << "Var(\"" << var_list[i].name() << "\")";
                }
                s << "))";

                // no need to add scheduling to pure def if the function has update defs
                // pure def is left undefined by construction
                if (def==PURE_DEF) {
                    if (!F.has_update_definition()) {
                        F.reorder(var_list);
                        rF.pure_schedule.push_back(s.str());
                    }
                } else {
                    F.update(def).reorder(var_list);
                    rF.update_schedule[def].push_back(s.str());
                }
            } else {
                cerr << "VarTags provided to reorder() resulted in less than 2 variables" << endl;
                assert(false);
            }
        }
    }
    return *this;
}

RecFilterSchedule& RecFilterSchedule::reorder_storage(vector<VarTag> vtag) {
    for (int j=0; j<func_list.size(); j++) {
        RecFilterFunc& rF = recfilter.internal_function(func_list[j]);
        Func            F = Func(rF.func);

        // no need to reorder storage if the function is not compute
        // don't reorder storage of final result
        if (!rF.func.schedule().compute_level().is_root() ||
                rF.func.name()==recfilter.func().name()) {
            continue;
        }

        vector<VarOrRVar> var_list;
        map< int,vector<VarOrRVar> >::iterator vit;

        for (int i=0; i<vtag.size(); i++) {
            map< int,vector<VarOrRVar> > vars_x = var_list_by_tag(rF, vtag[i]);
            for (vit=vars_x.begin(); vit!=vars_x.end(); vit++) {
                if (vit->first == PURE_DEF) {
                    for (int k=0; k<vit->second.size(); k++) {
                        // check if this var is already in the list
                        bool already_added = false;
                        for (int u=0; !already_added && u<var_list.size(); u++) {
                            if (var_list[u].name() == vit->second[k].name()) {
                                already_added = true;
                            }
                        }
                        if (!already_added) {
                            var_list.push_back(vit->second[k]);
                        }
                    }
                }
            }
        }

        stringstream s;

        if (var_list.size()>1) {                 // at least 2 to reorder
            s << "reorder_storage(";
            for (int i=0; i<var_list.size(); i++) {
                if (i>0) {
                    s  << ",";
                }
                s << "Var(\"" << var_list[i].name() << "\")";
            }
            s << ")";

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
            rF.pure_schedule.push_back(s.str());
        } else {
            cerr << "VarTags provided to reorder_storage() resulted in less than 2 variables" << endl;
            assert(false);
        }
    }
    return *this;
}


/// RecFilterSchedule& RecFilterSchedule::gpu_tile(vector< pair<VarTag,int> > vtag) {
///     if (vtag.empty() || vtag.size()>3) {
///         cerr << "GPU thread/block grid dimensions must be between 1 and 3" << endl;
///         assert(false);
///     }
///
///     for (int i=0; i<vtag.size(); i++) {
///         if (vtag[i].first & SCAN) {
///             cerr << "Cannot create parallel GPU tiles from a scan variable" << endl;
///             assert(false);
///         }
///
///         for (int j=i+1; j<vtag.size(); j++) {
///             if (vtag[i].first == vtag[j].first) {
///                 cerr << "Same variable tag mapped to multiple GPU block/thread tiling factors" << endl;
///                 assert(false);
///             }
///         }
///     }
///
///     for (int j=0; j<func_list.size(); j++) {
///         RecFilterFunc& rF = recfilter.internal_function(func_list[j]);
///         Func            F = Func(rF.func);
///
///         map< int,vector<pair<VarOrRVar,int> > > vars;
///
///         for (int i=0; i<vtag.size(); i++) {
///             map<int, vector<VarOrRVar> > vars_x = var_list_by_tag(rF, vtag[i].first);
///             map<int, vector<VarOrRVar> >::iterator vit;
///             for (vit=vars_x.begin(); vit!=vars_x.end(); vit++) {
///                 for (int k=0; k<vit->second.size(); k++) {
///                     vars[vit->first].push_back(make_pair(vit->second[k], vtag[i].second));
///                 }
///             }
///         }
///
///         map< int,vector< pair<VarOrRVar,int> > >::iterator vit;
///         for (vit=vars.begin(); vit!=vars.end(); vit++) {
///             int def = vit->first;
///             vector< pair<VarOrRVar,int> > var_list = vit->second;
///             stringstream s;
///
///             if (var_list.size()) {
///                 s << "gpu_tile(";
///                 for (int i=0; i<var_list.size(); i++) {
///                     if (i>0) {
///                         s << ",";
///                     }
///                     s << "Var(\"" << var_list[i].first.name() << "\")";
///                 }
///                 for (int i=0; i<var_list.size(); i++) {
///                     s << var_list[i].second;
///                 }
///                 s << ")";
///
///                 // only support up to 3 variables
///                 if (def==PURE_DEF) {
///                     switch (var_list.size()) {
///                         case 1: F.gpu_tile(var_list[0].first,
///                                         var_list[0].second); break;
///                         case 2: F.gpu_tile(var_list[0].first,
///                                         var_list[1].first,
///                                         var_list[0].second,
///                                         var_list[1].second); break;
///                         case 3: F.gpu_tile(var_list[0].first,
///                                         var_list[1].first,
///                                         var_list[2].first,
///                                         var_list[0].second,
///                                         var_list[1].second,
///                                         var_list[2].second); break;
///                         default:cerr << "Too many variables in gpu_tile()" << endl; assert(false); break;
///                     }
///                     rF.pure_schedule.push_back(s.str());
///                 } else {
///                     switch (var_list.size()) {
///                         case 1: F.update(def).gpu_tile(var_list[0].first,
///                                         var_list[0].second); break;
///                         case 2: F.update(def).gpu_tile(var_list[0].first,
///                                         var_list[1].first,
///                                         var_list[0].second,
///                                         var_list[1].second); break;
///                         case 3: F.update(def).gpu_tile(var_list[0].first,
///                                         var_list[1].first,
///                                         var_list[2].first,
///                                         var_list[0].second,
///                                         var_list[1].second,
///                                         var_list[2].second); break;
///                         default:cerr << "Too many variables in gpu_tile()" << endl; assert(false); break;
///                     }
///                     rF.update_schedule[def].push_back(s.str());
///                 }
///             }
///         }
///     }
///     return *this;
/// }

