#include "recfilter.h"
#include "recfilter_internals.h"

using std::cerr;
using std::endl;
using std::vector;
using std::map;
using std::make_pair;

using namespace Halide;

vector<RecFilterFunc> RecFilter::internal_functions(FuncTag ftag) {
    vector<RecFilterFunc> func_list;
    map<string,RecFilterFunc>::iterator f_it  = contents.ptr->func.begin();
    map<string,RecFilterFunc>::iterator f_end = contents.ptr->func.end();
    while (f_it != f_end) {
        if (f_it->second.func_category & ftag) {
            func_list.push_back(f_it->second);
        }
        f_it++;
    }
    if (func_list.empty()) {
        cerr << "No recursive filter has the given scheduling tag " << ftag << endl;
        assert(false);
    }
    return func_list;
}

map< int,vector<Var> > RecFilter::internal_function_vars(RecFilterFunc f, VarTag vtag) {
    map< int,vector<Var> > var_list;

    map<string,VarTag>::iterator vit;

    for (vit = f.pure_var_category.begin(); vit!=f.pure_var_category.end(); vit++) {
        if (vit->second & vtag) {
            var_list[-1].push_back(Var(vit->first)); // vars in pure defs are mapped to -1
        }
    }

    for (int i=0; i<f.update_var_category.size(); i++) {
        for (vit=f.update_var_category[i].begin(); vit!=f.update_var_category[i].end(); vit++) {
            if (vit->second & vtag) {
                var_list[i].push_back(Var(vit->first)); // vars in update defs are mapped to update def number
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

RecFilter& RecFilter::compute_root(FuncTag ftag) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::compute_at  (FuncTag ftag) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::split(FuncTag ftag, VarTag old, VarTag outer, VarTag inner, Expr factor) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::fuse(FuncTag ftag, VarTag inner, VarTag outer, VarTag fused) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::parallel(FuncTag ftag, VarTag var) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::parallel(FuncTag ftag, VarTag var, Expr task_size) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::vectorize(FuncTag ftag, VarTag var) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::unroll(FuncTag ftag, VarTag var) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::vectorize(FuncTag ftag, VarTag var, int factor) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::unroll(FuncTag ftag, VarTag var, int factor) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::bound(FuncTag ftag, VarTag var, Expr min, Expr extent) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::tile(FuncTag ftag, VarTag x, VarTag y, VarTag xo, VarTag yo, VarTag xi, VarTag yi, Expr xfactor, Expr yfactor) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::tile(FuncTag ftag, VarTag x, VarTag y, VarTag xi, VarTag yi, Expr xfactor, Expr yfactor) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::reorder(FuncTag ftag, VarTag x, VarTag y) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::reorder(FuncTag ftag, VarTag x, VarTag y, VarTag z) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::reorder(FuncTag ftag, VarTag x, VarTag y, VarTag z, VarTag w) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::reorder(FuncTag ftag, VarTag x, VarTag y, VarTag z, VarTag w, VarTag t) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::reorder(FuncTag ftag, VarTag x, VarTag y, VarTag z, VarTag w, VarTag t1, VarTag t2) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::reorder(FuncTag ftag, VarTag x, VarTag y, VarTag z, VarTag w, VarTag t1, VarTag t2, VarTag t3) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::reorder(FuncTag ftag, VarTag x, VarTag y, VarTag z, VarTag w, VarTag t1, VarTag t2, VarTag t3, VarTag t4) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::reorder(FuncTag ftag, VarTag x, VarTag y, VarTag z, VarTag w, VarTag t1, VarTag t2, VarTag t3, VarTag t4, VarTag t5) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::reorder(FuncTag ftag, VarTag x, VarTag y, VarTag z, VarTag w, VarTag t1, VarTag t2, VarTag t3, VarTag t4, VarTag t5, VarTag t6) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::gpu_threads(FuncTag ftag, VarTag thread_x) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::gpu_threads(FuncTag ftag, VarTag thread_x, VarTag thread_y) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::gpu_threads(FuncTag ftag, VarTag thread_x, VarTag thread_y, VarTag thread_z) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::gpu_blocks(FuncTag ftag, VarTag block_x) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::gpu_blocks(FuncTag ftag, VarTag block_x, VarTag block_y) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::gpu_blocks(FuncTag ftag, VarTag block_x, VarTag block_y, VarTag block_z) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::gpu(FuncTag ftag, VarTag block_x, VarTag thread_x) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::gpu(FuncTag ftag, VarTag block_x, VarTag block_y, VarTag thread_x, VarTag thread_y) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::gpu(FuncTag ftag, VarTag block_x, VarTag block_y, VarTag block_z, VarTag thread_x, VarTag thread_y, VarTag thread_z) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::gpu_tile(FuncTag ftag, VarTag x, int x_size) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::gpu_tile(FuncTag ftag, VarTag x, VarTag y, int x_size, int y_size) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::gpu_tile(FuncTag ftag, VarTag x, VarTag y, VarTag z, int x_size, int y_size, int z_size) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}


RecFilter& RecFilter::reorder_storage(FuncTag ftag, VarTag x, VarTag y) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::reorder_storage(FuncTag ftag, VarTag x, VarTag y, VarTag z) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::reorder_storage(FuncTag ftag, VarTag x, VarTag y, VarTag z, VarTag w) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}

RecFilter& RecFilter::reorder_storage(FuncTag ftag, VarTag x, VarTag y, VarTag z, VarTag w, VarTag t) {
    vector<RecFilterFunc> func_list = internal_functions(ftag);
    for (int i=0; i<func_list.size(); i++) {
        RecFilterFunc rF = func_list[i];
        Func           F = Func(rF.func);
    }
    return *this;
}
