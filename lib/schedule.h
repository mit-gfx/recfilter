#ifndef _SCHEDULE_H_
#define _SCHEDULE_H_

/** Scheduling interface for functions generated for the recursive filter */
class RecFilterSchedule {
public:
    typedef enum {
        NO_TAG                     = 0x00;
        INTRA_TILE_nD_SCAN         = 0x01,
        INTER_TILE_1D_SCAN         = 0x02,
        INTER_TILE_2D_SCAN         = 0x04,
        INTER_TILE_3D_SCAN         = 0x08,
        INTER_TILE_4D_SCAN         = 0x01,
        INTRA_TILE_nD_SCAN_WRAPPER = 0x00,
        INTER_TILE_1D_SCAN_WRAPPER = 0x01,
        INTER_TILE_2D_SCAN_WRAPPER = 0x02,
        INTER_TILE_3D_SCAN_WRAPPER = 0x04,
        INTER_TILE_4D_SCAN_WRAPPER = 0x08
    } FuncCategory;

    typedef enum {
        NO_TAG         = 0x00;
        INNER_PURE_VAR = 0x01,
        INNER_SCAN_VAR = 0x02,
        OUTER_PURE_VAR = 0x04,
        OUTER_SCAN_VAR = 0x08,
        OUTER_SCAN_VAR = 0x04
    } VarCategory;

private:
    Func                          func;
    FuncCategory                  func_category;
    map<string, VarCategory>      pure_var_category;
    map<map<string,VarCategory> > update_var_category;
};

// ----------------------------------------------------------------------------


#endif // _SCHEDULE_H_
