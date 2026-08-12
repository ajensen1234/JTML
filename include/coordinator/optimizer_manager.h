/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*Manages Optimization in a Seperate QT Thread*/

#ifndef OPTIMIZER_MANAGER_H
#define OPTIMIZER_MANAGER_H

/*Custom CUDA Headers*/
#include <gpu_dilated_frame.cuh>
#include <gpu_edge_frame.cuh>
#include <gpu_heatmaps.cuh>
#include <gpu_intensity_frame.cuh>
#include <gpu_metrics.cuh>
#include <gpu_model.cuh>

#include "services/calibration.h"

/*QT Threading*/
#include <qobject.h>
#include <qthread.h>

#include <QModelIndex>

/*Frame and Model and Location Storage*/
#include "compute/frame.h"
#include "services/location_storage.h"
#include "services/model.h"

/*Direct Library*/
#include "domain/data_structures_6D.h"
#include "domain/direct_data_storage.h"

/*Extracted pure DIRECT optimizer (plan U5/U6)*/
#include "domain/direct_optimizer.h"

/*Custom Calibration Struct (Used in CUDA GPU METRICS)*/
#include "services/calibration.h"

/*Optimizer Settings*/
#include "services/optimizer_settings.h"

/*Plan 008 U9 (Cut B): the pure stage-script surface (StageSpec/StageScript,
 * BuildStageScript, DeriveStageCostParams) — Optimize() consumes the script
 * built once per run in Initialize (the container's stage policy is data).*/
#include "coordinator/optimizer_stage_script.h"

/*std::function (the BuildGpuCostAdapter return type)*/
#include <functional>

/*Metric Types*/
#include "domain/metric_enum.h"

/*Cost Function Library*/
#include "domain/sym_trap_functions.h"
#include "compute/CostFunctionManager.h"

using namespace gpu_cost_function;

class OptimizerManager : public QObject {
    Q_OBJECT

public:
    explicit OptimizerManager(QObject* parent = 0);
    /*Sets up Everything for Optimizer and Also Handles CUDA Initialization, Can
     * Fail!*/
    bool Initialize(
        QThread& optimizer_thread,
        Calibration calibration_file,
        std::vector<Frame> camera_A_frame_list,
        std::vector<Frame> camera_B_frame_list,
        unsigned int current_frame_index,
        std::vector<Model> model_list,
        QModelIndexList selected_models,
        unsigned int primary_model_index,
        LocationStorage pose_matrix,
        OptimizerSettings opt_settings,
        jta_cost_function::CostFunctionManager trunk_manager,
        jta_cost_function::CostFunctionManager branch_manager,
        jta_cost_function::CostFunctionManager leaf_manager,
        QString opt_directive,
        QString& error_message,
        int iter_count,
        DirectOptimizer::Options direct_options = DirectOptimizer::Options());
    ~OptimizerManager();

    /* get cost numbers for symmetry plotting */
    double EvaluateCostFunctionAtPoint(Point6D point, int stage);
    void CalculateSymTrap();

signals:
    /*Update Blue Current Optimum*/
    void
    UpdateOptimum(double, double, double, double, double, double, unsigned int);
    /*Finished*/
    void finished();
    /*Finished Optimizing Frame, Send Optimum to MainScreen, The last bool
     * indicates if should move to next frame*/
    void OptimizedFrame(
        double,
        double,
        double,
        double,
        double,
        double,
        bool,
        unsigned int,
        bool,
        QString);
    /*Uh oh There was an Error. The string is the message*/
    void OptimizerError(QString);
    /*Update Display with Speed, Cost Function Calls, Current Minimum*/
    void UpdateDisplay(double, int, double, unsigned int);
    /*Update Dilation Background*/
    void UpdateDilationBackground();

    void CostFuncAtPoint(double);
    void
    onUpdateOrientationSymTrap(double, double, double, double, double, double);
    void onProgressBarUpdate(int);
    void get_iter_count();

public slots:
    /*Optimizer Biplane Single Model*/
    void Optimize();

    /*Emergency Stop*/
    void onStopOptimizer();

private:
    /*Initial Variables and Objects*/
    /*Calibration File*/
    Calibration calibration_;

    /*Optimizer Settings*/
    OptimizerSettings optimizer_settings_;

    /*SYM TRAP SETTINGS*/
    bool sym_trap_call;
    // sym_trap *sym_trap_obj;

    /*Frames*/
    std::vector<Frame> frames_A_;
    /*Camera B Frames*/
    std::vector<Frame> frames_B_;

    /*Models: All Models, Selected Non-Primary Models, and Primary Model*/
    std::vector<Model> all_models_;
    std::vector<Model> selected_non_primary_models_;
    Model primary_model_;
    /*Indices of All Selected Models*/
    QModelIndexList selected_model_list_;
    /*Index of Primary Model*/
    unsigned int primary_model_index_;

    /*Cost Function Managers For Each Stage*/
    jta_cost_function::CostFunctionManager trunk_manager_;
    jta_cost_function::CostFunctionManager branch_manager_;
    jta_cost_function::CostFunctionManager leaf_manager_;

    /*Should we progess to next frame?*/
    bool progress_next_frame_;
    /*Should we initialize with previous frame's best guess?*/
    bool init_prev_frame_;
    /*Index For Starting Frame in Optimization*/
    unsigned int start_frame_index_;
    unsigned int end_frame_index_;
    int iter_count;

    std::vector<int> img_indices_;

    QString optimization_directive_;

    void
    create_image_indices(std::vector<int>& img_indices, int start, int end);

    /*Error Check*/
    cudaError_t cuda_status_;

    /*Correctly Initialized*/
    bool succesfull_initialization_;

    /*Dilation Values Based on Parameter Names (Dilation or DILATION or
     * dilation) that are ints*/
    int trunk_dilation_val_;
    int branch_dilation_val_;
    int leaf_dilation_val_;

    /*Black Silhouette Values Based on Parameter Names (Black_Silhouette or
     * Dark_Silhouette or BLACK_SILHOUETTE or DARK_SILHOUETTE or
     * black_silhouette or dark_silhouette)*/
    bool trunk_dark_silhouette_val_;
    bool branch_dark_silhouette_val_;
    bool leaf_dark_silhouette_val_;

    /*GPU Metrics Class*/
    GPUMetrics* gpu_metrics_;

    /*CUDA Cost Function Objects (Vector of GPU Models and vector of GPU Frames
    - note Dilated and Intensity must have own vector for each stage because
    their values could change with the stage from a black silhouette bool or a
    dilation int)*/
    /*Camera A (Monoplane or Biplane)*/
    std::vector<GPUIntensityFrame*> gpu_intensity_frames_trunk_A_;
    std::vector<GPUIntensityFrame*> gpu_intensity_frames_branch_A_;
    std::vector<GPUIntensityFrame*> gpu_intensity_frames_leaf_A_;
    std::vector<GPUEdgeFrame*> gpu_edge_frames_A_;
    std::vector<GPUDilatedFrame*> gpu_dilated_frames_trunk_A_;
    std::vector<GPUDilatedFrame*> gpu_dilated_frames_branch_A_;
    std::vector<GPUDilatedFrame*> gpu_dilated_frames_leaf_A_;
    std::vector<GPUFrame*> gpu_distance_maps_;
    std::vector<GPUHeatmap*> gpu_heatmaps_;
    /*Camera B (Biplane only)*/
    std::vector<GPUIntensityFrame*> gpu_intensity_frames_trunk_B_;
    std::vector<GPUIntensityFrame*> gpu_intensity_frames_branch_B_;
    std::vector<GPUIntensityFrame*> gpu_intensity_frames_leaf_B_;
    std::vector<GPUEdgeFrame*> gpu_edge_frames_B_;
    std::vector<GPUDilatedFrame*> gpu_dilated_frames_trunk_B_;
    std::vector<GPUDilatedFrame*> gpu_dilated_frames_branch_B_;
    std::vector<GPUDilatedFrame*> gpu_dilated_frames_leaf_B_;

    /*Models*/
    GPUModel* gpu_principal_model_;
    std::vector<GPUModel*> gpu_non_principal_models_;

    /*Set Search Range*/
    void SetSearchRange(Point6D range);

    /*Set Search Range*/
    void SetStartingPoint(Point6D starting_point);

    /*Actual Range of Search Direction for Each Variable*/
    Point6D range_;

    /*Starting Point For Search*/
    Point6D starting_point_;

    /*Valid Search Range*/
    bool valid_range_;

    /*Budget*/
    unsigned int budget_;
    /*Plan 008 U8: the per-stage optimizer-variant slot, stored from
     * Initialize and consumed by RunDirectStage's DirectOptimizer ctor.
     * Defaults = the bit-identical classic search (non-default fields are
     * fail-fast stubs in this plan).*/
    DirectOptimizer::Options direct_options_;

    /*Plan 008 U7 (Cut C): the one-line shim over jta::DeriveStageCostParams
     * for a stage CostFunctionManager — the six-name variant list lives in
     * ONE place, the pure TU. Every call site stays exactly where the
     * pre-shim DeriveStageCostParams calls were (in the stage loop AFTER the
     * stage's InitializeActiveCostFunction — the init-gating order is
     * load-bearing).*/
    jta::StageCostParams DeriveStageParams(
        jta_cost_function::CostFunctionManager& manager);

    /*The stage dilate-A / dilate-B (if biplane) / emit UpdateDilationBackground
     * block — one place for the trunk/branch/leaf stage specs (the dilation
     * value is the only per-stage difference).*/
    void ResetStageDilation(size_t frame_index, int dilation);

    /*Run one DIRECT stage (trunk/branch/leaf) using the extracted pure
     * DirectOptimizer with the real GPU eval (plan U6). `range` is the stage
     * search range (already applied to range_ by the caller), and
     * `stage_manager` is the stage's CostFunctionManager; starting_point_ and
     * the cumulative budget_ / cost_function_calls_ members are read by the
     * caller before invoking and written back on return. The stage's Options
     * slot (plan 008 U8: direct_options_) is passed to the DirectOptimizer
     * ctor -- defaults reproduce the pre-Options search bit-identically.*/
    void RunDirectStage(
        Point6D range,
        jta_cost_function::CostFunctionManager& stage_manager);

    /*Cost Function Calls*/
    unsigned int cost_function_calls_;

    /*Lowest Min Value*/
    double current_optimum_value_;

    /*Argument (Location) of Lowest Min Value*/
    Point6D current_optimum_location_;

    /*Error Ocurred*/
    bool error_occurrred_;

    /*Clock for Timing Speed*/
    /*(Milliseconds)*/
    clock_t start_clock_, update_screen_clock_;

    /*Store Post Matrix on Cost Functions*/
    PoseMatrix pose_storage_;

    /*Plan 008 U9 (Cut B): the run's stage script, built ONCE in Initialize
     * from the settings + directive (jta::BuildStageScript, U7). Optimize()'s
     * per-frame loop iterates it — the stage policy is data, not code (the
     * pre-Cut-B trunk/branch/leaf blocks are transcribed verbatim into the
     * per-spec cases).*/
    jta::StageScript stage_script_;

    /*Flag For Being in Either Trunk, Branch, or Z*/
    unsigned int search_stage_flag_;
};

namespace jta {

/*Plan 008 U9 (Cut B): the shared GPU cost adapter — the injected-cost lambda
 * body of RunDirectStage (src/coordinator/optimizer_manager.cpp) and the
 * Tier-2 oracle's twin (test/oracle/oracle_test.cpp) as one named free
 * function. Three consumers converge on it: the production runner
 * (OptimizerManager::RunDirectStage), the flat oracle, and the z-profile
 * probe's cost path.
 *
 * Returns a std::function that (a) sets the already-physical pose on the
 * principal GPU model (biplane: camera-A-to-B conversion via the calibration),
 * then (b) scores the stage's ACTIVE cost function — the exact body of the
 * pre-Cut-B RunDirectStage lambda, transcribed verbatim. Calibration is
 * carried BY VALUE (monoplane default; the future biplane consumer needs no
 * signature change — plan 008 Key Technical Decisions). The caller owns
 * `principal_model` and `stage_manager`; both must outlive the returned
 * std::function (RunDirectStage guarantees this: the search runs synchronously
 * within the call).*/
std::function<double(const Point6D&)> BuildGpuCostAdapter(
    gpu_cost_function::GPUModel* principal_model,
    Calibration calibration,
    jta_cost_function::CostFunctionManager& stage_manager);

}  // namespace jta

#endif /* OPTIMIZER_MANAGER_H */
