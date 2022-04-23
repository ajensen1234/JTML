/*Manages Optimization in a Seperate QT Thread*/

#ifndef OPTIMIZER_MANAGER_H
#define OPTIMIZER_MANAGER_H

/*Custom CUDA Headers*/
#include <gpu_model.cuh>
#include <gpu_intensity_frame.cuh>
#include <gpu_edge_frame.cuh>
#include <gpu_dilated_frame.cuh>
#include <gpu_metrics.cuh>
#include "calibration.h"

/*QT Threading*/
#include <qobject.h>
#include <qthread.h>
#include <QModelIndex>

/*Frame and Model and Location Storage*/
#include "frame.h"
#include "model.h"
#include "location_storage.h"

/*Direct Library*/
#include "data_structures_6D.h"
#include "direct_data_storage.h"

/*Custom Calibration Struct (Used in CUDA GPU METRICS)*/
#include "calibration.h"

/*Optimizer Settings*/
#include "optimizer_settings.h"

/*Metric Types*/
#include "metric_enum.h"

/*Cost Function Library*/
#include "CostFunctionManager.h"

#include "sym_trap.h"

#include <vector>
#include <string>

using namespace gpu_cost_function;

class OptimizerManager : public QObject
{
	Q_OBJECT

public:
	explicit OptimizerManager(QObject* parent = 0);
	/*Sets up Everything for Optimizer and Also Handles CUDA Initialization, Can Fail!*/
	bool Initialize(
		QThread& optimizer_thread,
		Calibration calibration_file,
		std::vector<Frame> camera_A_frame_list, std::vector<Frame> camera_B_frame_list, unsigned int current_frame_index,
		std::vector<Model> model_list, QModelIndexList selected_models, unsigned int primary_model_index,
		LocationStorage pose_matrix,
		OptimizerSettings opt_settings,
		jta_cost_function::CostFunctionManager trunk_manager, jta_cost_function::CostFunctionManager branch_manager, jta_cost_function::CostFunctionManager leaf_manager,
		QString opt_directive,
		QString& error_message,
		sym_trap* sym_trap_obj);
	~OptimizerManager();


	/* get cost numbers for symmetry plotting */
	double EvaluateCostFunctionAtPoint(Point6D point, int stage);
	void CalculateSymTrap();

signals:
	/*Update Blue Current Optimum*/
	void UpdateOptimum(double, double, double, double, double, double, unsigned int);
	/*Finished*/
	void finished();
	/*Finished Optimizing Frame, Send Optimum to MainScreen, The last bool indicates if should move to next frame*/
	void OptimizedFrame(double, double, double, double, double, double,bool, unsigned int, bool, QString);
	/*Uh oh There was an Error. The string is the message*/
	void OptimizerError(QString);
	/*Update Display with Speed, Cost Function Calls, Current Minimum*/
	void UpdateDisplay(double, int, double, unsigned int);
	/*Update Dilation Background*/
	void UpdateDilationBackground();

	void CostFuncAtPoint(double);
	void onUpdateOrientationSymTrap(double, double, double, double, double, double);
	void onProgressBarUpdate(int);

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
	sym_trap* sym_trap_obj;

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

	std::vector<int> img_indices_;

	QString optimization_directive_;

	void create_image_indices(std::vector<int> &img_indices, int start, int end);

	/*Error Check*/
	cudaError_t cuda_status_;

	/*Correctly Initialized*/
	bool succesfull_initialization_;

	/*Dilation Values Based on Parameter Names (Dilation or DILATION or dilation) that are ints*/
	int trunk_dilation_val_;
	int branch_dilation_val_;
	int leaf_dilation_val_;

	/*Black Silhouette Values Based on Parameter Names (Black_Silhouette or Dark_Silhouette or BLACK_SILHOUETTE or DARK_SILHOUETTE or black_silhouette or dark_silhouette)*/
	bool trunk_dark_silhouette_val_;
	bool branch_dark_silhouette_val_;
	bool leaf_dark_silhouette_val_;

	/*GPU Metrics Class*/
	GPUMetrics* gpu_metrics_;

	/*CUDA Cost Function Objects (Vector of GPU Models and vector of GPU Frames - note Dilated and Intensity must have own vector
	for each stage because their values could change with the stage from a black silhouette bool or a dilation int)*/
	/*Camera A (Monoplane or Biplane)*/
	std::vector<GPUIntensityFrame*> gpu_intensity_frames_trunk_A_;
	std::vector<GPUIntensityFrame*> gpu_intensity_frames_branch_A_;
	std::vector<GPUIntensityFrame*> gpu_intensity_frames_leaf_A_;
	std::vector<GPUEdgeFrame*> gpu_edge_frames_A_;
	std::vector<GPUDilatedFrame*> gpu_dilated_frames_trunk_A_;
	std::vector<GPUDilatedFrame*> gpu_dilated_frames_branch_A_; 
	std::vector<GPUDilatedFrame*> gpu_dilated_frames_leaf_A_;
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

	/*Data Storage*/
	DirectDataStorage data_;

	/*Potentially Optimal Column Ids (Given by Convex Hull)*/
	std::vector<int> potentially_optimal_col_ids_;

	/*Potentially Optimal Hyperboxes (Taken from potentially optimal column ids)*/
	std::vector<HyperBox6D> potentially_optimal_hyperboxes_;

	/*Convex Hull Loop of DIRECT*/
	void ConvexHull();

	/*Trisect Potentially Optimal Hypers and Sample and Add
	to the storage. Delete old ones.*/
	void TrisectPotentiallyOptimal();

	/*Evaluate Cost Function at Given Point*/
	double EvaluateCostFunction(Point6D point);

	/*Denormalize Range Point (converts Unit Point to correct values)*/
	Point6D DenormalizeRange(Point6D unit_point);

	/*Denormalize Point From Center (converts Unit Point to correct values)*/
	Point6D DenormalizeFromCenter(Point6D unit_point);

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

	/*Flag For Being in Either Trunk, Branch, or Z*/
	unsigned int search_stage_flag_;

};


#endif /* OPTIMIZER_MANAGER_H */