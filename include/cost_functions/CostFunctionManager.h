#ifndef COSTFUNCTIONMANAGER_H
#define COSTFUNCTIONMANAGER_H

/*Class for Storing Cost Function Info*/
#include "CostFunction.h"

/*Cost Function Tools Library*/
#include "gpu/gpu_image.cuh"
#include "gpu/gpu_frame.cuh"
#include "gpu/gpu_dilated_frame.cuh"
#include "gpu/gpu_edge_frame.cuh"
#include "gpu/gpu_intensity_frame.cuh"
#include "gpu/gpu_model.cuh"
#include "gpu/gpu_metrics.cuh"
#include "gpu/render_engine.cuh"

/*Stage Enum*/
#include "Stage.h"

/*Standard Library*/
#include <vector>

namespace jta_cost_function {
	class CostFunctionManager {
	public:
		/******************************************************************************/
		/**************************PUBLIC DLL FUNCTIONS BEGIN *************************/
		/******************************(DO NOT EDIT)  *********************************/
		/******************************************************************************/
		/*Constructor
		Called once when the client initially loads and populates the list of available
		cost functions. There will be three instances, one for each stage of DIRECT.
		Also sets an active cost function (default is the DIRECT_DILATION).
		The parameters are all default. To load previously saved session parameters, the constructor
		for the client will call the updateCostFunctionParameterValues(...)*/
		__declspec(dllexport) CostFunctionManager(Stage stage);
		__declspec(dllexport) CostFunctionManager();
		__declspec(dllexport) ~CostFunctionManager();

		/*Set Active Cost Function*/
		__declspec(dllexport) void setActiveCostFunction(std::string cost_function_name);

		/*Update Cost Function Values from Saved Session*/
		__declspec(dllexport) bool updateCostFunctionParameterValues(std::string cost_function_name, std::string parameter_name, double value);
		__declspec(dllexport) bool updateCostFunctionParameterValues(std::string cost_function_name, std::string parameter_name, int value);
		__declspec(dllexport) bool updateCostFunctionParameterValues(std::string cost_function_name, std::string parameter_name, bool value);

		/*Call Initialization for Active Cost Function*/
		__declspec(dllexport) bool InitializeActiveCostFunction(std::string &error_message);

		/*Call Destructor for Active Cost Function*/
		__declspec(dllexport) bool DestructActiveCostFunction(std::string &error_message);

		/*Call Active Cost Function*/
		__declspec(dllexport) double callActiveCostFunction();

		/*Get Active Cost Function*/
		__declspec(dllexport) std::string getActiveCostFunction();


		/*Get Active Cost Function Class*/
		__declspec(dllexport) CostFunction* getActiveCostFunctionClass();

		/*Get Cost Function Class*/
		__declspec(dllexport) CostFunction* getCostFunctionClass(std::string cost_function_name);

		/*Get Vector of Cost Functions*/
		__declspec(dllexport) std::vector<CostFunction> getAvailableCostFunctions();

		/*Set Current Frame Index*/
		__declspec(dllexport) void setCurrentFrameIndex(unsigned int current_frame_index);

		/*Upload Data (Images,Poses etc.)*/
		__declspec(dllexport) void UploadData(std::vector<gpu_cost_function::GPUEdgeFrame*>* gpu_edge_frames_A,
		std::vector<gpu_cost_function::GPUDilatedFrame*>* gpu_dilated_frames_A,
		std::vector<gpu_cost_function::GPUIntensityFrame*>* gpu_intensity_frames_A,
		std::vector<gpu_cost_function::GPUEdgeFrame*>* gpu_edge_frames_B,
		std::vector<gpu_cost_function::GPUDilatedFrame*>* gpu_dilated_frames_B,
		std::vector<gpu_cost_function::GPUIntensityFrame*>* gpu_intensity_frames_B,
		gpu_cost_function::GPUModel* gpu_principal_model,
		std::vector<gpu_cost_function::GPUModel*>* gpu_non_principal_models,
		gpu_cost_function::GPUMetrics* gpu_metrics,
		PoseMatrix* pose_storage,
		bool biplane_mode);
		/******************************************************************************/
		/***************************PUBLIC DLL FUNCTIONS END **************************/
		/******************************************************************************/

	private:
		/******************************************************************************/
		/* *******************ESSENTIAL CLASS VARIABLES BEGIN *************************/
		/******************************(DO NOT EDIT)  *********************************/
		/******************************************************************************/

		/*List Cost Functions
		In this function a cost function that will be loaded to the client and optimizer
		must be listed by name. The parameters should also be specified.*/
		void listCostFunctions();

		/*Vector of Cost Functions*/
		std::vector<CostFunction> available_cost_functions_;

		/*Active Cost Function*/
		std::string active_cost_function_;

		/******************************************************************************/
		/*********************ESSENTIAL CLASS VARIABLES END ***************************/
		/******************************************************************************/


		/******************************************************************************/
		/*********************COST FUNCTION VARIABLES BEGIN ***************************/
		/******************************(DO NOT EDIT)  *********************************/
		/******************************************************************************/

		/*Stage Enum*/
		Stage stage_;

		/*Storage for GPU Metrics class*/
		gpu_cost_function::GPUMetrics* gpu_metrics_;

		/*Storage for Data (images, poses ,etc.)*/
		/*Pointer to Vector of GPU Frame Pointers*/
		/*Camera A*/
		std::vector<gpu_cost_function::GPUEdgeFrame*>* gpu_edge_frames_A_;
		std::vector<gpu_cost_function::GPUDilatedFrame*>* gpu_dilated_frames_A_;
		std::vector<gpu_cost_function::GPUIntensityFrame*>* gpu_intensity_frames_A_;
		/*Camera B*/
		std::vector<gpu_cost_function::GPUEdgeFrame*>* gpu_edge_frames_B_;
		std::vector<gpu_cost_function::GPUDilatedFrame*>* gpu_dilated_frames_B_;
		std::vector<gpu_cost_function::GPUIntensityFrame*>* gpu_intensity_frames_B_;

		/*Pointer to Vector of principal GPU Model Pointer*/
		gpu_cost_function::GPUModel* gpu_principal_model_;
		/*Pointer to Vector of non-principal GPU Model Pointers*/
		std::vector<gpu_cost_function::GPUModel*>* gpu_non_principal_models_;
		float* prin_dist_;
		/*Current Frame Index (0 based)*/
		unsigned int current_frame_index_;

		/*Pose Matrix*/
		PoseMatrix* pose_storage_;

		/*Biplane Mode?*/
		bool biplane_mode_;

		/******************************************************************************/
		/************************COST FUNCTION VARIABLES END***************************/
		/******************************************************************************/


		
		/******************************************************************************/
		/*********************CUSTOM COST FUNCTION VARIABLES BEGIN ********************/
		/******************************(DO NOT EDIT)  *********************************/
		/******************************************************************************/
		/*HEADERS THAT INTERACT WITH WIZARD*/
		/*Custom Variable Headers for Cost Functions*/
		#include "sym_trap_functionCustomVariables.h"
		#include "DD_NEW_POLE_CONSTRAINTCustomVariables.h"
		#include "DIRECT_DILATION_POLE_CONSTRAINTCustomVariables.h"
		#include "DIRECT_DILATION_SAME_ZCustomVariables.h"
		#include "DIRECT_DILATION_T1CustomVariables.h"
		#include "DIRECT_DILATIONCustomVariables.h"
		#include "DIRECT_MAHFOUZCustomVariables.h"
		/*END HEADERS THAT INTERACT WITH WIZARD*/
		/******************************************************************************/
		/************************CUSTOM COST FUNCTION VARIABLES END********************/
		/******************************************************************************/




		/******************************** WARNING *************************************/
		/******************************************************************************/
		/*************************DO NOT EDIT FUNCTIONS BELOW *************************/
		/******************************************************************************/
		/*FUNCTIONS THAT INTERACT WITH WIZARD*/
		/*Cost Function Implementations*/
		double costFunctionsym_trap_function();
		double costFunctionDD_NEW_POLE_CONSTRAINT();
		double costFunctionDIRECT_DILATION_POLE_CONSTRAINT();
		double costFunctionDIRECT_DILATION_SAME_Z();
		double costFunctionDIRECT_DILATION_T1();
		double costFunctionDIRECT_DILATION();
		double costFunctionDIRECT_MAHFOUZ();
		/*Cost Function Initializations*/
		bool initializesym_trap_function(std::string& error_message);
		bool initializeDD_NEW_POLE_CONSTRAINT(std::string& error_message);
		bool initializeDIRECT_DILATION_POLE_CONSTRAINT(std::string& error_message);
		bool initializeDIRECT_DILATION_SAME_Z(std::string& error_message);
		bool initializeDIRECT_DILATION_T1(std::string& error_message);
		bool initializeDIRECT_DILATION(std::string& error_message);
		bool initializeDIRECT_MAHFOUZ(std::string& error_message);
		/*Cost Function Destructors*/
		bool destructsym_trap_function(std::string& error_message);
		bool destructDD_NEW_POLE_CONSTRAINT(std::string& error_message);
		bool destructDIRECT_DILATION_POLE_CONSTRAINT(std::string& error_message);
		bool destructDIRECT_DILATION_SAME_Z(std::string& error_message);
		bool destructDIRECT_DILATION_T1(std::string& error_message);
		bool destructDIRECT_DILATION(std::string& error_message);
		bool destructDIRECT_MAHFOUZ(std::string& error_message);
		/*END FUNCTIONS THAT INTERACT WITH WIZARD*/
		/******************************** END WARNING *********************************/
		/******************************************************************************/
		/*************************DO NOT EDIT FUNCTIONS ABOVE *************************/
		/******************************************************************************/
	};
}

#endif //COSTFUNCTIONMANAGER_H
