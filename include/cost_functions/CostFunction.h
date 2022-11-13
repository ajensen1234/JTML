#pragma once

/*Cost Function Parameters*/
#include "Parameter.h"

/*Cost Function Tools Library*/

/*Standard Library*/
#include <vector>
#include <string>


namespace jta_cost_function
{
	class __declspec(dllexport) CostFunction
	{
	public:
		/*Constructor*/
		CostFunction();
		CostFunction(std::string cost_function_name);
		virtual ~CostFunction();

		/*Add Parameter (w/ Default Value)*/
		void addParameter(Parameter<double> new_parameter);
		void addParameter(Parameter<int> new_parameter);
		void addParameter(Parameter<bool> new_parameter);

		/*Set Parameter Values (Bool for Success)*/
		bool setDoubleParameterValue(std::string parameter_name, double value);
		bool setIntParameterValue(std::string parameter_name, int value);
		bool setBoolParameterValue(std::string parameter_name, bool value);

		/*Get Parameter Values (Bool for Success)*/
		bool getDoubleParameterValue(std::string parameter_name, double& value);
		bool getIntParameterValue(std::string parameter_name, int& value);
		bool getBoolParameterValue(std::string parameter_name, bool& value);

		/*Get Parameters by Type Groups*/
		std::vector<Parameter<double>> getDoubleParameters();
		std::vector<Parameter<int>> getIntParameters();
		std::vector<Parameter<bool>> getBoolParameters();

		/*Get/Set Cost Function Name*/
		std::string getCostFunctionName();
		void setCostFunctionName(std::string cost_function_name);

		/*Upload Data to the Cost Function*/
		void uploadDataToCostFunction(std::vector<gpu_cost_function::GPUEdgeFrame*>* gpu_edge_frames_A,
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

		/*Values belonging to the cost function to initialize, destroy, and run it*/
		virtual bool Initialize(std::string& error_message);
		virtual bool Destruct(std::string& error_message);
		virtual double Run();


	private:
		/*Containers for Parameters*/
		std::vector<Parameter<double>> double_parameters_;
		std::vector<Parameter<int>> int_parameters_;
		std::vector<Parameter<bool>> bool_parameters_;

		/*Cost Function Name*/
		std::string cost_function_name_;

	protected:
		/*Storage for GPU Metrics class*/
		gpu_cost_function::GPUMetrics* gpu_metrics_ = nullptr;

		/*Storage for Data (images, poses ,etc.)*/
		/*Pointer to Vector of GPU Frame Pointers*/
		/*Camera A*/
		std::vector<gpu_cost_function::GPUEdgeFrame*>* gpu_edge_frames_A_ = nullptr;
		std::vector<gpu_cost_function::GPUDilatedFrame*>* gpu_dilated_frames_A_ = nullptr;
		std::vector<gpu_cost_function::GPUIntensityFrame*>* gpu_intensity_frames_A_ = nullptr;
		/*Camera B*/
		std::vector<gpu_cost_function::GPUEdgeFrame*>* gpu_edge_frames_B_ = nullptr;
		std::vector<gpu_cost_function::GPUDilatedFrame*>* gpu_dilated_frames_B_ = nullptr;
		std::vector<gpu_cost_function::GPUIntensityFrame*>* gpu_intensity_frames_B_ = nullptr;

		/*Pointer to Vector of principal GPU Model Pointer*/
		gpu_cost_function::GPUModel* gpu_principal_model_ = nullptr;
		/*Pointer to Vector of non-principal GPU Model Pointers*/
		std::vector<gpu_cost_function::GPUModel*>* gpu_non_principal_models_ = nullptr;
		float* prin_dist_;
		/*Current Frame Index (0 based)*/
		unsigned int current_frame_index_;

		/*Pose Matrix*/
		PoseMatrix* pose_storage_;

		/*Biplane Mode?*/
		bool biplane_mode_;
	};
}
