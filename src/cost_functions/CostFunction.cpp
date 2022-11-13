/*Cost Function Header*/
#include "CostFunction.h"

namespace jta_cost_function
{
	/*Constructor/Destructor*/

	CostFunction::CostFunction()
	{
		cost_function_name_ = "Nameless_Cost_Function";
	};

	CostFunction::CostFunction(std::string cost_function_name)
	{
		cost_function_name_ = cost_function_name;
	};

	CostFunction::~CostFunction()
	{
	};

	/*Add Parameter (w/ Default Value)*/
	void CostFunction::addParameter(Parameter<double> new_parameter)
	{
		double_parameters_.push_back(new_parameter);
	};

	void CostFunction::addParameter(Parameter<int> new_parameter)
	{
		int_parameters_.push_back(new_parameter);
	};

	void CostFunction::addParameter(Parameter<bool> new_parameter)
	{
		bool_parameters_.push_back(new_parameter);
	};

	/*Set Parameter Values*/
	bool CostFunction::setDoubleParameterValue(std::string parameter_name, double value)
	{
		/*Search for Parameter*/
		for (auto& double_parameter : double_parameters_)
		{
			if (double_parameter.getParameterName() == parameter_name)
			{
				double_parameter.setParameterValue(value);
				return true;
			}
		}
		/*Couldn't Find Parameter*/
		return false;
	};

	bool CostFunction::setIntParameterValue(std::string parameter_name, int value)
	{
		/*Search for Parameter*/
		for (auto& int_parameter : int_parameters_)
		{
			if (int_parameter.getParameterName() == parameter_name)
			{
				int_parameter.setParameterValue(value);
				return true;
			}
		}
		/*Couldn't Find Parameter*/
		return false;
	};

	bool CostFunction::setBoolParameterValue(std::string parameter_name, bool value)
	{
		/*Search for Parameter*/
		for (auto& bool_parameter : bool_parameters_)
		{
			if (bool_parameter.getParameterName() == parameter_name)
			{
				bool_parameter.setParameterValue(value);
				return true;
			}
		}
		/*Couldn't Find Parameter*/
		return false;
	};

	/*Get Parameter Values*/
	bool CostFunction::getDoubleParameterValue(std::string parameter_name, double& value)
	{
		/*Search for Parameter*/
		for (auto& double_parameter : double_parameters_)
		{
			if (double_parameter.getParameterName() == parameter_name)
			{
				value = double_parameter.getParameterValue();
				return true;
			}
		}
		/*Couldn't Find Parameter*/
		return false;
	};

	bool CostFunction::getIntParameterValue(std::string parameter_name, int& value)
	{
		/*Search for Parameter*/
		for (auto& int_parameter : int_parameters_)
		{
			if (int_parameter.getParameterName() == parameter_name)
			{
				value = int_parameter.getParameterValue();
				return true;
			}
		}
		/*Couldn't Find Parameter*/
		return false;
	};

	bool CostFunction::getBoolParameterValue(std::string parameter_name, bool& value)
	{
		/*Search for Parameter*/
		for (auto& bool_parameter : bool_parameters_)
		{
			if (bool_parameter.getParameterName() == parameter_name)
			{
				value = bool_parameter.getParameterValue();
				return true;
			}
		}
		/*Couldn't Find Parameter*/
		return false;
	};

	/*Get Parameters by Type Groups*/
	std::vector<Parameter<double>> CostFunction::getDoubleParameters()
	{
		return double_parameters_;
	};

	std::vector<Parameter<int>> CostFunction::getIntParameters()
	{
		return int_parameters_;
	};

	std::vector<Parameter<bool>> CostFunction::getBoolParameters()
	{
		return bool_parameters_;
	};

	/*Get/Set Cost Function Name*/
	std::string CostFunction::getCostFunctionName()
	{
		return cost_function_name_;
	};

	void CostFunction::setCostFunctionName(std::string cost_function_name)
	{
		cost_function_name_ = cost_function_name;
	};

	bool CostFunction::Initialize(std::string& error_message)
	{

		error_message = "You are loading the default cost function, not a specific class, please reach out to tech support";
		return false;
	};

	bool CostFunction::Destruct(std::string& error_message)
	{
		return false;
	};

	double CostFunction::Run()
	{
		return 0.0;
	};

	/*Upload Data (Images,Poses etc.)*/
	void CostFunction::uploadDataToCostFunction(std::vector<gpu_cost_function::GPUEdgeFrame*>* gpu_edge_frames_A,
	                                            std::vector<gpu_cost_function::GPUDilatedFrame*>* gpu_dilated_frames_A,
	                                            std::vector<gpu_cost_function::GPUIntensityFrame*>*
	                                            gpu_intensity_frames_A,
	                                            std::vector<gpu_cost_function::GPUEdgeFrame*>* gpu_edge_frames_B,
	                                            std::vector<gpu_cost_function::GPUDilatedFrame*>* gpu_dilated_frames_B,
	                                            std::vector<gpu_cost_function::GPUIntensityFrame*>*
	                                            gpu_intensity_frames_B,
	                                            gpu_cost_function::GPUModel* gpu_principal_model,
	                                            std::vector<gpu_cost_function::GPUModel*>* gpu_non_principal_models,
	                                            gpu_cost_function::GPUMetrics* gpu_metrics,
	                                            PoseMatrix* pose_storage,
	                                            bool biplane_mode)
	{
		/*Storage for Data (images, poses ,etc.) set to null*/
		/*Pointer to Vector of GPU Frame Pointers*/
		/*Camera A*/
		gpu_edge_frames_A_ = gpu_edge_frames_A;
		gpu_dilated_frames_A_ = gpu_dilated_frames_A;
		gpu_intensity_frames_A_ = gpu_intensity_frames_A;
		/*Camera B*/
		gpu_edge_frames_B_ = gpu_edge_frames_B;
		gpu_dilated_frames_B_ = gpu_dilated_frames_B;
		gpu_intensity_frames_B_ = gpu_intensity_frames_B;
		/*Pointer to Vector of principal GPU Model Pointer*/
		gpu_principal_model_ = gpu_principal_model;
		/*Pointer to Vector of non-principal GPU Model Pointers*/
		gpu_non_principal_models_ = gpu_non_principal_models;
		/*GPU Metrics Initialize*/
		gpu_metrics_ = gpu_metrics;

		/*Pose Storage Initialize*/
		pose_storage_ = pose_storage;

		/*Biplane Mode*/
		biplane_mode_ = biplane_mode;
	};
}
