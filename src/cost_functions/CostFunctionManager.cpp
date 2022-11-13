/*Cost Function Manager*/
#include "CostFunctionManager.h"
namespace fs = std::filesystem;
/******************************************************************************/
/******************************************************************************/
/******************************** BEGIN WARNING *******************************/
/******************************************************************************/
/*************************DO NOT EDIT ANYTING IN THIS FILE ********************/
/******************************************************************************/
/******************************************************************************/
using create_cf = jta_cost_function::CostFunction* (*)();
std::string get_path_as_str(const fs::path& p) { return (p.stem().string()); }

namespace jta_cost_function
{
	/*Constructor/Destructor*/
	CostFunctionManager::CostFunctionManager(Stage stage)
	{
		/*Load the listed cost functions to the vector of available cost functions*/
		list_cost_functions();
		std::cout << "Construction of CF Manager" << std::endl;

		/*Set Active Cost Function as the Default (DIRECT_DILATION)*/
		setActiveCostFunction("DIRECT_DILATION");

		/*Storage for Data (images, poses ,etc.) set to null*/
		/*Pointer to Vector of GPU Frame Pointers*/
		/*Camera A*/
		gpu_edge_frames_A_ = nullptr;
		gpu_dilated_frames_A_ = nullptr;
		gpu_intensity_frames_A_ = nullptr;
		/*Camera B*/
		gpu_edge_frames_B_ = nullptr;
		gpu_dilated_frames_B_ = nullptr;
		gpu_intensity_frames_B_ = nullptr;
		/*Pointer to Vector of principal GPU Model Pointer*/
		gpu_principal_model_ = nullptr;
		/*Pointer to Vector of non-principal GPU Model Pointers*/
		gpu_non_principal_models_ = nullptr;

		/*GPU Metrics Initialize*/
		gpu_metrics_ = nullptr;

		/*Pose Storage Initialize*/
		pose_storage_ = nullptr;

		/*Initialize stage*/
		stage_ = stage;
		if (stage_ != Stage::Trunk || stage_ != Stage::Branch || stage_ != Stage::Leaf)
			stage_ = Stage::Trunk;

		/*Current Frame Index (0 based)*/
		current_frame_index_ = 0;

		///*Pose Storage*/
	};

	CostFunctionManager::CostFunctionManager()
	{
		/*Load the listed cost functions to the vector of available cost functions*/
		list_cost_functions();
		std::cout << "default constructor for manager" << std::endl;

		/*Set Active Cost Function as the Default (DIRECT_DILATION)*/
		setActiveCostFunction("DIRECT_DILATION");

		/*Storage for Data (images, poses ,etc.) set to null*/
		/*Pointer to Vector of GPU Frame Pointers*/
		/*Camera A*/
		gpu_edge_frames_A_ = nullptr;
		gpu_dilated_frames_A_ = nullptr;
		gpu_intensity_frames_A_ = nullptr;
		/*Camera B*/
		gpu_edge_frames_B_ = nullptr;
		gpu_dilated_frames_B_ = nullptr;
		gpu_intensity_frames_B_ = nullptr;
		/*Pointer to Vector of principal GPU Model Pointer*/
		gpu_principal_model_ = nullptr;
		/*Pointer to Vector of non-principal GPU Model Pointers*/
		gpu_non_principal_models_ = nullptr;

		/*GPU Metrics Initialize*/
		gpu_metrics_ = nullptr;

		/*Pose Storage Initialize*/
		pose_storage_ = nullptr;

		/*Initialize stage*/
		stage_ = Stage::Trunk;

		/*Current Frame Index (0 based)*/
		current_frame_index_ = 0;

		/*Biplane Mode*/
		biplane_mode_ = false;
	};

	CostFunctionManager::~CostFunctionManager()
	{
	};

	/*Upload Data (Images,Poses etc.)*/
	void CostFunctionManager::UploadData(std::vector<gpu_cost_function::GPUEdgeFrame*>* gpu_edge_frames_A,
	                                     std::vector<gpu_cost_function::GPUDilatedFrame*>* gpu_dilated_frames_A,
	                                     std::vector<gpu_cost_function::GPUIntensityFrame*>* gpu_intensity_frames_A,
	                                     std::vector<gpu_cost_function::GPUEdgeFrame*>* gpu_edge_frames_B,
	                                     std::vector<gpu_cost_function::GPUDilatedFrame*>* gpu_dilated_frames_B,
	                                     std::vector<gpu_cost_function::GPUIntensityFrame*>* gpu_intensity_frames_B,
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

	/*Set Active Cost Function*/
	void CostFunctionManager::setActiveCostFunction(std::string cost_function_name)
	{
		/*Check Active Cost Function Name Exists*/
		for (auto& available_cost_function : available_cost_functions_)
		{
			if (available_cost_function->getCostFunctionName() == cost_function_name)
			{
				active_cost_function_ = cost_function_name;
				active_cost_function_class_ = available_cost_function;
				return;
			}
		}

		/*Otherwise set DIRECT_DILATION as default
		 * This should never be called if we are defining
		 * things correctely.
		 */
		active_cost_function_ = "DIRECT_DILATION";
	};

	/*Update Cost Function Values from Saved Session*/
	bool CostFunctionManager::updateCostFunctionParameterValues(std::string cost_function_name,
	                                                            std::string parameter_name, double value)
	{
		/*Check Active Cost Function Name Exists*/
		for (auto& available_cost_function : available_cost_functions_)
		{
			if (available_cost_function->getCostFunctionName() == cost_function_name)
			{
				/*Check Parameter Name Exsits*/
				for (int j = 0; j < available_cost_function->getDoubleParameters().size(); j++)
					if (available_cost_function->getDoubleParameters()[j].getParameterName() == parameter_name)
					{
						available_cost_function->getDoubleParameters()[j].setParameterValue(value);
						return true;
					}
			}
		}
		/*Unsuccessful*/
		return false;
	};

	bool CostFunctionManager::updateCostFunctionParameterValues(std::string cost_function_name,
	                                                            std::string parameter_name, int value)
	{
		/*Check Active Cost Function Name Exists*/
		for (auto& available_cost_function : available_cost_functions_)
		{
			if (available_cost_function->getCostFunctionName() == cost_function_name)
			{
				/*Check Parameter Name Exsits*/
				for (int j = 0; j < available_cost_function->getIntParameters().size(); j++)
					if (available_cost_function->getIntParameters()[j].getParameterName() == parameter_name)
					{
						available_cost_function->getIntParameters()[j].setParameterValue(value);
						return true;
					}
			}
		}
		/*Unsuccessful*/
		return false;
	};

	bool CostFunctionManager::updateCostFunctionParameterValues(std::string cost_function_name,
	                                                            std::string parameter_name, bool value)
	{
		/*Check Active Cost Function Name Exists*/
		for (auto& available_cost_function : available_cost_functions_)
		{
			if (available_cost_function->getCostFunctionName() == cost_function_name)
			{
				/*Check Parameter Name Exsits*/
				for (int j = 0; j < available_cost_function->getBoolParameters().size(); j++)
				{
					if (available_cost_function->getBoolParameters()[j].getParameterName() == parameter_name)
					{
						available_cost_function->getBoolParameters()[j].setParameterValue(value);
						return true;
					}
				}
			}
		}
		/*Unsuccessful*/
		return false;
	};


	/*Return Available Cost Functions*/
	std::vector<CostFunction> CostFunctionManager::getAvailableCostFunctions()
	{
		std::cout << " this is before we try to create the vector" << std::endl;
		std::vector<CostFunction> cfs;
		for (auto& cf : available_cost_functions_)
		{
			cfs.push_back(*cf);
		}
		std::cout << "This is after we create the vector" << std::endl;
		return cfs;
	};

	/*Return Active Cost Function*/
	std::string CostFunctionManager::getActiveCostFunction()
	{
		return active_cost_function_;
	}

	/*Return Active Cost Function Class*/
	/*Removing this function for now*/
	CostFunction* CostFunctionManager::getActiveCostFunctionClass()
	{
		/*for (auto& available_cost_function : available_cost_functions_)
		{
			if (available_cost_function.getCostFunctionName() == active_cost_function_)
			{
				return &available_cost_function;
			}
		}*/
		return active_cost_function_class_;

		/*If all fails return blank class*/
		//return new CostFunction();
	};

	/*Return Cost Function Class*/
	CostFunction* CostFunctionManager::getCostFunctionClass(std::string cost_function_name)
	{
		for (auto& available_cost_function : available_cost_functions_)
		{
			if (available_cost_function->getCostFunctionName() == cost_function_name)
			{
				return available_cost_function;
			}
		}

		/*If all fails return blank class*/
		return new CostFunction();
	};

	/*Set Current Frame Index*/
	void CostFunctionManager::setCurrentFrameIndex(unsigned int current_frame_index)
	{
		current_frame_index_ = current_frame_index;
	};

	/******************************** WARNING *************************************/
	/******************************************************************************/
	/*************************DO NOT EDIT FUNCTIONS BELOW *************************/
	/******************************************************************************/
	/*FUNCTIONS THAT INTERACT WITH WIZARD*/
	/*Call Active Cost Function*/
	double CostFunctionManager::callActiveCostFunction()
	{
		return active_cost_function_class_->Run();
	};
	/*Call Stage Initializer for Active Cost Function*/
	bool CostFunctionManager::InitializeActiveCostFunction(std::string& error_message)
	{
	/*	if (active_cost_function_ == "DIRECT_DILATION")
		{
			return false;
		}*/
		std::cout << "Before uploading data" << std::endl;
		active_cost_function_class_->uploadDataToCostFunction(
			gpu_edge_frames_A_,
			gpu_dilated_frames_A_,
			gpu_intensity_frames_A_,
			gpu_edge_frames_B_,
			gpu_dilated_frames_B_,
			gpu_intensity_frames_B_,
			gpu_principal_model_,
			gpu_non_principal_models_,
			gpu_metrics_,
			pose_storage_,
			biplane_mode_
		);
		std::cout << "After uploading data, before initialize function called" << std::endl;
		std::cout << active_cost_function_class_->getCostFunctionName() << std::endl;
		bool success = active_cost_function_class_->Initialize(error_message);
		if (success)
		{
			std::cout << "Initialization worked";
		} else
		{
			std::cout << "Initialization didn't work";
		}
		return success;

		//error_message = "Could not find active cost function: " + active_cost_function_;
		//return false;
	};
	/*Call Stage Destructor for Active Cost Function*/
	bool CostFunctionManager::DestructActiveCostFunction(std::string& error_message)
	{
		/*if (active_cost_function_ == "DIRECT_DILATION")
		{
			return false;
		}
		error_message = "Could not find active cost function: " + active_cost_function_;
		return false;*/
		return active_cost_function_class_->Destruct(error_message);
	};


	void CostFunctionManager::list_cost_functions()
	{
		using cf_creator = CostFunction* (*)();

		/*Need to loop through the directory listing*/
		for (auto& p : fs::directory_iterator("cost_functions"))
		{
			std::string path = "./cost_functions/" + get_path_as_str(p);
			std::cout << path << std::endl;
			cost_function_paths_.push_back(path);
			HMODULE hMod = LoadLibrary(path.c_str());

			if (hMod == nullptr)
			{
				std::cout << "DLL was not loaded" << std::endl;
			}
			else
			{
				std::cout << "DLL was loaded" << std::endl;
				auto func = (cf_creator)GetProcAddress(hMod, "create_fn");
				if (func == nullptr)
				{
					std::cout << "Function was not loaded correctly" << std::endl;
				}
				else
				{
					std::cout << "Function was loaded properly" << std::endl;
					CostFunction* cf = func();
					//CostFunction temp_cf(*cf)
					available_cost_functions_.push_back(cf);
					//delete cf; // do I delete this?
				}
			}
		}
	}

	/*END FUNCTIONS THAT INTERACT WITH WIZARD*/
	/******************************** END WARNING *********************************/
	/******************************************************************************/
	/*************************DO NOT EDIT FUNCTIONS ABOVE *************************/
	/******************************************************************************/
}

/******************************************************************************/
/******************************************************************************/
/******************************** END WARNING *********************************/
/******************************************************************************/
/*************************DO NOT EDIT ANYTING IN THIS FILE ********************/
/******************************************************************************/
/******************************************************************************/
