#pragma once
#include <iostream>
#include "CostFunction.h"
#include <windows.h>
#include "Parameter.h"
#include "cuda_launch_parameters.h"
#include "gpu_imports.h"
#include "cuda_runtime_api.h"

using namespace jta_cost_function;

class DIRECT_DILATION final : public CostFunction
{
public:
	DIRECT_DILATION();
	double Run() override;
	bool Initialize(std::string& error_message) override;
	bool Destruct(std::string& error_message) override;
	int DIRECT_DILATION_current_dilation_parameter;
	int DIRECT_DILATION_current_white_pix_sum_dilated_comparison_image_A_;
	int DIRECT_DILATION_current_white_pix_sum_dilated_comparison_image_B_;
};


extern "C" __declspec(dllexport) CostFunction* create_fn()
{
	return new DIRECT_DILATION();
}
