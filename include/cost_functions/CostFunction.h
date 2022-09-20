#pragma once

/*Cost Function Parameters*/
#include "Parameter.h"

/*Standard Library*/
#include <vector>
#include <string>

namespace jta_cost_function {

	class CostFunction {
	public:
		/*Constructor*/
		__declspec(dllexport) CostFunction();
		__declspec(dllexport) CostFunction(std::string cost_function_name);
		__declspec(dllexport) ~CostFunction();

		/*Add Parameter (w/ Default Value)*/
		__declspec(dllexport) void addParameter(Parameter<double> new_parameter);
		__declspec(dllexport) void addParameter(Parameter<int> new_parameter);
		__declspec(dllexport) void addParameter(Parameter<bool> new_parameter);
		 
		/*Set Parameter Values (Bool for Success)*/
		__declspec(dllexport) bool setDoubleParameterValue(std::string parameter_name, double value);
		__declspec(dllexport) bool setIntParameterValue(std::string parameter_name, int value);
		__declspec(dllexport) bool setBoolParameterValue(std::string parameter_name, bool value);

		/*Get Parameter Values (Bool for Success)*/
		__declspec(dllexport) bool getDoubleParameterValue(std::string parameter_name, double &value);
		__declspec(dllexport) bool getIntParameterValue(std::string parameter_name, int &value);
		__declspec(dllexport) bool getBoolParameterValue(std::string parameter_name, bool &value);

		/*Get Parameters by Type Groups*/
		__declspec(dllexport) std::vector<Parameter<double>> getDoubleParameters();
		__declspec(dllexport) std::vector<Parameter<int>> getIntParameters();
		__declspec(dllexport) std::vector<Parameter<bool>> getBoolParameters();

		/*Get/Set Cost Function Name*/
		__declspec(dllexport) std::string getCostFunctionName();
		__declspec(dllexport) void setCostFunctionName(std::string cost_function_name);


	private:
		/*Containers for Parameters*/
		std::vector<Parameter<double>> double_parameters_;
		std::vector<Parameter<int>> int_parameters_;
		std::vector<Parameter<bool>> bool_parameters_;

		/*Cost Function Name*/
		std::string cost_function_name_;
	};
}

