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
		  CostFunction();
		  CostFunction(std::string cost_function_name);
		  ~CostFunction();

		/*Add Parameter (w/ Default Value)*/
		  void addParameter(Parameter<double> new_parameter);
		  void addParameter(Parameter<int> new_parameter);
		  void addParameter(Parameter<bool> new_parameter);
		 
		/*Set Parameter Values (Bool for Success)*/
		  bool setDoubleParameterValue(std::string parameter_name, double value);
		  bool setIntParameterValue(std::string parameter_name, int value);
		  bool setBoolParameterValue(std::string parameter_name, bool value);

		/*Get Parameter Values (Bool for Success)*/
		  bool getDoubleParameterValue(std::string parameter_name, double &value);
		  bool getIntParameterValue(std::string parameter_name, int &value);
		  bool getBoolParameterValue(std::string parameter_name, bool &value);

		/*Get Parameters by Type Groups*/
		  std::vector<Parameter<double>> getDoubleParameters();
		  std::vector<Parameter<int>> getIntParameters();
		  std::vector<Parameter<bool>> getBoolParameters();

		/*Get/Set Cost Function Name*/
		  std::string getCostFunctionName();
		  void setCostFunctionName(std::string cost_function_name);


	private:
		/*Containers for Parameters*/
		std::vector<Parameter<double>> double_parameters_;
		std::vector<Parameter<int>> int_parameters_;
		std::vector<Parameter<bool>> bool_parameters_;

		/*Cost Function Name*/
		std::string cost_function_name_;
	};
}

