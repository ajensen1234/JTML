/*Cost Function Header*/
#include "CostFunction.h"

namespace jta_cost_function {
/*Constructor/Destructor*/
CostFunction::CostFunction() {
    cost_function_name_ = "Nameless_Cost_Function";
};
CostFunction::CostFunction(std::string cost_function_name) {
    cost_function_name_ = cost_function_name;
};
CostFunction::~CostFunction(){};

/*Add Parameter (w/ Default Value)*/
void CostFunction::addParameter(Parameter<double> new_parameter) {
    double_parameters_.push_back(new_parameter);
};
void CostFunction::addParameter(Parameter<int> new_parameter) {
    int_parameters_.push_back(new_parameter);
};
void CostFunction::addParameter(Parameter<bool> new_parameter) {
    bool_parameters_.push_back(new_parameter);
};

/*Set Parameter Values*/
bool CostFunction::setDoubleParameterValue(std::string parameter_name,
                                           double value) {
    /*Search for Parameter*/
    for (int i = 0; i < double_parameters_.size(); i++) {
        if (double_parameters_[i].getParameterName() == parameter_name) {
            double_parameters_[i].setParameterValue(value);
            return true;
        }
    }
    /*Couldn't Find Parameter*/
    return false;
};
bool CostFunction::setIntParameterValue(std::string parameter_name, int value) {
    /*Search for Parameter*/
    for (int i = 0; i < int_parameters_.size(); i++) {
        if (int_parameters_[i].getParameterName() == parameter_name) {
            int_parameters_[i].setParameterValue(value);
            return true;
        }
    }
    /*Couldn't Find Parameter*/
    return false;
};
bool CostFunction::setBoolParameterValue(std::string parameter_name,
                                         bool value) {
    /*Search for Parameter*/
    for (int i = 0; i < bool_parameters_.size(); i++) {
        if (bool_parameters_[i].getParameterName() == parameter_name) {
            bool_parameters_[i].setParameterValue(value);
            return true;
        }
    }
    /*Couldn't Find Parameter*/
    return false;
};

/*Get Parameter Values*/
bool CostFunction::getDoubleParameterValue(std::string parameter_name,
                                           double &value) {
    /*Search for Parameter*/
    for (int i = 0; i < double_parameters_.size(); i++) {
        if (double_parameters_[i].getParameterName() == parameter_name) {
            value = double_parameters_[i].getParameterValue();
            return true;
        }
    }
    /*Couldn't Find Parameter*/
    return false;
};
bool CostFunction::getIntParameterValue(std::string parameter_name,
                                        int &value) {
    /*Search for Parameter*/
    for (int i = 0; i < int_parameters_.size(); i++) {
        if (int_parameters_[i].getParameterName() == parameter_name) {
            value = int_parameters_[i].getParameterValue();
            return true;
        }
    }
    /*Couldn't Find Parameter*/
    return false;
};
bool CostFunction::getBoolParameterValue(std::string parameter_name,
                                         bool &value) {
    /*Search for Parameter*/
    for (int i = 0; i < bool_parameters_.size(); i++) {
        if (bool_parameters_[i].getParameterName() == parameter_name) {
            value = bool_parameters_[i].getParameterValue();
            return true;
        }
    }
    /*Couldn't Find Parameter*/
    return false;
};

/*Get Parameters by Type Groups*/
std::vector<Parameter<double>> CostFunction::getDoubleParameters() {
    return double_parameters_;
};
std::vector<Parameter<int>> CostFunction::getIntParameters() {
    return int_parameters_;
};
std::vector<Parameter<bool>> CostFunction::getBoolParameters() {
    return bool_parameters_;
};

/*Get/Set Cost Function Name*/
std::string CostFunction::getCostFunctionName() { return cost_function_name_; };
void CostFunction::setCostFunctionName(std::string cost_function_name) {
    cost_function_name_ = cost_function_name;
};

}  // namespace jta_cost_function