/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

#ifndef PARAMETER_H
#define PARAMETER_H
/*Parameter Class Header*/
/* Info: The cost function class contains a vector of parameter classes which
represent any parameters (must be either double, integer, or bool) that the cost
function might require. These cost function parameters are added to the
parameter storage vector in the constructor of the cost function class. Values
for the parameters can be set from the JTA client and are saved between
sessions. Default values for each parameter must be provided in the constructor
for the cost function along with a parameter type and parameter name.*/

/*Standard Library*/
#include <string>
#include <type_traits>

#include "core/preprocessor-defs.h"

/*Custom Namespace for JTA Cost Function Library (Compiling as DLL)*/
namespace jta_cost_function {

template <typename Parameter_Type>
class Parameter {
    static_assert((std::is_same<double, Parameter_Type>::value ||
                   std::is_same<int, Parameter_Type>::value ||
                   std::is_same<bool, Parameter_Type>::value),
                  "Parameter type must be double, int, or bool!");
};

template <>
class Parameter<double> {
   public:
    /*Constructors*/
    JTML_DLL Parameter() {
        parameter_name_ = "Nameless Parameter";
        parameter_value_ = 0;
        parameter_type_ = "DOUBLE";
    };
    JTML_DLL Parameter(std::string parameter_name, double parameter_value) {
        parameter_name_ = parameter_name;
        parameter_value_ = parameter_value;
        parameter_type_ = "DOUBLE";
    };

    /*Methods*/
    /*Get Parameter Name*/
    JTML_DLL std::string getParameterName() { return parameter_name_; };

    /*Get/Set Parameter Value*/
    JTML_DLL double getParameterValue() { return parameter_value_; };
    JTML_DLL void setParameterValue(double parameter_value) {
        parameter_value_ = parameter_value;
    };

    /*Get Class Type*/
    JTML_DLL std::string getParameterType() { return parameter_type_; };

   private:
    /*Variables*/
    /*Parameter Name*/
    std::string parameter_name_;

    /*Parameter Value*/
    int parameter_value_;

    /*Class Type*/
    std::string parameter_type_;
};

template <>
class Parameter<int> {
   public:
    /*Constructors*/
    JTML_DLL Parameter() {
        parameter_name_ = "Nameless Parameter";
        parameter_value_ = 0;
        parameter_type_ = "INT";
    };
    JTML_DLL Parameter(std::string parameter_name, int parameter_value) {
        parameter_name_ = parameter_name;
        parameter_value_ = parameter_value;
        parameter_type_ = "INT";
    };

    /*Methods*/
    /*Get Parameter Name*/
    JTML_DLL std::string getParameterName() { return parameter_name_; };

    /*Get/Set Parameter Value*/
    JTML_DLL int getParameterValue() { return parameter_value_; };
    JTML_DLL void setParameterValue(int parameter_value) {
        parameter_value_ = parameter_value;
    };

    /*Get Class Type*/
    JTML_DLL std::string getParameterType() { return parameter_type_; };

   private:
    /*Variables*/
    /*Parameter Name*/
    std::string parameter_name_;

    /*Parameter Value*/
    int parameter_value_;

    /*Class Type*/
    std::string parameter_type_;
};

template <>
class Parameter<bool> {
   public:
    /*Constructors*/
    JTML_DLL Parameter() {
        parameter_name_ = "Nameless Parameter";
        parameter_value_ = 0;
        parameter_type_ = "BOOL";
    };
    JTML_DLL Parameter(std::string parameter_name, bool parameter_value) {
        parameter_name_ = parameter_name;
        parameter_value_ = parameter_value;
        parameter_type_ = "BOOL";
    };

    /*Methods*/
    /*Get Parameter Name*/
    JTML_DLL std::string getParameterName() { return parameter_name_; };

    /*Get/Set Parameter Value*/
    JTML_DLL bool getParameterValue() { return parameter_value_; };
    JTML_DLL void setParameterValue(bool parameter_value) {
        parameter_value_ = parameter_value;
    };

    /*Get Class Type*/
    JTML_DLL std::string getParameterType() { return parameter_type_; };

   private:
    /*Variables*/
    /*Parameter Name*/
    std::string parameter_name_;

    /*Parameter Value*/
    bool parameter_value_;

    /*Class Type*/
    std::string parameter_type_;
};
}  // namespace jta_cost_function

#endif  // PARAMETER_H
