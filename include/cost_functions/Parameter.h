#ifndef PARAMETER_H
#define PARAMETER_H
/*Parameter Class Header*/
/* Info: The cost function class contains a vector of parameter classes which represent
any parameters (must be either double, integer, or bool) that the cost function might require.
These cost function parameters are added to the parameter storage vector in the constructor of
the cost function class. Values for the parameters can be set from the JTA client and are
saved between sessions. Default values for each parameter must be provided in the constructor
for the cost function along with a parameter type and parameter name.*/

/*Standard Library*/
#include <string>
#include <type_traits>

/*Custom Namespace for JTA Cost Function Library (Compiling as DLL)*/
namespace jta_cost_function {

	template <typename Parameter_Type>
	class Parameter {
		static_assert((std::is_same<double, Parameter_Type>::value
			|| std::is_same<int, Parameter_Type>::value
			|| std::is_same<bool, Parameter_Type>::value), "Parameter type must be double, int, or bool!");
	};

	template <>
	class Parameter<double> {
	public:
		/*Constructors*/
		__declspec(dllexport) Parameter() {
			parameter_name_ = "Nameless Parameter";
			parameter_value_ = 0;
			parameter_type_ = "DOUBLE";
		};
		__declspec(dllexport) Parameter(std::string parameter_name, double parameter_value) {
			parameter_name_ = parameter_name;
			parameter_value_ = parameter_value;
			parameter_type_ = "DOUBLE";
		};

		/*Methods*/
		/*Get Parameter Name*/
		__declspec(dllexport) std::string getParameterName() {
			return parameter_name_;
		};

		/*Get/Set Parameter Value*/
		__declspec(dllexport) double getParameterValue() {
			return parameter_value_;
		};
		__declspec(dllexport) void setParameterValue(double parameter_value) {
			parameter_value_ = parameter_value;
		};

		/*Get Class Type*/
		__declspec(dllexport) std::string getParameterType() {
			return parameter_type_;
		};

	private:
		/*Variables*/
		/*Parameter Name*/
		std::string parameter_name_;

		/*Parameter Value*/
		double parameter_value_;

		/*Class Type*/
		std::string parameter_type_;
	};

	template <>
	class Parameter<int> {
	public:
		/*Constructors*/
		__declspec(dllexport) Parameter() {
			parameter_name_ = "Nameless Parameter";
			parameter_value_ = 0;
			parameter_type_ = "INT";
		};
		__declspec(dllexport) Parameter(std::string parameter_name, int parameter_value) {
			parameter_name_ = parameter_name;
			parameter_value_ = parameter_value;
			parameter_type_ = "INT";
		};

		/*Methods*/
		/*Get Parameter Name*/
		__declspec(dllexport) std::string getParameterName() {
			return parameter_name_;
		};

		/*Get/Set Parameter Value*/
		__declspec(dllexport) int getParameterValue() {
			return parameter_value_;
		};
		__declspec(dllexport) void setParameterValue(int parameter_value) {
			parameter_value_ = parameter_value;
		};

		/*Get Class Type*/
		__declspec(dllexport) std::string getParameterType() {
			return parameter_type_;
		};

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
		__declspec(dllexport) Parameter() {
			parameter_name_ = "Nameless Parameter";
			parameter_value_ = 0;
			parameter_type_ = "BOOL";
		};
		__declspec(dllexport) Parameter(std::string parameter_name, bool parameter_value) {
			parameter_name_ = parameter_name;
			parameter_value_ = parameter_value;
			parameter_type_ = "BOOL";
		};

		/*Methods*/
		/*Get Parameter Name*/
		__declspec(dllexport) std::string getParameterName() {
			return parameter_name_;
		};

		/*Get/Set Parameter Value*/
		__declspec(dllexport) bool getParameterValue() {
			return parameter_value_;
		};
		__declspec(dllexport) void setParameterValue(bool parameter_value) {
			parameter_value_ = parameter_value;
		};

		/*Get Class Type*/
		__declspec(dllexport) std::string getParameterType() {
			return parameter_type_;
		};

	private:
		/*Variables*/
		/*Parameter Name*/
		std::string parameter_name_;

		/*Parameter Value*/
		bool parameter_value_;

		/*Class Type*/
		std::string parameter_type_;
	};
}

#endif //PARAMETER_H