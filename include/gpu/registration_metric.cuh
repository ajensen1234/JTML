//#ifndef REGISTRATION_METRIC_H
//#define REGISTRATION_METRIC_H
//
///*Cuda*/
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
///*Standard*/
//#include <string>
//
///*CUDA Custom Registration Namespace (Compiling as DLL)*/
//namespace gpu_cost_function {
//
//	/*The dilation metric from direct-jta. Note: ignores dilation sized padding on the boundaries, therefore
//	render engine must output an image that is bigger than original by dilation sized padding on all 4 edges.*/
//	class RegistrationMetric {
//	public:
//		/*Constructor & Destructor*/
//		JTML_DLL RegistrationMetric(int width, int height);
//		JTML_DLL RegistrationMetric();
//		JTML_DLL ~RegistrationMetric();
//
//		/*CUDA API Initialization (Allocation, etc.) Must return cudaSuccess else call destructor.*/
//		JTML_DLL cudaError_t InitializeCUDA(unsigned char* dev_image,
//			unsigned char* edge_comparison_image, unsigned char* dilation_comparison_image, unsigned char* intensity_comparison_image, int device);
//
//		/*Reset Edge Comparison Image Pointer (must be same size as one used in cuda initialization*/
//		JTML_DLL cudaError_t SetEdgeComparisonImage(unsigned char* edge_comparison_image);
//
//		/*Reset Dilation Comparison Image Pointer (must be same size as one used in cuda initialization*/
//		JTML_DLL cudaError_t SetDilationComparisonImage(unsigned char* dilation_comparison_image);
//
//		/*Reset Intensity Comparison Image Pointer (must be same size as one used in cuda initialization*/
//		JTML_DLL cudaError_t SetIntensityComparisonImage(unsigned char* intensity_comparison_image);
//
//		/*Write a .png to Location of Device Image*/
//		JTML_DLL void WriteImage(std::string file_location);
//
//	protected:
//		/*Initialized CUDA Check*/
//		bool intialized_cuda_;
//
//		/*Free CUDA*/
//		void FreeCuda();
//
//		/*Device Pointer for Dilation Comparison Image (Initialized in registration metric)*/
//		unsigned char* dev_edge_comparison_image_;
//
//		/*Device Pointer for Dilation Comparison Image (Initialized in registration metric)*/
//		unsigned char* dev_dilation_comparison_image_;
//
//		/*Device Pointer for Intensity Comparison Image (Initialized in registration metric)*/
//		unsigned char* dev_intensity_comparison_image_;
//
//		/*Device Pointer for Rendered Image (Initialized in render engine)*/
//		unsigned char* dev_image_;
//
//		/*X-ray size with dilation padding on each of the four borders*/
//		int width_;
//		int height_;
//
//		/*Metric Score*/
//		double metric_score_;
//
//		/*Counts Number of White Pixels in Dilation Comparison Image*/
//		int dilation_comparison_white_pix_count_;
//
//		/*Sum of Pixel Intensities in Dilation Comparison Image*/
//		int dilation_comparison_pixel_sum_;
//
//		/*Integer for Pinned Memory if Metric Counts Pixels on GPU (as in dilation metric)
//		This is often used to compute the metric so we include it in the base class definition.*/
//		/*Initialize Pinned Memory for Slightly Faster Transfer*/
//		int* pixel_score_;
//
//		/*Gpu Counterpart of the pixel score*/
//		int* dev_pixel_score_;
//
//		/*Gpu Counterpart of white pixel count*/
//		int* dev_dilation_comparison_white_pix_count_;
//
//		/*Gpu Counterpart pixel intensities sum*/
//		int* dev_dilation_comparison_pixel_sum_;
//	};
//}
//#endif /* REGISTRATION_METRIC_H */
