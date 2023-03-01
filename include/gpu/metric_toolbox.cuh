//#ifndef METRIC_TOOLBOX_H
//#define METRIC_TOOLBOX_H
//
///*Cuda*/
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
///*Base Class*/
//#include "registration_metric.cuh"
//
///*Launch Parameters*/
//#include "cuda_launch_parameters.h"
//
///*CUDA Custom Registration Namespace (Compiling as DLL)*/
//namespace gpu_cost_function {
//
//	/*Collection of the standard metrics available in JTA. The dilation metric from direct-jta is the standard algorithm
//	Also included are a L1 intensity metric,  and the Mahfouz metric*/
//	class MetricToolbox : public RegistrationMetric {
//	public:
//		/*Constructor & Destructor*/
//		  MetricToolbox(int width, int height);
//		  MetricToolbox();
//
//		/*Compute Dilation Metrics*/
//		  double ComputeDilationMetric(int *bounding_box_packet);
//
//		/*Compute Mahfouz Metric*/
//		  double ComputeMahfouzMetric(int *bounding_box_packet);
//
//		/*Set/Get Dilation*/
//		  void SetDilation(int dilation);
//		  int GetDilation();
//
//		/*Set/Get "Is the Silhouette Black?"*/
//		  void SetBlackSilhouette(bool black_silhouette);
//		  bool GetBlackSilhouette();
//
//	private:
//		/*Dilation Constant*/
//		int dilation_;
//
//		/*Is the Silhouette Black?*/
//		bool black_comp_silhouette_;
//
//		/*CUDA Kernel Launch Parameters*/
//		dim3 dim_block_image_processing_;
//		dim3 dim_grid_image_processing_;
//
//	};
//
//}
//
//#endif /* METRIC_TOOLBOX_H */