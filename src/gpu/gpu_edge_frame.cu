/*GPU Edge Frame Header*/
#include "gpu/gpu_edge_frame.cuh"

/*CUDA Custom Registration Namespace (Compiling as DLL)*/
namespace gpu_cost_function {
	/*Constructors & Destructor*/
	GPUEdgeFrame::GPUEdgeFrame(int width, int height,
	                           int gpu_device,
	                           unsigned char* host_edge_image,
	                           int high_threshold, int low_threshold, int aperture) : GPUFrame(width, height,
		gpu_device,
		host_edge_image) {
		/*If Initialized Correctly*/
		if (this->IsInitializedCorrectly()) {
			low_threshold_ = low_threshold;
			high_threshold_ = high_threshold;
			aperture_ = aperture;
		}
		else {
			low_threshold_ = 0;
			high_threshold_ = 0;
			aperture_ = 0;
		}
	};

	GPUEdgeFrame::GPUEdgeFrame() {
		low_threshold_ = 0;
		high_threshold_ = 0;
		aperture_ = 0;
	};

	GPUEdgeFrame::~GPUEdgeFrame() {
	};

	/*Get Canny Parameters*/
	int GPUEdgeFrame::GetCannyHighThreshold() {
		return high_threshold_;
	};

	int GPUEdgeFrame::GetCannyLowThreshold() {
		return low_threshold_;
	};

	int GPUEdgeFrame::GetCannyAperture() {
		return aperture_;
	};

}
