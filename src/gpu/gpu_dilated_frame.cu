/*GPU Dilated Frame Header*/
#include "gpu/gpu_dilated_frame.cuh"

/*CUDA Custom Registration Namespace (Compiling as DLL)*/
namespace gpu_cost_function {
	/*Constructors & Destructor*/
	GPUDilatedFrame::GPUDilatedFrame(int width, int height,
	                                 int gpu_device,
	                                 unsigned char* host_dilated_image,
	                                 int dilation) : GPUFrame(width, height,
	                                                          gpu_device,
	                                                          host_dilated_image) {
		/*If Initialized Correctly*/
		if (this->IsInitializedCorrectly()) {
			dilation_ = dilation;
		}
		else {
			dilation = 0;
		}
	};

	GPUDilatedFrame::GPUDilatedFrame() {
		dilation_ = 0;
	};

	GPUDilatedFrame::~GPUDilatedFrame() {
	};

	/*Get Dilation Parameters*/
	int GPUDilatedFrame::GetDilation() {
		return dilation_;
	};
}
