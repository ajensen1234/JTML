#ifndef GPU_DILATED_FRAME_H
#define GPU_DILATED_FRAME_H

/*GPU Frame Class*/
#include "gpu/gpu_frame.cuh"

/*CUDA Custom Registration Namespace (Compiling as DLL)*/
namespace gpu_cost_function {
	/*Dilated GPU Frame Class*/
	class GPUDilatedFrame : public GPUFrame {
	public:
		/*Constructors & Destructor*/
		__declspec(dllexport) GPUDilatedFrame(int width, int height,
			int gpu_device,
			unsigned char* host_dilated_image,
			int dilation);
		__declspec(dllexport) GPUDilatedFrame();
		__declspec(dllexport) ~GPUDilatedFrame();

		/*Get Dilation Parameters*/
		__declspec(dllexport) int GetDilation();

	private:
		/*Dilation*/
		int dilation_;
	};
}

#endif /* GPU_DILATED_FRAME_H */