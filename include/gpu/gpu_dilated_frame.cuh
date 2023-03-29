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
		  GPUDilatedFrame(int width, int height,
			int gpu_device,
			unsigned char* host_dilated_image,
			int dilation);
		  GPUDilatedFrame();
		  ~GPUDilatedFrame();

		/*Get Dilation Parameters*/
		  int GetDilation();

	private:
		/*Dilation*/
		int dilation_;
	};
}

#endif /* GPU_DILATED_FRAME_H */