#ifndef GPU_DILATED_FRAME_H
#define GPU_DILATED_FRAME_H

/*GPU Frame Class*/
#include "gpu/gpu_frame.cuh"
#include "core/preprocessor-defs.h"

/*CUDA Custom Registration Namespace (Compiling as DLL)*/
namespace gpu_cost_function {
	/*Dilated GPU Frame Class*/
	class GPUDilatedFrame : public GPUFrame {
	public:
		/*Constructors & Destructor*/
		JTML_DLL GPUDilatedFrame(int width, int height,
			int gpu_device,
			unsigned char* host_dilated_image,
			int dilation);
		JTML_DLL GPUDilatedFrame();
		JTML_DLL ~GPUDilatedFrame();

		/*Get Dilation Parameters*/
		JTML_DLL int GetDilation();

	private:
		/*Dilation*/
		int dilation_;
	};
}

#endif /* GPU_DILATED_FRAME_H */
