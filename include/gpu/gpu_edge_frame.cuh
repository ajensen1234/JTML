#ifndef GPU_EDGE_FRAME_H
#define GPU_EDGE_FRAME_H

/*GPU Frame Class*/
#include "gpu/gpu_frame.cuh"

/*CUDA Custom Registration Namespace (Compiling as DLL)*/
namespace gpu_cost_function {
	/*Edge GPU Frame Class*/
	class GPUEdgeFrame : public GPUFrame {
	public:
		/*Constructors & Destructor*/
		__declspec(dllexport) GPUEdgeFrame(int width, int height,
			int gpu_device,
			unsigned char* host_edge_image,
			int high_threshold, int low_threshold, int aperture);
		__declspec(dllexport) GPUEdgeFrame();
		__declspec(dllexport) ~GPUEdgeFrame();

		/*Get Canny Parameters*/
		__declspec(dllexport) int GetCannyHighThreshold();
		__declspec(dllexport) int GetCannyLowThreshold();
		__declspec(dllexport) int GetCannyAperture();
	private:

		/*Canny Parameters*/
		int high_threshold_;
		int low_threshold_;
		int aperture_;
	};
}

#endif /* GPU_EDGE_FRAME_H */