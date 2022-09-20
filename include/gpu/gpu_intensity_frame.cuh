#ifndef GPU_INTENSITY_FRAME_H
#define GPU_INTENSITY_FRAME_H

/*GPU Frame Class*/
#include "gpu/gpu_frame.cuh"


/*CUDA Custom Registration Namespace (Compiling as DLL)*/
namespace gpu_cost_function {
	/*Intensity GPU Frame Class*/
	class GPUIntensityFrame : public GPUFrame {
	public:
		/*Constructors & Destructor*/
		__declspec(dllexport) GPUIntensityFrame(int width, int height,
			int gpu_device,
			unsigned char* host_intensity_image,
			bool dark_silhouette, unsigned char* host_inverted_image);
		__declspec(dllexport) GPUIntensityFrame();
		__declspec(dllexport) ~GPUIntensityFrame();

		/*Set/Get Dark Silhohuette*/
		__declspec(dllexport) bool IsSilhouetteDark();
		__declspec(dllexport) void SetSilhouetteDark(bool dark_silhouette);

		/*Get pointer to the images on the GPU Images for the Intensity Image or if Dark Silhouette True
		returns a pointer to the Inverted Image*/
		__declspec(dllexport) unsigned char* GetWhiteSilhouetteDeviceImagePointer();

		/*Get pointer to the actual GPU Image*/
		__declspec(dllexport) GPUImage* GetInvertedGPUImage();

	private:
		/*GPU Image for Inverted Intensity Image*/
		GPUImage* gpu_inverted_image_;

		/*Model Silhouette Black? (Or in Case or Alpha Value, Model Makes Image Darker?)*/
		bool dark_silhouette_;
	};
}

#endif /* GPU_INTENSITY_FRAME_H */