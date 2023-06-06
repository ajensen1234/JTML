#ifndef GPU_INTENSITY_FRAME_H
#define GPU_INTENSITY_FRAME_H

/*GPU Frame Class*/
#include "gpu/gpu_frame.cuh"
#include "core/preprocessor-defs.h"


/*CUDA Custom Registration Namespace (Compiling as DLL)*/
namespace gpu_cost_function {
	/*Intensity GPU Frame Class*/
	class GPUIntensityFrame : public GPUFrame {
	public:
		/*Constructors & Destructor*/
		JTML_DLL GPUIntensityFrame(int width, int height,
			int gpu_device,
			unsigned char* host_intensity_image,
			bool dark_silhouette, unsigned char* host_inverted_image);
		JTML_DLL GPUIntensityFrame();
		JTML_DLL ~GPUIntensityFrame();

		/*Set/Get Dark Silhohuette*/
		JTML_DLL bool IsSilhouetteDark();
		JTML_DLL void SetSilhouetteDark(bool dark_silhouette);

		/*Get pointer to the images on the GPU Images for the Intensity Image or if Dark Silhouette True
		returns a pointer to the Inverted Image*/
		JTML_DLL unsigned char* GetWhiteSilhouetteDeviceImagePointer();

		/*Get pointer to the actual GPU Image*/
		JTML_DLL GPUImage* GetInvertedGPUImage();

	private:
		/*GPU Image for Inverted Intensity Image*/
		GPUImage* gpu_inverted_image_;

		/*Model Silhouette Black? (Or in Case or Alpha Value, Model Makes Image Darker?)*/
		bool dark_silhouette_;
	};
}

#endif /* GPU_INTENSITY_FRAME_H */
