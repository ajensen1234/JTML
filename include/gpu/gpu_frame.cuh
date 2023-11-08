#ifndef GPU_FRAME_H
#define GPU_FRAME_H

/*GPU Image Class*/
#include "gpu/gpu_image.cuh"
#include "core/preprocessor-defs.h"
#include <string>

/*CUDA Custom Registration Namespace (Compiling as DLL)*/
namespace gpu_cost_function {
	/*Class that holds GPU Images Representing a Frame - Very similar to GPU Image
	but not a subclass because certain functions are purposely not available. 
	There are three subclasses for frames:
		- Intensity (Can be inverted)
		- Edge Detected 
		- Dilated
	In addition to the images various metadata is stored (depending on the subclass):
		- Height/width
		- Canny Parameters
		- Dilation Value (0 if no dilation)
		- Bool marking if this is the current frame or not (the one being optimized)
		- Frame index (order determined by the mainscreen frame list)
		- Bool representing silhouette color (black? if not, must be white)
	- Various set/get functions to access the above information*/
	class GPUFrame{
	public:
		/*If successful, uploads the four host images
		else, marked as not initialized correctly*/
		JTML_DLL GPUFrame(int width, int height,
			int gpu_device,
			unsigned char* host_image);
		/*Default constructor. Marked as not initialized correctly.*/
		JTML_DLL GPUFrame();

		/*Default Destructor*/
		JTML_DLL ~GPUFrame();

		/*Get pointer to the images on the GPU Image*/
		JTML_DLL unsigned char* GetDeviceImagePointer();

		/*Get pointer to the actual GPU Image*/
		JTML_DLL GPUImage* GetGPUImage();
        JTML_DLL void WriteGPUImage(std::string file_name);

		/*Get Image Size Parameters*/
		JTML_DLL int GetFrameHeight();
		JTML_DLL int GetFrameWidth();

		/*Get Is Initialized Correctly*/
		JTML_DLL bool IsInitializedCorrectly();

	private:
		/*GPU Image*/
		GPUImage* gpu_image_;

		/*Image Size*/
		int height_;
		int width_;

		/*Model Initialized Correctly?*/
		bool initialized_correctly_;

	};
}

#endif /* GPU_FRAME_H */
