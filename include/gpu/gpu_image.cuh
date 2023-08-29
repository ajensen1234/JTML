#ifndef GPU_IMAGE_H
#define GPU_IMAGE_H

/*Standard Library*/
#include <string>
#include <iostream>
#include "core/preprocessor-defs.h"

/*GPU Cost Function Library Namespace (Compiling as DLL)*/
namespace gpu_cost_function {
	/*Image on Device*/
	class GPUImage {
	public:
		/*If successful, uploads blank image
		else, DeviceImage marked as not having uploaded to GPU*/

		JTML_DLL GPUImage(int width, int height, int gpu_device);
		/*If successful, uploads host_image
		else, DeviceImage marked as not having uploaded to GPU*/
		JTML_DLL GPUImage(int width, int height, int gpu_device, unsigned char* host_image);
		/*Default constructor. DeviceImage marked as not having uploaded to GPU*/
		JTML_DLL GPUImage();

		/*Destructor*/
		JTML_DLL ~GPUImage();

		/*Upload Blank Image to Device (returns true if successful)*/
		JTML_DLL bool UploadBlankImageToGPU(int width, int height);
		/*Upload Image to Device (returns true if successful)*/
		JTML_DLL bool UploadImageToGPU(int width, int height, unsigned char* host_image);

		/*Remove Image from Device (returns true if successful)*/
		JTML_DLL bool RemoveImageFromGPU();

		/*Check to See if Image is On the Device*/
		JTML_DLL bool CheckImageOnGPU();

		/*Get Pointer to Device (GPU) Image*/
		JTML_DLL unsigned char* GetDeviceImagePointer();

		/*Get Pointer to Bounding Box*/
		JTML_DLL int* GetBoundingBox();

		/*Is the GPU model properly initialized?*/
		JTML_DLL bool IsInitializedCorrectly();

		/*Write a .png to Location of Device Image*/
		JTML_DLL bool WriteImage(std::string file_name);

		/*Get Image Size Parameters*/
		JTML_DLL int GetFrameHeight();
		JTML_DLL int GetFrameWidth();

	private:
		/*Pointer to Device Image on GPU*/
		unsigned char* dev_image_;

		/*Boolean indicating whether or not an dev_image_ exists on the GPU (device)*/
		bool image_on_gpu_;

		/*Boolean indicating if the GPU image was initialized correctly*/
		bool initialized_correctly_;

		/*Width and height integers*/
		int width_;
		int height_;

		/*Device index (aka which GPU)*/
		int device_;

		/*Bounding Box Packet for Subsequent Image Processing
		Size 4 array with:
		0: Bounding box of image - LX
		1: Bounding box of image - BY
		2: Bounding box of image - RX
		3: Bounding box of image - TY
		Default: (0, 0, width - 1, height - 1)*/
		int *bounding_box_;
	};
}

#endif /* GPU_IMAGE_H */
