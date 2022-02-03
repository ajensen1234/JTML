/*GPU Frame Header*/
#include "gpu_frame.cuh"

/*CUDA Custom Registration Namespace (Compiling as DLL)*/
namespace gpu_cost_function {

	/*If successful, uploads the four host images
	else, marked as not initialized correctly*/
	GPUFrame::GPUFrame(int width, int height,
		int gpu_device,
		unsigned char* host_image) {
		/*Try Initializing GPU Images First*/
		gpu_image_ = new GPUImage(width, height, gpu_device, host_image);

		/*If Successful*/
		if (gpu_image_->IsInitializedCorrectly()) {
				height_ = height;
				width_ = width;
				initialized_correctly_ = true;
		}
		else {
			height_ = 0;
			width_ = 0;
			initialized_correctly_ = false;
		}

	};

	/*Default constructor. Marked as not initialized correctly.*/
	GPUFrame::GPUFrame() {
		height_ = 0;
		width_ = 0;
		initialized_correctly_ = false;
		gpu_image_ = 0;

		/*GPU Images Will Auto Initialize to Default GPU Image Constructor (which is basically empty)*/
	};

	/*Default Destructor*/
	GPUFrame::~GPUFrame() {
		delete gpu_image_;
	};

	/*Get pointer to the image on the GPU Images */
	unsigned char* GPUFrame::GetDeviceImagePointer() {
		return  gpu_image_->GetDeviceImagePointer();
	};

	/*Get pointer to the actual GPU Image*/
	GPUImage* GPUFrame::GetGPUImage() {
		return gpu_image_;
	};

	/*Get Image Size Parameters*/
	int GPUFrame::GetFrameHeight() {
		return height_;
	};
	int GPUFrame::GetFrameWidth() {
		return width_;
	};

	/*Get Is Initialized Correctly*/
	bool GPUFrame::IsInitializedCorrectly() {
		return initialized_correctly_;
	};

}