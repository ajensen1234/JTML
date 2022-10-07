/*GPU Intensity Frame Header*/
#include "gpu/gpu_intensity_frame.cuh"

/*CUDA Custom Registration Namespace (Compiling as DLL)*/
namespace gpu_cost_function {
	/*Constructors & Destructor*/
	GPUIntensityFrame::GPUIntensityFrame(int width, int height,
	                                     int gpu_device,
	                                     unsigned char* host_intensity_image,
	                                     bool dark_silhouette, unsigned char* host_inverted_image) : GPUFrame(
		width, height,
		gpu_device,
		host_intensity_image) {
		/*Upload Inverted Image*/
		gpu_inverted_image_ = new GPUImage(width, height, gpu_device, host_inverted_image);

		/*If Initialized Correctly*/
		if (this->IsInitializedCorrectly() && gpu_inverted_image_->IsInitializedCorrectly()) {
			dark_silhouette_ = dark_silhouette;

		}
		else {
			dark_silhouette_ = false;
		}
	};

	GPUIntensityFrame::GPUIntensityFrame() {
		dark_silhouette_ = false;
	};

	GPUIntensityFrame::~GPUIntensityFrame() {
		delete gpu_inverted_image_;
	};

	/*Get pointer to the images on the GPU Images for the Intensity Image or if Dark Silhouette True
	returns a pointer to the Inverted Image*/
	unsigned char* GPUIntensityFrame::GetWhiteSilhouetteDeviceImagePointer() {
		if (dark_silhouette_) {
			return gpu_inverted_image_->GetDeviceImagePointer();
		}
		return GPUFrame::GetDeviceImagePointer();
	};

	/*Get pointer to the  GPU Images for the Inverted Image*/
	GPUImage* GPUIntensityFrame::GetInvertedGPUImage() {
		return gpu_inverted_image_;
	};

	/*Set/Get Dark Silhohuette*/
	bool GPUIntensityFrame::IsSilhouetteDark() {
		return dark_silhouette_;
	};

	void GPUIntensityFrame::SetSilhouetteDark(bool dark_silhouette) {
		dark_silhouette_ = dark_silhouette;
	};
}
