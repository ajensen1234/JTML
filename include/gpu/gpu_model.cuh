#ifndef GPU_MODEL_H
#define GPU_MODEL_H

/*Render Engine Class */
#include "gpu/render_engine.cuh"

/*Include opencv for image processing on rendered outputs*/
#include <opencv2/core/mat.hpp>

#include "core/preprocessor-defs.h"

/*GPU Cost Function Library Namespace (Compiling as DLL)*/
namespace gpu_cost_function {

class GPUModel {
    /*GPU Model Class:
            Resources:
                    - Two Render Engines (One for the primary camera used in
       monoplane and biplane mode and one for the secondary camera used only in
       biplane mode). As of now everything will be done on the same GPU.
                    - Strings for the model name and the type of model
                    - Bool indicating if this is the principal model
                    - Bool indicating if GPU Model was initialized correctly
       (essentially checking if the two renderers were initialized correctly).
                    - Bool indicating if monoplane or biplane (automatically
       done in the constructor). Functions:
                    - RenderPrimaryCamera(Pose model_pose) will output a
       rendered silhouette of the model into the Render Engine's device image
       cache (stored with GPU image type) at the pose location.
                    - GetPrimaryCameraRenderedImagePointer() will return a
       device pointer (aka pointer on the GPU) to the image.
                            - Biplane versions for the secondary camera of the
       above two functions also exist and will only run if initialized correctly
       and in biplane mode.
                    - Write Rendered Image Cache to png file.
                    - Get and Set methods for a variety of variables (see
       below).

    */
   public:
    /*Monoplane constructor*/
    JTML_DLL GPUModel(std::string model_name, bool principal_model, int width,
                      int height, int device_primary_cam,
                      bool use_backface_culling_primary_cam, float* triangles,
                      float* normals, int triangle_count,
                      CameraCalibration camera_calibration_primary_cam);

    /*Biplane constructor*/
    JTML_DLL GPUModel(std::string model_name, bool principal_model, int width,
                      int height, int device_primary_cam,
                      int device_secondary_cam,
                      bool use_backface_culling_primary_cam,
                      bool use_backface_secondary_cam, float* triangles,
                      float* normals, int triangle_count,
                      CameraCalibration camera_calibration_primary_cam,
                      CameraCalibration camera_calibration_secondary_cam);

    /*Default Constructor and Destructor*/
    JTML_DLL GPUModel();
    JTML_DLL ~GPUModel();

    /*Render to cache function (returns true if worked correctly)
    Primary is used in monoplane and biplane, Secondary only used in biplane*/
    JTML_DLL bool RenderPrimaryCamera(Pose model_pose);
    JTML_DLL bool RenderPrimaryCamera_RotationMatrix(
        RotationMatrix model_pose_matrix);
    JTML_DLL bool RenderSecondaryCamera(Pose model_pose);

    JTML_DLL void RenderPrimaryCameraAndWriteImage(Pose model_pose,
                                                   std::string img_name);

    /*Render DRR to cache function (returns true if worked correctly)
    Primary is used in monoplane and biplane, Secondary only used in biplane*/
    JTML_DLL bool RenderDRRPrimaryCamera(Pose model_pose, float lower_bound,
                                         float upper_bound);
    JTML_DLL bool RenderDRRSecondaryCamera(Pose model_pose, float lower_bound,
                                           float upper_bound);

    /*Get pointer to rendered image on GPU
    Primary is used in monoplane and biplane, Secondary only used in biplane*/
    JTML_DLL unsigned char* GetPrimaryCameraRenderedImagePointer();
    JTML_DLL unsigned char* GetSecondaryCameraRenderedImagePointer();

    /*Get pointer to rendered image on GPU (GPUImage)
    Primary is used in monoplane and biplane, Secondary only used in biplane*/
    JTML_DLL GPUImage* GetPrimaryCameraRenderedImage();
    JTML_DLL GPUImage* GetSecondaryCameraRenderedImage();

    /*Get the cv::Mat output off the GPU and available for analysis
    Mostly used for image analysis that will not happen on the GPU
    In the future, it might be better to do this in a parallel way if all views
    can be rendered in parallel*/
    JTML_DLL cv::Mat GetOpenCVPrimaryRenderedImage();

    /*Write Image to File for Primary or Secondary Cameras (bool indicates
    success). Include the image extension (e.g. "femur_image.png").*/
    JTML_DLL bool WritePrimaryCameraRenderedImage(std::string file_name);
    JTML_DLL bool WriteSecondaryCameraRenderedImage(std::string file_name);

    /*Set/Get Model Name*/
    JTML_DLL void SetModelName(std::string model_name);
    JTML_DLL std::string GetModelName();

    /*Is this the Principal Model?*/
    JTML_DLL bool IsPrincipalModel();

    /*Is the GPU model properly initialized?*/
    JTML_DLL bool IsInitializedCorrectly();

    /*Is the GPU model being rendered in biplane mode (two cameras) or monoplane
     * mode (one camera)?*/
    JTML_DLL bool IsBiplaneMode();

    /*Get/Set Current Pose*/
    JTML_DLL Pose GetCurrentPrimaryCameraPose();
    JTML_DLL void SetCurrentPrimaryCameraPose(Pose current_pose);
    JTML_DLL Pose GetCurrentSecondaryCameraPose();
    JTML_DLL void SetCurrentSecondaryCameraPose(Pose current_pose);

   private:
    /*Render Engines*/
    /*Render Engine for primary camera (used in both monoplane and biplane
     * mode).*/
    RenderEngine* primary_cam_render_engine_;
    /*Render Engine for secondary camera (used solely in biplane mode).*/
    RenderEngine* secondary_cam_render_engine_;

    /*Model name is the name (usually the company and a number) - this is the
    same as the text loaded to the model list on the home screen.*/
    std::string model_name_;

    /*Principal Model (the orange one on the homescreen, and the one we are
    trying to optimize. If multiple models sent to the optimizer only one can be
    the primary (curse of dimensionality).*/
    bool principal_model_;

    /*Bool indicating the GPU model was initialized correctly*/
    bool initialized_correctly_;

    /*Bool indicating biplane mode or monoplane mode - initialized via choice of
     * constructor*/
    bool biplane_mode_;

    /*Current Poses (Decided by Optimizer)
    A: Camera A (Monoplane and Biplane Mode)
    B: Camera B (Biplane Mode only)*/
    Pose current_pose_A_;
    Pose current_pose_B_;
};
}  // namespace gpu_cost_function

#endif /* GPU_MODEL_H */
