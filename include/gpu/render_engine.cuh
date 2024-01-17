#ifndef RENDER_ENGINE_H
#define RENDER_ENGINE_H

/*Cuda*/
#include <cuda_runtime.h>

#include "device_launch_parameters.h"

/*Camera Principal Calibration*/
#include <string>

#include "camera_calibration.h"

/*Launch Parameters*/
#include "cuda_launch_parameters.h"

/*GPU Image Class*/
#include "gpu/gpu_image.cuh"

/*Standard Library*/
#include <iostream>
#include <opencv2/core/mat.hpp>

#include "core/preprocessor-defs.h"
/*CUDA Custom Registration Namespace (Compiling as DLL)*/
namespace gpu_cost_function {
/*Pose Structure to Store Model Pose (6 D.O.F. - orientation and location)*/
struct Pose {
    JTML_DLL Pose(float x_location, float y_location, float z_location,
                  float x_angle, float y_angle, float z_angle);
    JTML_DLL Pose();
    float x_location_, y_location_, z_location_, x_angle_, y_angle_, z_angle_;
};

/*Rotation Matrix structure to store ZXY Rotation Matrix*/
struct RotationMatrix {
    JTML_DLL RotationMatrix(float rotation_00, float rotation_01,
                            float rotation_02, float rotation_10,
                            float rotation_11, float rotation_12,
                            float rotation_20, float rotation_21,
                            float rotation_22);
    JTML_DLL RotationMatrix();
    float rotation_00_, rotation_01_, rotation_02_, rotation_10_, rotation_11_,
        rotation_12_, rotation_20_, rotation_21_, rotation_22_;
};

/*CUDA Based Rendering Engine for Model Silhouette*/
class RenderEngine {
   public:
    /*Constructor & Destructor*/
    JTML_DLL RenderEngine(int width, int height, int device,
                          bool use_backface_culling, float *triangles,
                          float *normals, int triangle_count,
                          CameraCalibration camera_calibration);
    JTML_DLL RenderEngine();
    JTML_DLL ~RenderEngine();

    /*Set Pose of Model*/
    JTML_DLL void SetPose(Pose model_pose);
    JTML_DLL void SetRotationMatrix(RotationMatrix model_rotation_matrix);

    /*Write a .png to Location of Device Image*/
    JTML_DLL bool WriteImage(std::string file_name);

    /*Render (Opaque) Image to GPUImage renderer_output_'s dev_image_ via Kernel
     * calls (CUDA)*/
    JTML_DLL cudaError_t Render();

    /*Create an object that returns the pointer to the opencv Mat object*/
    JTML_DLL cv::Mat GetcvMatImage();

    /*Render DRR (Digitally Reconstructed Radiograph) to GPUImage
    renderer_output_'s dev_image_ via Kernel calls (CUDA) Lower Bound Variable:
    All line integrals below this value will be marked as 0 in the intensity
    display. Upper Bound Variable: All line integrals above this value will be
    marked as 255 in the intensity display. The remaning line integrals that
    belong to the interval [lower_bound,upper_bound] will be linearly
    interpolated and then converted to an uchar (i.e. an int from 0 to 255)
    using (uchar)(255*(value-lower_bound)/(upper_bound-lower_bound)). Note: 0 <=
    lower_bound <= upper_bound. */
    JTML_DLL cudaError_t RenderDRR(float lower_bound, float upper_bound);

    /*Get Pointer to Rendererd GPU Image*/
    JTML_DLL GPUImage *GetRenderOutput();

    /*Is the Render Engine properly initialized?*/
    JTML_DLL bool IsInitializedCorrectly();

   private:
    /*Host (CPU) Variables*/

    /*X-ray size with dilation padding on each of the four borders*/
    int width_;
    int height_;

    /*Render with back face culling*/
    bool use_backface_culling_;

    /*Triangles in model*/
    int triangle_count_;

    /*Model Location and Orientation (Z-X-Y Euler Angles), and Corresponding
     * Rotation Matrix*/
    Pose model_pose_;
    RotationMatrix model_rotation_mat_;

    /*Check to see if too many fragments*/
    bool fragment_overflow_;

    /*Camera calibration parameters*/
    CameraCalibration camera_calibration_;

    /*Intermediary Calculations to Speed Rendering*/
    float pix_conversion_x_;
    float pix_conversion_y_;
    float dist_over_pix_pitch_;
    float fx_, fy_, cx_, cy_;

    /*Fragment Fill (Number of Fragments to Test for Fill)*/
    int *fragment_fill_;

    /*Device (GPU) Variables*/

    /*Pointer to Container for the Image and Bounding Box (See GPU Image
     * Class)*/
    GPUImage *renderer_output_;

    /*Device Pointer to Array (Same Size as Image) of Floats that represent
    values used to compute DRR Each value is the amount of z translation a line
    from the origin to a pixel spends inside a model. To compute the line
    integral, simply take the value, divide by the principal distance and then
    myltiply by the norm of the 3D pixel location (in world coordinates)*/
    float *dev_z_line_values_;

    /*Device pointer to array of the transformed (rotated and translated) world
    vertices for the triangles only at the z values. This is used in the DRR
    render method and has length equal to 3 * the # of triangles.*/
    float *dev_transf_vertex_zs_;

    /*Device pointer to array of booleans indicating if a line to any point in
    transformed triangle is tangent (orthogonal to the normal). This is computed
    when checking for backface culling (if dotproduct with normal == 0) and is
    used in DRR rendering to skip adding the distance to the origin from the
    intersection since there is no line integral density because the line is
    simply to this transformed triangle and doesn't actually enter the model.
    This array has size equal to the # of triangles. TRUE if tangent, else
    FALSE*/
    bool *dev_tangent_triangle_;

    /*The triangle coordinates in millimeters loaded from the STL file.
    Each triangle is represented as a 9-tuple in the following order: x_1, y_1,
    z_ 1, x_2, y_2, z_ 2, x_3, y_3, z_ 3 where Vertex 1: (x_1, y_1, z_1) Vertex
    2: (x_2, y_2, z_2) Vertex 3: (x_3, y_3, z_3) Therefore the size of this
    array is 9 * triangle_count. Note we are using a right-hand coordinate
    system.*/
    float *dev_triangles_;

    /*The triangle normals in millimeters loaded from the STL file.
    Each triangle has a 3-tuple normal: N_x, N_y, N_z.
    Therefore the size of this array is 3 * triangle_count.
    */
    float *dev_normals_;

    /*True if the triangle is facing away from the camera (and thus we don't
    render it by making the fragment size = 1 (of course this really hsould be 0
    but I think it breaks the prefix search if this is the case...)). This is
    tested by check that the dotproduct between the normal and the first
    triangle index is < 0. The size of this array is triangle_count.
    */
    bool *dev_backface_;

    /*The projected triangle coordinates in pixels.
    Each triangle is represented as a 6-tuple in the following order x_1', y_1',
    x_2', y_2', x_3', y_3' where Vertex 1: (x_1', y_1') Vertex 2: (x_2', y_2')
    Vertex 3: (x_3', y_3')
    Therefore the size of this array is 6 * triangle_count.
    Note pixel coordinates are zero based at the bottom-left corner, and placed
    using the calibration parameters. For more details see
    https://en.wikipedia.org/wiki/Pinhole_camera_model (7/7/2016).*/
    float *dev_projected_triangles_;

    /*Snapped projected triangle coordinates to nearest integer,
    Same format as dev_screen_triangles. */
    int *dev_projected_triangles_snapped_;

    /*Bounding boxes on screen in pixels for each triangle.
    Each bounding box is represented as a 4-tuple in the follwing order LX, BY,
    RX, TY where LX: left-most x BY: bottom y RX: right-most x TY: top y
    Therefore the size of this array is 4 * triangle_count.
    Again note that we are using 0-based coordinates for the pixels.*/
    int *dev_bounding_box_triangles_;

    /*The size (number of pixels) in the bounding boxes for each triangle.
    The size of this array is simply the triangle_count.*/
    int *dev_bounding_box_triangles_sizes_;

    /*The exclusive (0-based) prefix sum of the
    dev_bounding_box_triangles_sizes_. This is clearly also of size
    triangle_count.*/
    int *dev_bounding_box_triangles_sizes_prefix_;

    /*Device version of bounding box and fragment fill. For more details see the
     * host versions*/
    int *dev_bounding_box_;
    int *dev_fragment_fill_;

    /*The container for the stride prefixes (every 256th of the fragment count)
    Could potentially overflow (highly unlikely) so need to do an error check.
    Allocated at maximum_stride_size (currently 10,000,000 (40 MB) should be
    enough for 2.56 billion fragments). CUDA will allow about 500 billion
    fragments to be processed so that won't fail first (though also needs error
    check). */
    int *dev_stride_prefixes_;

    /*CUB Variables*/
    void *dev_cub_storage_;
    size_t cub_storage_bytes_;

    /*CUDA API Initialization (Allocation, etc.) Must return cudaSuccess to
     * proceed to Render, else initialization marked as failure.*/
    cudaError_t InitializeCUDA(float *triangles, float *normals, int device);

    /*Free CUDA*/
    void FreeCuda();

    /*Initialized Render Ending Correctly Check*/
    bool initialized_correctly_;

    /*Kernel Launch Sizes*/
    dim3 dim_grid_triangles_;
    dim3 dim_grid_vertices_;
    dim3 dim_grid_bounding_box_;
    dim3 dim_grid_fill_;
};
}  // namespace gpu_cost_function
#endif /* RENDER_ENGINE_H */
