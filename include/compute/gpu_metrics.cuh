#ifndef GPU_METRICS_H
#define GPU_METRICS_H

/*Cuda*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*GPU Frame/Model*/
#include "compute/bank_state.cuh"
#include "compute/gpu_dilated_frame.cuh"
#include "compute/gpu_edge_frame.cuh"
#include "compute/gpu_frame.cuh"
#include "compute/gpu_heatmaps.cuh"
#include "compute/gpu_intensity_frame.cuh"
#include "compute/gpu_model.cuh"
/*Pose Matrix Class*/
#include "compute/pose_matrix.h"
#include "domain/preprocessor-defs.h"

/*CUDA Custom Registration Namespace (Compiling as DLL)*/
namespace gpu_cost_function {
struct BankState;  // non-owning compatibility view; allocation arrives in a
                   // later U12 stage
struct EvaluationContext;  // U1: primary executed type

/*Class of GPU Metrics*/
class GPUMetrics {
public:
    /*GPU Metrics Constructor/Destuctor*/
    JTML_DLL GPUMetrics();
    JTML_DLL ~GPUMetrics();

    /*Gets if Initialized Correctly*/
    JTML_DLL bool IsInitializedCorrectly();

    /*Computes DIRECT-JTA Dilation Metric Very Quickly
    The score returned is: after dilating the rendered image, if the comparison
    frame (which is assumed to also be dilated) and dilated rendered frame
    overlap at a pixel, -1 is added to the score. If the dilated rendered frame
    has a dilated edge at a pixel but the comparison image doesn't, then 1 is
    added to the score. Can be normalized so that the minimum is zero by adding
    the sum of the white pixels in the comparison image (which can be
    precomputed). This function is for implants. THIS FUNCTION DOES NOT HAVE AN
    ERROR CHECK*/
    JTML_DLL double FastImplantDilationMetric(
        GPUImage* rendered_image,
        GPUDilatedFrame* comparison_frame,
        int dilation);
    /* U12 enqueue/complete path. */
    JTML_DLL cudaError_t EnqueueFastImplantDilationMetric(
        GPUImage* rendered_image,
        GPUDilatedFrame* comparison_frame,
        int dilation,
        cudaStream_t stream);
    JTML_DLL double CompleteFastImplantDilationMetric(cudaStream_t stream);
    /* Compatibility explicit-stream wrapper: enqueue + complete. */
    JTML_DLL double FastImplantDilationMetric(
        GPUImage* rendered_image,
        GPUDilatedFrame* comparison_frame,
        int dilation,
        cudaStream_t stream);
    /*Computes Version of Mahfouz Metric Very Quickly (Subtle Changes and also
    Not Using Simulated Annealing Obviously...) The score returned is: This
    function is for implants. THIS FUNCTION DOES NOT HAVE AN ERROR CHECK*/
    JTML_DLL double ImplantMahfouzMetric(
        GPUImage* rendered_image,
        GPUDilatedFrame* comparison_dilated_frame,
        GPUIntensityFrame* comparison_intensity_frame);

    /*Computes Sum of White Pixels in Image*/
    JTML_DLL int ComputeSumWhitePixels(GPUImage* image, cudaError* error);

    /*Edge Detect Rendered Silhouette from GPU Model's GPU Image (Returns true
    if no error) Edge detected version is spit back out to the GPU Image on the
    Implant Model Can technically use on any image, just know it only marks the
    border between white pixels and black pixels*/
    JTML_DLL bool EdgeDetectRenderedImplantModel(
        GPUImage* rendered_model_image);

    /*Takes Black and White Edge Detected Image and Dilates
    Note like Edge Detection this is done in place. (Returns true if no error)*/
    JTML_DLL bool DilateEdgeDetectedImage(
        GPUImage* edge_detected_image,
        int dilation);

    /*Takes L_{1,1} norm of the difference of images (matrices) A and B.
    Note L_{1,1} norm is simply the sum of the absolute value of the difference
    of each element of A and B. E.g. for 2x2 matrices it is |A_{1,1} - B_{1,1}|
    + |A_{1,2} - B_{1,2}| + |A_{2,1} - B_{2,1}| + |A_{2,2} - B_{2,2}| THIS
    FUNCTION DOES NOT HAVE AN ERROR CHECK WARNING: ASSUMES IMAGES HAVE SAME
    DIMENSIONS!!!!*/
    JTML_DLL double L_1_1_MatrixDifferenceNorm(
        GPUImage* image_A,
        GPUImage* image_B);

    /*IOU (Jaccard Index) is the sum of the interesection of the white spaces
    (non-zero elements) in the two images (matrices) A and B divided by the
    union of the white spaces in these same two matrices.
    THIS FUNCTION DOES NOT HAVE AN ERROR CHECK
    WARNING: ASSUMES IMAGES HAVE SAME DIMENSIONS!!!!*/
    JTML_DLL double IOU(GPUImage* image_A, GPUImage* image_B);

    JTML_DLL double DistanceMapMetric(
        GPUImage* projected_image,
        GPUFrame* distance_map,
        int dilation);
    /* U12 enqueue/complete path. */
    JTML_DLL cudaError_t EnqueueDistanceMapMetric(
        GPUImage* projected_image,
        GPUFrame* distance_map,
        int dilation,
        cudaStream_t stream);
    JTML_DLL double CompleteDistanceMapMetric(cudaStream_t stream);
    /* Compatibility explicit-stream wrapper: enqueue + complete. */
    JTML_DLL double DistanceMapMetric(
        GPUImage* projected_image,
        GPUFrame* distance_map,
        int dilation,
        cudaStream_t stream);

    /* U1: explicit EvaluationContext overloads — primary design. */
    JTML_DLL cudaError_t EnqueueFastImplantDilationMetric(
        GPUImage* rendered_image,
        GPUDilatedFrame* cf,
        int dilation,
        EvaluationContext& ctx);
    JTML_DLL cudaError_t EnqueueDistanceMapMetric(
        GPUImage* projected_image,
        GPUFrame* dm,
        int dilation,
        EvaluationContext& ctx);

    JTML_DLL double CurvatureHeatmapMetric(
        GPUImage* projected_image,
        GPUHeatmap* gpu_heatmap);

    JTML_DLL void AllocateCurvatureHausdorfScore(int num_keypoints);

    /* U12 Stage 2: bind non-owning bank metadata and an optional execution
     * stream. Legacy metric wrappers remain unchanged in this compatibility
     * slice; later stages consume these seams for explicit bank execution. */
    JTML_DLL void SetActiveBank(BankState* bank);
    /* Returns false and restores bank 0 when the supplied view is incomplete.
     * SetActiveBank remains void for the Stage 2 compatibility API. */
    JTML_DLL bool TrySetActiveBank(BankState* bank);
    JTML_DLL void SetExecutionStream(cudaStream_t stream);
    JTML_DLL BankState* GetActiveBank() const;
    JTML_DLL cudaStream_t GetExecutionStream() const;

private:
    /*Integer for Pinned Memory if Metric Counts Pixels on GPU (as in dilation
    metric) This is often used to compute the metric so we include it in the
    base class definition.*/
    /*Initialize Pinned Memory for Slightly Faster Transfer*/
    int* pixel_score_;

    /*Gpu Counterpart of the pixel score*/
    int* dev_pixel_score_;

    /*Integers for Pinned Memory for IOU calculation(intermediary values).*/
    int* intersection_score_;
    int* union_score_;

    /*GPU counterparts for IOU intermediary values*/
    int* dev_intersection_score_;
    int* dev_union_score_;

    /*Initialized Correctly Variable Check*/
    bool initialized_correctly_;

    /*Dim3 Cuda Holders for launch kernels*/
    dim3 dim_block_image_processing_;
    dim3 dim_grid_image_processing_;

    /*Counts Number of White Pixels in  Image*/
    int white_pix_count_;

    /*Gpu Counterpart of white pixel count*/
    int* dev_white_pix_count_;

    /*Counts the number of edge pixels in an image (with GPU counterpart)*/
    int* edge_pixels_count_;
    int* dev_edge_pixels_count_;

    // Distance map count total (with GPU counterpart)
    int* distance_map_score_;
    int* dev_distance_map_score_;

    // Curvature heatmap score (going to be min distance)
    int* curvature_hausdorf_score_ = nullptr;
    int* dev_curvature_hausdorf_score_ = nullptr;

    /* U12 Stage 2/4A: non-owning compatibility metadata. */
    BankState* active_bank_ = nullptr;
    cudaStream_t execution_stream_ = nullptr;
    MetricBuffers bank0_metrics_{};
    bool bank0_metrics_captured_ = false;

    void CaptureBank0Metrics();
    void RestoreBank0Metrics();
    bool BindMetricBank(const BankState& bank);
};
}  // namespace gpu_cost_function
#endif /*GPU_METRICS_H*/
