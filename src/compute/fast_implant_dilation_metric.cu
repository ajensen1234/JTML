/*GPU Metrics Header*/
// #include "compute/gpu_metrics.cuh"

// /*Cuda*/
// #include "cuda_runtime.h"
// #include "cuda.h"

// /*Grayscale Colors*/
// #include "pixel_grayscale_colors.h"

// /*Launch Parameters*/
// #include "cuda_launch_parameters.h"

#include "cuda_launch_parameters.h"
#include "fast_implant_dilation_metric.cuh"
/* U4: device-driven metric crop — fixed-max grid with early exit (if
 * x>=cropW||y>=cropH) derived from device AABB. Host AABB read removed from
 * graph path; see render_engine persistent workers. */

/*Kernels*/
__global__ void FastImplantDilationMetric_ResetPixelScoreKernel(
    int* dev_pixel_score) {
    if ((blockDim.x * blockIdx.x) + threadIdx.x == 0) {
        dev_pixel_score[0] = 0;
    }
}

__global__ void FastImplantDilationMetric_EdgeKernel_new(
    unsigned char* dev_image,
    const int* dev_bounding_box,
    int* dev_pixel_score,
    int width,
    int height,
    int dilation) {
    /*
     * Existing launch is assumed to remain 16x16:
     *
     *     dim3 block(16, 16);
     *
     * Each logical tile therefore loads 16x16 pixels into shared memory
     * and processes the interior 14x14 pixels.
     */

    if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        dev_pixel_score[0] = 0;
    }

    const int core_width = blockDim.x - 2;
    const int core_height = blockDim.y - 2;

    __shared__ int sub_left_x;
    __shared__ int sub_bottom_y;
    __shared__ int sub_right_x;
    __shared__ int sub_top_y;
    __shared__ int tiles_x;
    __shared__ int tiles_y;
    __shared__ int tile_count;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        sub_left_x = max(dev_bounding_box[0] - dilation, dilation);

        sub_bottom_y = max(dev_bounding_box[1] - dilation, dilation);

        sub_right_x = min(dev_bounding_box[2] + dilation, width - dilation - 1);

        sub_top_y = min(dev_bounding_box[3] + dilation, height - dilation - 1);

        const int crop_width = sub_right_x - sub_left_x + 1;

        const int crop_height = sub_top_y - sub_bottom_y + 1;

        if (crop_width > 0 && crop_height > 0) {
            tiles_x = (crop_width + 13) / 14;

            tiles_y = (crop_height + 13) / 14;

            tile_count = tiles_x * tiles_y;

        } else {
            tiles_x = 0;
            tiles_y = 0;
            tile_count = 0;
        }
    }

    __syncthreads();

    extern __shared__ unsigned char shared_silhouette[];

    const int local_id = threadIdx.y * blockDim.x + threadIdx.x;

    /*
     * This CUDA block acts as a worker and processes logical tiles:
     *
     * blockIdx.x
     * blockIdx.x + gridDim.x
     * blockIdx.x + 2*gridDim.x
     * ...
     */
    for (int tile = blockIdx.x; tile < tile_count; tile += gridDim.x) {
        const int tile_x = tile % tiles_x;

        const int tile_y = tile / tiles_x;

        /*
         * Same mapping as the old kernel:
         *
         * old:
         *
         * sub_left_x - 1
         *     + blockIdx.x * (blockDim.x - 2)
         *     + threadIdx.x
         *
         * except blockIdx.{x,y} are now logical tile coordinates.
         */
        const int pixel_x = sub_left_x - 1 + tile_x * core_width + threadIdx.x;

        const int pixel_y =
            sub_bottom_y - 1 + tile_y * core_height + threadIdx.y;

        /*
         * Every thread MUST write its shared-memory slot.
         *
         * The final logical tile can extend beyond the image because
         * crop dimensions usually aren't exact multiples of 14.
         *
         * Using BLACK_PIXEL here is safe because valid crop pixels have
         * at least `dilation` pixels of image margin. Therefore an
         * out-of-image shared-memory slot cannot be an actual neighbor
         * required by a valid core pixel.
         */
        const bool in_image =
            pixel_x >= 0 && pixel_x < width && pixel_y >= 0 && pixel_y < height;

        if (in_image) {
            const int projection_id = pixel_y * width + pixel_x;

            shared_silhouette[local_id] = dev_image[projection_id];
        } else {
            shared_silhouette[local_id] = BLACK_PIXEL;
        }

        /*
         * IMPORTANT:
         *
         * Every thread reaches this barrier.
         */
        __syncthreads();

        /*
         * Only the interior 14x14 threads perform edge detection.
         */
        const bool core_thread = threadIdx.x > 0 &&
            threadIdx.x < blockDim.x - 1 && threadIdx.y > 0 &&
            threadIdx.y < blockDim.y - 1;

        /*
         * The final tile may have an interior extending beyond the
         * actual crop. Do not edge-detect those padding pixels.
         *
         * This removes the old kernel's acknowledged padding artifacts.
         */
        const bool inside_crop = pixel_x >= sub_left_x &&
            pixel_x <= sub_right_x && pixel_y >= sub_bottom_y &&
            pixel_y <= sub_top_y;

        if (core_thread && inside_crop) {
            const int left = local_id - 1;

            const int right = local_id + 1;

            const int top = local_id - blockDim.x;

            const int bottom = local_id + blockDim.x;

            if (shared_silhouette[local_id] == WHITE_PIXEL &&
                (shared_silhouette[left] == BLACK_PIXEL ||
                 shared_silhouette[right] == BLACK_PIXEL ||
                 shared_silhouette[top] == BLACK_PIXEL ||
                 shared_silhouette[bottom] == BLACK_PIXEL ||
                 shared_silhouette[bottom - 1] == BLACK_PIXEL ||
                 shared_silhouette[bottom + 1] == BLACK_PIXEL ||
                 shared_silhouette[top - 1] == BLACK_PIXEL ||
                 shared_silhouette[top + 1] == BLACK_PIXEL)) {
                const int projection_id = pixel_y * width + pixel_x;

                dev_image[projection_id] = EDGE_PIXEL;
            }
        }

        /*
         * Required before this CUDA block moves to its next logical tile.
         *
         * Otherwise some threads could begin overwriting
         * shared_silhouette while other threads are still reading it.
         */
        __syncthreads();
    }
}

__global__ void FastImplantDilationMetric_EdgeKernel(
    unsigned char* dev_image,
    int sub_left_x,
    int sub_bottom_y,
    int sub_right_x,
    int sub_top_y,
    int width,
    int dilation) {
    /*Following notes assume 16 by 16 block size.
    /*Note: THERE MIGHT BE ARTIFACTS IN THE BUFFER PADDINGS (SIDES OF IMAGES).
    SHOULD BE HARMLESS, and fixing would decrease speed.*/

    /*This section is a little complicated. We are loading in 14 by 14 sections
    of the image in 16 by 16 chunks. Therefore we have 1 pixel of padding on
    each side. This is why we must have at least one dilation. Inside of the
    image the inner "core" of the loaded tiles touch each other. Ah, shared
    memory...*/

    /*Convert thread ID to pixel ID in original image coordinates (zero based,
     * width by height sized)*/
    int correspondingPixelXToThread =
        sub_left_x - 1 + blockIdx.x * (blockDim.x - 2) + threadIdx.x;
    int correspondingPixelYToThread =
        sub_bottom_y - 1 + blockIdx.y * (blockDim.y - 2) + threadIdx.y;

    /*Make Sure in subCroppedImage (can only overflow above or to right since
    anchored at bottom left). Dilation is included to prevent a line on top
    and/or right.*/
    if (correspondingPixelXToThread <= sub_right_x + dilation &&
        correspondingPixelYToThread <= sub_top_y + dilation) {
        int localThreadId = (threadIdx.y * blockDim.x) + threadIdx.x;
        int projectionId =
            correspondingPixelYToThread * width + correspondingPixelXToThread;

        /*Now load to shared 16 by 16 array the silhouette image surrounding the
         * 14 by 14 block that is being edge detected*/
        extern __shared__ unsigned char sharedSilhouette[];
        sharedSilhouette[localThreadId] = dev_image[projectionId];
        __syncthreads();

        /* Now Only Care about inside 14 by 14 grid */
        if (0 < threadIdx.x && threadIdx.x < blockDim.x - 1 &&
            0 < threadIdx.y && threadIdx.y < blockDim.y - 1) {
            int left = localThreadId - 1;
            int right = localThreadId + 1;
            int top = localThreadId - blockDim.x;
            int bottom = localThreadId + blockDim.x;
            if (sharedSilhouette[localThreadId] == WHITE_PIXEL &&
                (sharedSilhouette[left] == BLACK_PIXEL ||
                 sharedSilhouette[right] == BLACK_PIXEL ||
                 sharedSilhouette[top] == BLACK_PIXEL ||
                 sharedSilhouette[bottom] == BLACK_PIXEL ||
                 sharedSilhouette[bottom - 1] == BLACK_PIXEL ||
                 sharedSilhouette[bottom + 1] == BLACK_PIXEL ||
                 sharedSilhouette[top - 1] == BLACK_PIXEL ||
                 sharedSilhouette[top + 1] == BLACK_PIXEL)) {
                dev_image[projectionId] = EDGE_PIXEL;
            }
        }
    }
}

__global__ void FastImplantDilationMetric_DilateKernel(
    unsigned char* dev_image,
    int width,
    int height,
    int sub_left_x,
    int sub_bottom_y,
    int sub_cropped_width,
    int dilation) {
    /*Global Thread*/
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    /*Search Direction*/
    int l = 2 * ((i % 4) / 2) - 1;
    int r = 2 * (i % 2) - 1;
    i = i / 4;
    i = (i / sub_cropped_width) * width + (i % sub_cropped_width) +
        sub_bottom_y * width + sub_left_x;

    /*Reused local variables*/
    int pixel;
    int location;

    /*If Correct Width and Height*/
    if (i < width * height) {
        if (dev_image[i] == EDGE_PIXEL) {
            for (int j = 1; j <= dilation; j++) {
                for (int k = 1; k <= dilation; k++) {
                    location = i + l * j * width + r * k;
                    pixel = dev_image[location];
                    if (pixel == WHITE_PIXEL || pixel == BLACK_PIXEL) {
                        dev_image[location] = DILATED_PIXEL;
                    }
                }
            }
        }
    }
}

__global__ void FastImplantDilationMetric_DilateKernel_new(
    unsigned char* dev_image,
    const int* dev_bounding_box,
    int width,
    int height,
    int dilation) {
    __shared__ int sub_left_x;
    __shared__ int sub_bottom_y;
    __shared__ int sub_right_x;
    __shared__ int sub_top_y;
    __shared__ int sub_cropped_width;
    __shared__ int sub_cropped_height;
    __shared__ int work_count;

    if (threadIdx.x == 0) {
        sub_left_x = max(dev_bounding_box[0] - dilation, dilation);

        sub_bottom_y = max(dev_bounding_box[1] - dilation, dilation);

        sub_right_x = min(dev_bounding_box[2] + dilation, width - dilation - 1);

        sub_top_y = min(dev_bounding_box[3] + dilation, height - dilation - 1);

        sub_cropped_width = sub_right_x - sub_left_x + 1;

        sub_cropped_height = sub_top_y - sub_bottom_y + 1;

        if (sub_cropped_width > 0 && sub_cropped_height > 0) {
            /*
             * Original kernel creates four workers per crop pixel:
             *
             * (-x,-y)
             * (-x,+y)
             * (+x,-y)
             * (+x,+y)
             */
            work_count = 4 * sub_cropped_width * sub_cropped_height;
        } else {
            work_count = 0;
        }
    }

    __syncthreads();

    for (int work = blockIdx.x * blockDim.x + threadIdx.x; work < work_count;
         work += blockDim.x * gridDim.x) {
        /*
         * Preserve the original four-direction mapping.
         */
        const int direction = work % 4;

        const int l = 2 * ((direction / 2)) - 1;

        const int r = 2 * (direction % 2) - 1;

        const int crop_index = work / 4;

        const int crop_x = crop_index % sub_cropped_width;

        const int crop_y = crop_index / sub_cropped_width;

        const int pixel_x = sub_left_x + crop_x;

        const int pixel_y = sub_bottom_y + crop_y;

        const int image_index = pixel_y * width + pixel_x;

        if (dev_image[image_index] != EDGE_PIXEL) {
            continue;
        }

        for (int j = 1; j <= dilation; ++j) {
            for (int k = 1; k <= dilation; ++k) {
                const int location = image_index + l * j * width + r * k;

                const unsigned char pixel = dev_image[location];

                if (pixel == WHITE_PIXEL || pixel == BLACK_PIXEL) {
                    dev_image[location] = DILATED_PIXEL;
                }
            }
        }
    }
}

__global__ void FastImplantDilationMetric_DifferenceKernel(
    unsigned char* dev_image,
    unsigned char* dev_comparison_image,
    int* result,
    int width,
    int height,
    int diff_kernel_left_x,
    int diff_kernel_bottom_y,
    int diff_kernel_cropped_width) {
    /*Global Thread*/
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    /*Convert to Subsize*/
    i = (i / diff_kernel_cropped_width) * width +
        (i % diff_kernel_cropped_width) + diff_kernel_bottom_y * width +
        diff_kernel_left_x;

    /*Storage Container for Loaded Pixel*/
    int pixel;

    /*If Correct Width and Height*/
    if (i < width * height) {
        pixel = dev_image[i];
        if (pixel == DILATED_PIXEL || pixel == EDGE_PIXEL) {
            if (dev_comparison_image[i] == WHITE_PIXEL) {
                atomicAdd(&result[0], 1);
            } else {
                atomicSub(&result[0], 1);
            }
        }
    }
}

__global__ void FastImplantDilationMetric_DifferenceKernel_new(
    unsigned char* dev_image,
    unsigned char* dev_comparison_image,
    int* result,
    const int* dev_bounding_box,
    int width,
    int height,
    int dilation) {
    __shared__ int left;
    __shared__ int bottom;
    __shared__ int right;
    __shared__ int top;
    __shared__ int crop_width;
    __shared__ int crop_height;
    __shared__ int pixel_count;

    if (threadIdx.x == 0) {
        /*
         * These are exactly the crop rules currently calculated by the host
         * for DifferenceKernel.
         */
        left = max(dev_bounding_box[0] - dilation, 0);

        bottom = max(dev_bounding_box[1] - dilation, 0);

        right = min(dev_bounding_box[2] + dilation, width - 1);

        top = min(dev_bounding_box[3] + dilation, height - 1);

        crop_width = right - left + 1;

        crop_height = top - bottom + 1;

        if (crop_width > 0 && crop_height > 0) {
            pixel_count = crop_width * crop_height;
        } else {
            pixel_count = 0;
        }
    }

    __syncthreads();

    for (int crop_index = blockIdx.x * blockDim.x + threadIdx.x;
         crop_index < pixel_count;
         crop_index += blockDim.x * gridDim.x) {
        const int crop_x = crop_index % crop_width;

        const int crop_y = crop_index / crop_width;

        const int pixel_x = left + crop_x;

        const int pixel_y = bottom + crop_y;

        const int image_index = pixel_y * width + pixel_x;

        const unsigned char pixel = dev_image[image_index];

        if (pixel == DILATED_PIXEL || pixel == EDGE_PIXEL) {
            if (dev_comparison_image[image_index] == WHITE_PIXEL) {
                atomicAdd(result, 1);
            } else {
                atomicSub(result, 1);
            }
        }
    }
}

__global__ void ResetDistanceTransformScoreKernel(
    int* dev_distance_transform_score_) {
    dev_distance_transform_score_[0] = 0;
}
namespace gpu_cost_function {

/*Computes DIRECT-JTA Dilation Metric Very Quickly*/
double GPUMetrics::FastImplantDilationMetric(
    GPUImage* rendered_image,
    GPUDilatedFrame* comparison_frame,
    int dilation) {
    const int height = rendered_image->GetFrameHeight();

    const int width = rendered_image->GetFrameWidth();

    unsigned char* image = rendered_image->GetDeviceImagePointer();

    const int* dev_bounding_box = rendered_image->GetDeviceBoundingBox();

    /*
     * Reset score.
     */
    // FastImplantDilationMetric_ResetPixelScoreKernel<<<1,
    // 1>>>(dev_pixel_score_);

    /*
     * Edge detection:
     *
     * 16x16 loaded tile -> 14x14 useful core.
     */
    dim3 edge_block(16, 16);

    FastImplantDilationMetric_EdgeKernel_new<<<
        edge_grid_,
        edge_block,
        edge_threads * sizeof(unsigned char)>>>(
        image, dev_bounding_box, dev_pixel_score_, width, height, dilation);

    /*
     * Dilate edge.
     */
    FastImplantDilationMetric_DilateKernel_new<<<
        dilate_grid_,
        threads_per_block>>>(image, dev_bounding_box, width, height, dilation);

    /*
     * Calculate overlap score.
     */
    FastImplantDilationMetric_DifferenceKernel_new<<<
        difference_grid_,
        threads_per_block>>>(
        image,
        comparison_frame->GetDeviceImagePointer(),
        dev_pixel_score_,
        dev_bounding_box,
        width,
        height,
        dilation);

    /*
     * This is still the one FastMetric D2H.
     *
     * Leave this alone for now. We'll eliminate/merge score D2Hs separately.
     */
    cudaMemcpy(
        pixel_score_, dev_pixel_score_, sizeof(int), cudaMemcpyDeviceToHost);

    return -1.0 * pixel_score_[0];
};

}  // namespace gpu_cost_function
