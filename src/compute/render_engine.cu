/*Render Engine Header*/
#include "compute/render_engine.cuh"

/*Cub Library (CUDA)*/
#include <cub/block/block_scan.cuh>

#include "cub/cub.cuh"
#include "cub/device/device_scan.cuh"
#include "cub/util_allocator.cuh"

/*Standard Library*/
#include <algorithm>
#include <iostream>

/*OpenCV 3.1 Library*/
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/*CUDA Custom Registration Namespace (Compiling as DLL)*/
namespace gpu_cost_function {

Pose::Pose(
    float x_location,
    float y_location,
    float z_location,
    float x_angle,
    float y_angle,
    float z_angle) {
    x_location_ = x_location;
    y_location_ = y_location;
    z_location_ = z_location;
    x_angle_ = x_angle;
    y_angle_ = y_angle;
    z_angle_ = z_angle;
}

Pose::Pose() {
    x_location_ = 0;
    y_location_ = 0;
    z_location_ = 0;
    x_angle_ = 0;
    y_angle_ = 0;
    z_angle_ = 0;
}

RotationMatrix::RotationMatrix(
    float rotation_00,
    float rotation_01,
    float rotation_02,
    float rotation_10,
    float rotation_11,
    float rotation_12,
    float rotation_20,
    float rotation_21,
    float rotation_22) {
    rotation_00_ = rotation_00;
    rotation_01_ = rotation_01;
    rotation_02_ = rotation_02;
    rotation_10_ = rotation_10;
    rotation_11_ = rotation_11;
    rotation_12_ = rotation_12;
    rotation_20_ = rotation_20;
    rotation_21_ = rotation_21;
    rotation_22_ = rotation_22;
}

RotationMatrix::RotationMatrix() {
    rotation_00_ = 1;
    rotation_01_ = 0;
    rotation_02_ = 0;
    rotation_10_ = 0;
    rotation_11_ = 1;
    rotation_12_ = 0;
    rotation_20_ = 0;
    rotation_21_ = 0;
    rotation_22_ = 1;
}

// U4: forward declarations for occupancy query in InitializeCUDA
__global__ void FillTrianglePersistentKernel(
    int* dev_nextCandidate,
    int chunkSize,
    int* dev_fragment_fill,
    int* dev_overflowFlag,
    int triangle_count,
    int* dev_bbox_triangles,
    int* dev_sizes,
    int* dev_prefix,
    unsigned char* dev_image,
    int width,
    int height,
    float* dev_projected_triangles,
    int* dev_stride_prefixes);
__global__ void StridePrefixPersistentKernel(
    int* dev_nextChunk,
    int chunkSize,
    int* dev_fragment_fill,
    int* dev_overflowFlag,
    int* dev_sizes,
    int* dev_prefix,
    int* dev_stride_prefixes,
    int triangle_count,
    int stride);

__global__ void FillTriangleKernel_new(
    int* sizes,
    int* prefix,
    int* bounding_boxes,
    unsigned char* image,
    int triangle_count,
    int width,
    int height,
    float* projected);

RenderEngine::RenderEngine(
    int width,
    int height,
    int device,
    bool use_backface_culling,
    float* triangles,
    float* normals,
    int triangle_count,
    CameraCalibration camera_calibration) {
    /*Initialize Private Host Variables*/
    width_ = width;
    height_ = height;
    use_backface_culling_ = use_backface_culling;
    triangle_count_ = triangle_count;
    fragment_overflow_ = false;
    initialized_correctly_ = true;
    camera_calibration_ = camera_calibration;
    if (camera_calibration_.type_ == "UF") {
        fx_ = -1.0f * camera_calibration_.principal_distance_ /
            camera_calibration_.pixel_pitch_;
        fy_ = -1.0f * camera_calibration_.principal_distance_ /
            camera_calibration_.pixel_pitch_;
        cx_ = static_cast<float>(width_) / 2.0f -
            camera_calibration_.principal_x_ / camera_calibration_.pixel_pitch_;
        cy_ = static_cast<float>(height) / 2.0f -
            camera_calibration_.principal_y_ / camera_calibration_.pixel_pitch_;
    } else if (camera_calibration_.type_ == "Denver") {
        fx_ = camera_calibration_.fx();
        fy_ = -camera_calibration_.fy();
        cx_ = camera_calibration_.cx();
        cy_ = height - camera_calibration_.cy();
    }
    pix_conversion_x_ = static_cast<float>(width_) / 2.0f -
        camera_calibration_.principal_x_ / camera_calibration_.pixel_pitch_;
    pix_conversion_y_ = static_cast<float>(height_) / 2.0f -
        camera_calibration_.principal_y_ / camera_calibration_.pixel_pitch_;
    dist_over_pix_pitch_ = -1.0f * camera_calibration_.principal_distance_ /
        camera_calibration_.pixel_pitch_;

    /*Initialize Kernel Launch Sizes*/
    dim_grid_triangles_ = dim3(
        ceil(sqrt(
            static_cast<double>(triangle_count_) /
            static_cast<double>(threads_per_block))),
        ceil(sqrt(
            static_cast<double>(triangle_count_) /
            static_cast<double>(threads_per_block))));
    dim_grid_vertices_ = dim3(
        ceil(sqrt(
            3.0 * triangle_count_ / static_cast<double>(threads_per_block))),
        ceil(sqrt(
            3.0 * triangle_count_ / static_cast<double>(threads_per_block))));
    dim_grid_bounding_box_ = dim3(
        ceil(sqrt(
            4.0 * triangle_count_ / static_cast<double>(threads_per_block))),
        ceil(sqrt(
            4.0 * triangle_count_ / static_cast<double>(threads_per_block))));

    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        initialized_correctly_ = false;
        return;
    }

    cudaDeviceProp props{};

    err = cudaGetDeviceProperties(&props, device);
    if (err != cudaSuccess) {
        initialized_correctly_ = false;
        return;
    }

    int blocks_per_sm = 0;

    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, FillTriangleKernel_new, threads_per_block, 0);

    if (err != cudaSuccess) {
        initialized_correctly_ = false;
        return;
    }

    fill_triangle_grid_ = props.multiProcessorCount * blocks_per_sm;

    std::cout << "fill_blocks_per_sm: " << blocks_per_sm << '\n';

    std::cout << "fill_triangle_grid_: " << fill_triangle_grid_ << '\n';

    /*Initialize Host Variables*/
    fragment_fill_ = 0;

    /*Initialize Private Device Variables*/
    dev_z_line_values_ = 0;
    dev_triangles_ = 0;
    dev_normals_ = 0;
    dev_backface_ = 0;
    dev_transf_vertex_zs_ = 0;
    dev_tangent_triangle_ = 0;
    dev_projected_triangles_ = 0;
    dev_projected_triangles_snapped_ = 0;
    dev_bounding_box_triangles_ = 0;
    dev_bounding_box_triangles_sizes_ = 0;
    dev_bounding_box_triangles_sizes_prefix_ = 0;
    dev_bounding_box_ = 0;
    dev_fragment_fill_ = 0;
    dev_stride_prefixes_ = 0;

    /*Initialize CUB Temporary Storage*/
    dev_cub_storage_ = 0;
    cub_storage_bytes_ = 0;

    /*Initialize Renderer Output*/
    renderer_output_ = 0;

    /*Initialize CUDA*/
    if (InitializeCUDA(triangles, normals, device) != cudaSuccess) {
        initialized_correctly_ = false;
    }
}

RenderEngine::RenderEngine() {
    /*Initialize Host Variables*/
    fragment_fill_ = 0;

    /*Initialize Private Device Variables*/
    dev_z_line_values_ = 0;
    dev_triangles_ = 0;
    dev_normals_ = 0;
    dev_backface_ = 0;
    dev_transf_vertex_zs_ = 0;
    dev_tangent_triangle_ = 0;
    dev_projected_triangles_ = 0;
    dev_projected_triangles_snapped_ = 0;
    dev_bounding_box_triangles_ = 0;
    dev_bounding_box_triangles_sizes_ = 0;
    dev_bounding_box_triangles_sizes_prefix_ = 0;
    dev_bounding_box_ = 0;
    dev_fragment_fill_ = 0;
    dev_stride_prefixes_ = 0;

    /*Initialize CUB Temporary Storage*/
    dev_cub_storage_ = 0;
    cub_storage_bytes_ = 0;

    /*Initialize Renderer Output*/
    renderer_output_ = 0;

    /*Default Constructor Never Initialized*/
    initialized_correctly_ = false;
}

RenderEngine::~RenderEngine() {
    /*Free CUDA*/
    FreeCuda();

    delete renderer_output_;
}

void RenderEngine::FreeCuda() {
    /*Free CUDA*/
    cudaFree(dev_z_line_values_);
    cudaFree(dev_triangles_);
    cudaFree(dev_normals_);
    cudaFree(dev_backface_);
    cudaFree(dev_transf_vertex_zs_);
    cudaFree(dev_tangent_triangle_);
    cudaFree(dev_projected_triangles_);
    cudaFree(dev_projected_triangles_snapped_);
    cudaFree(dev_bounding_box_triangles_);
    cudaFree(dev_bounding_box_triangles_sizes_);
    cudaFree(dev_bounding_box_triangles_sizes_prefix_);
    cudaFree(dev_cub_storage_);
    cudaFree(dev_bounding_box_);
    cudaFree(dev_fragment_fill_);
    cudaFree(dev_stride_prefixes_);
    /* U4: persistent worker counters */
    cudaFree(dev_nextCandidate_);
    cudaFree(dev_nextChunk_);
    cudaFree(dev_overflowFlag_);
    dev_nextCandidate_ = nullptr;
    dev_nextChunk_ = nullptr;
    dev_overflowFlag_ = nullptr;

    /*Free Host*/
    cudaFreeHost(fragment_fill_);
    cudaFreeHost(host_overflowFlag_);
    host_overflowFlag_ = nullptr;
}

cudaError_t
RenderEngine::InitializeCUDA(float* triangles, float* normals, int device) {
    /*CUDA Error Status*/
    cudaGetLastError();  // Resets Errors
    cudaError_t cudaStatus;

    /*Choose which GPU to run on, change this on a multi-GPU system.*/
    cudaSetDevice(device);

    /*Check for Errors*/
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        initialized_correctly_ = false;
        FreeCuda();
        return cudaStatus;
    }

    /*Initialize Pinned Memory for Slightly Faster Transfer*/
    cudaHostAlloc(
        (void**)&fragment_fill_, 1 * sizeof(int), cudaHostAllocDefault);

    /*Allocate GPU buffers for image, triangles.*/
    cudaMalloc((void**)&dev_z_line_values_, width_ * height_ * sizeof(float));

    cudaMalloc((void**)&dev_triangles_, triangle_count_ * 9 * sizeof(float));

    cudaMalloc((void**)&dev_normals_, triangle_count_ * 3 * sizeof(float));

    cudaMalloc((void**)&dev_backface_, triangle_count_ * sizeof(bool));

    cudaMalloc(
        (void**)&dev_transf_vertex_zs_, triangle_count_ * 3 * sizeof(float));

    cudaMalloc((void**)&dev_tangent_triangle_, triangle_count_ * sizeof(bool));

    cudaMalloc(
        (void**)&dev_projected_triangles_, triangle_count_ * 6 * sizeof(float));

    cudaMalloc(
        (void**)&dev_projected_triangles_snapped_,
        triangle_count_ * 6 * sizeof(int));

    cudaMalloc(
        (void**)&dev_bounding_box_triangles_,
        triangle_count_ * 4 * sizeof(int));

    cudaMalloc(
        (void**)&dev_bounding_box_triangles_sizes_,
        triangle_count_ * sizeof(int));

    cudaMalloc(
        (void**)&dev_bounding_box_triangles_sizes_prefix_,
        triangle_count_ * sizeof(int));

    cudaMalloc((void**)&dev_bounding_box_, 4 * sizeof(int));

    cudaMalloc((void**)&dev_fragment_fill_, 1 * sizeof(int));

    cudaMalloc(
        (void**)&dev_stride_prefixes_, maximum_stride_size * sizeof(int));

    /*Check for Errors*/
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        initialized_correctly_ = false;
        FreeCuda();
        return cudaStatus;
    }

    /*Initialize the GPU Image*/
    renderer_output_ = new GPUImage(width_, height_, device);
    renderer_output_->SetDeviceBoundingBox(dev_bounding_box_);

    /*Check for errors*/
    if (!renderer_output_->IsInitializedCorrectly()) {
        initialized_correctly_ = false;
        FreeCuda();
        delete renderer_output_;
        renderer_output_ = 0;
        return cudaErrorUnknown;
    }

    /*Before Allocating Temporary Buffer for CUB to GPU, learn the size by
     * calling the function*/
    cub::DeviceScan::ExclusiveSum(
        dev_cub_storage_,
        cub_storage_bytes_,
        dev_bounding_box_triangles_sizes_,
        dev_bounding_box_triangles_sizes_,
        triangle_count_);

    cudaMalloc(&dev_cub_storage_, cub_storage_bytes_);
    /* U4: device-driven persistent worker counters */
    cudaMalloc((void**)&dev_nextCandidate_, 1 * sizeof(int));
    cudaMalloc((void**)&dev_nextChunk_, 1 * sizeof(int));
    cudaMalloc((void**)&dev_overflowFlag_, 1 * sizeof(int));
    cudaHostAlloc(
        (void**)&host_overflowFlag_, 1 * sizeof(int), cudaHostAllocDefault);
    // Fixed grid sizing for persistent workers: min(maxBlocksPerSM*SM,
    // ceil(SAFE_CAP/256)) Compute occupancy-derived fixed upper bounds from the
    // real kernels. SAFE_CAP = maximum_stride_size * threads_per_block (max
    // fragment count).
    {
        int numSMs = 0;
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device);

        int maxActiveFill = 0;
        cudaError_t occErr1 = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveFill, FillTrianglePersistentKernel, threads_per_block, 0);

        int maxActiveStride = 0;
        cudaError_t occErr2 = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveStride,
            StridePrefixPersistentKernel,
            threads_per_block,
            0);

        // Fail initialization if occupancy/device queries fail — do not
        // silently restore hardcoded values that could produce incorrect grid
        // sizes.
        if (occErr1 != cudaSuccess || occErr2 != cudaSuccess || numSMs <= 0 ||
            maxActiveFill <= 0 || maxActiveStride <= 0) {
            initialized_correctly_ = false;
            FreeCuda();
            return cudaErrorUnknown;
        }

        // Upper bound on useful chunks per kernel type:
        //   Fill: each chunk processes threads_per_block candidates → max
        //   chunks = maximum_stride_size Stride: each chunk processes
        //   threads_per_block stride entries → max chunks = maximum_stride_size
        //   / threads_per_block
        const int maxFillChunks = maximum_stride_size;
        const int maxStrideChunks =
            (maximum_stride_size + threads_per_block - 1) / threads_per_block;

        // Occupancy-derived = maxActive * numSMs, clamped to the safe upper
        // bound.
        persistent_fill_blocks_ =
            std::min(maxActiveFill * numSMs, maxFillChunks);
        persistent_stride_blocks_ =
            std::min(maxActiveStride * numSMs, maxStrideChunks);
    }

    /*Check for Errors*/
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        initialized_correctly_ = false;
        FreeCuda();
        return cudaStatus;
    }

    /*Copy input from host memory to GPU.*/
    cudaMemcpy(
        dev_triangles_,
        triangles,
        triangle_count_ * 9 * sizeof(float),
        cudaMemcpyHostToDevice);

    cudaMemcpy(
        dev_normals_,
        normals,
        triangle_count_ * 3 * sizeof(float),
        cudaMemcpyHostToDevice);

    /*Check for Errors*/
    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess) {
        initialized_correctly_ = true;
    } else {
        initialized_correctly_ = false;
        FreeCuda();
    }
    return cudaStatus;
}

void RenderEngine::SetPose(Pose model_pose) {
    model_pose_ = model_pose;

    float cz = cos(model_pose_.z_angle_ * 3.14159265358979323846f / 180.0f);
    float sz = sin(model_pose_.z_angle_ * 3.14159265358979323846f / 180.0f);
    float cx = cos(model_pose_.x_angle_ * 3.14159265358979323846f / 180.0f);
    float sx = sin(model_pose_.x_angle_ * 3.14159265358979323846f / 180.0f);
    float cy = cos(model_pose_.y_angle_ * 3.14159265358979323846f / 180.0f);
    float sy = sin(model_pose_.y_angle_ * 3.14159265358979323846f / 180.0f);

    /* R*v = RzRxRy*v */
    model_rotation_mat_ = RotationMatrix(
        cz * cy - sz * sx * sy,
        -1.0 * sz * cx,
        cz * sy + sz * cy * sx,
        sz * cy + cz * sx * sy,
        cz * cx,
        sz * sy - cz * cy * sx,
        -1.0 * cx * sy,
        sx,
        cx * cy);
}
void RenderEngine::SetRotationMatrix(RotationMatrix model_rotation_matrix) {
    model_rotation_mat_ = model_rotation_matrix;
}

GPUImage* RenderEngine::GetRenderOutput() {
    return renderer_output_;
}

__global__ void ResetKernel(int* dev_bounding_box, int width, int height) {
    dev_bounding_box[0] = width - 1;
    /*Left Most X -> initialize with width - 1 (we are zero based) since can
     * only be brought down*/
    dev_bounding_box[1] = height - 1;
    /*Bottom Most Y -> initialize with height - 1 (we are zero based) since can
     * only be brought down*/
    dev_bounding_box[2] =
        0; /*Right Most X -> initialize with zero since can only be brought up*/
    dev_bounding_box[3] =
        0; /*Top Most Y -> initialize with zero since can only be brought up*/
}

__global__ void WorldToPixelKernel(
    float* dev_triangles,
    float* dev_projected_triangles,
    int* dev_projected_triangles_snapped,
    int vertex_count,
    float dist_over_pix_pitch,
    float pix_conversion_x,
    float pix_conversion_y,
    float x_location,
    float y_location,
    float z_location,
    RotationMatrix model_rotation_mat,
    float* dev_normals,
    bool* dev_backface,
    bool use_backface_culling,
    float fx,
    float fy,
    float cx,
    float cy,
    unsigned char* dev_image,
    int* dev_bounding_box,
    int image_width,
    int image_height) {
    const int i =
        (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    const int total_threads = gridDim.x * gridDim.y * blockDim.x;

    /*
     * Clear the previous render.
     *
     * For every iteration of this loop, adjacent threads write adjacent
     * image bytes, so the stores remain coalesced.
     */
    for (int pixel = i; pixel < image_width * image_height;
         pixel += total_threads) {
        dev_image[pixel] = 0;
    }

    /*
     * Reset model bbox.
     *
     * WorldToPixelKernel completes before BoundingBoxSizesKernel runs,
     * so no global synchronization inside this kernel is needed.
     */
    if (i == 0) {
        dev_bounding_box[0] = image_width - 1;

        dev_bounding_box[1] = image_height - 1;

        dev_bounding_box[2] = 0;

        dev_bounding_box[3] = 0;
    }

    if (i < vertex_count) {
        /*Read in Vertices*/
        float vX = dev_triangles[(3 * i)];
        float vY = dev_triangles[(3 * i) + 1];
        float vZ = dev_triangles[(3 * i) + 2];

        /*Transform (Rotate then Translate) Vertices*/
        float tX = model_rotation_mat.rotation_00_ * vX +
            model_rotation_mat.rotation_01_ * vY +
            model_rotation_mat.rotation_02_ * vZ + x_location;
        float tY = model_rotation_mat.rotation_10_ * vX +
            model_rotation_mat.rotation_11_ * vY +
            model_rotation_mat.rotation_12_ * vZ + y_location;
        float tZ = model_rotation_mat.rotation_20_ * vX +
            model_rotation_mat.rotation_21_ * vY +
            model_rotation_mat.rotation_22_ * vZ + z_location;

        /*Transform normal and compute dot product with vertex. Backface if >=
         * 0. Only do on first vertex.*/
        if (i % 3 == 0) {
            float nX = dev_normals[3 * (i / 3)];
            float nY = dev_normals[3 * (i / 3) + 1];
            float nZ = dev_normals[3 * (i / 3) + 2];
            float dotProduct = (model_rotation_mat.rotation_00_ * nX +
                                model_rotation_mat.rotation_01_ * nY +
                                model_rotation_mat.rotation_02_ * nZ) *
                    tX +
                (model_rotation_mat.rotation_10_ * nX +
                 model_rotation_mat.rotation_11_ * nY +
                 model_rotation_mat.rotation_12_ * nZ) *
                    tY +
                (model_rotation_mat.rotation_20_ * nX +
                 model_rotation_mat.rotation_21_ * nY +
                 model_rotation_mat.rotation_22_ * nZ) *
                    tZ;
            if (dotProduct >= 0) {
                dev_backface[i / 3] = true;
            } else {
                dev_backface[i / 3] = false;
            }
            if (!use_backface_culling) {
                dev_backface[i / 3] = false;
            }
        }
        // Need to change this condition - it definitely can be higher than zero
        // if you are using a different calibration setup.
        if (tZ == 0) {
            tZ = -.000001; /*Can't be above or at zero, so make very
                              small..should never happen*/
        }

        // float sX = (tX / tZ) * dist_over_pix_pitch + pix_conversion_x;
        // float sY = (tY / tZ) * dist_over_pix_pitch + pix_conversion_y;

        float sX = (tX / tZ) * fx + cx;
        float sY = (tY / tZ) * fy + cy;

        /*Store Projected Triangles Actual Location*/
        dev_projected_triangles[(2 * i)] = sX;
        dev_projected_triangles[(2 * i) + 1] = sY;

        /*Store Nearest Pixel of Projected Triangles (Round >= X.5 up to X + 1
         * and < X.5 down to X)*/
        dev_projected_triangles_snapped[(2 * i)] =
            static_cast<int>(floorf(sX + 0.5));
        dev_projected_triangles_snapped[(2 * i) + 1] =
            static_cast<int>(floorf(sY + 0.5));
    }
}

template <int BLOCK_THREADS>
__global__ void BoundingBoxSizesAndExclusiveScanKernel(
    const int* dev_bounding_box_triangles,
    int* dev_bounding_box_triangles_sizes,
    int* dev_bounding_box_triangles_sizes_prefix,
    int triangle_count,
    int* dev_bounding_box,
    const bool* dev_backface) {
    using BlockScan = cub::BlockScan<int, BLOCK_THREADS>;

    __shared__ typename BlockScan::TempStorage scan_storage;
    __shared__ int running_total;

    if (threadIdx.x == 0) {
        running_total = 0;
    }

    __syncthreads();

    /*
     * This single CUDA block walks the triangle array in
     * BLOCK_THREADS-sized tiles.
     *
     * ~12,500 triangles / 256 threads ≈ 49 iterations.
     */
    for (int base = 0; base < triangle_count; base += BLOCK_THREADS) {
        const int triangle_index = base + threadIdx.x;

        int size = 0;

        if (triangle_index < triangle_count) {
            const int bbox_index = 4 * triangle_index;

            const int left_x = dev_bounding_box_triangles[bbox_index];

            const int bottom_y = dev_bounding_box_triangles[bbox_index + 1];

            const int right_x = dev_bounding_box_triangles[bbox_index + 2];

            const int top_y = dev_bounding_box_triangles[bbox_index + 3];

            /*
             * Preserve current behavior exactly:
             * backfaces contribute one fragment.
             */
            if (!dev_backface[triangle_index]) {
                size = (1 + right_x - left_x) * (1 + top_y - bottom_y);
            } else {
                size = 1;
            }

            dev_bounding_box_triangles_sizes[triangle_index] = size;

            /*
             * Overall rendered-model bbox.
             *
             * Keep this identical to your existing
             * BoundingBoxSizesKernel for the first benchmark.
             */
            atomicMin(&dev_bounding_box[0], left_x);

            atomicMin(&dev_bounding_box[1], bottom_y);

            atomicMax(&dev_bounding_box[2], right_x);

            atomicMax(&dev_bounding_box[3], top_y);
        }

        int exclusive_prefix = 0;
        int tile_total = 0;

        /*
         * Exclusive scan of this 256-element tile.
         */
        BlockScan(scan_storage)
            .ExclusiveSum(size, exclusive_prefix, tile_total);

        const int tile_base = running_total;

        if (triangle_index < triangle_count) {
            dev_bounding_box_triangles_sizes_prefix[triangle_index] =
                tile_base + exclusive_prefix;
        }

        /*
         * CUB requires synchronization before reusing
         * scan_storage on the next iteration.
         */
        __syncthreads();

        if (threadIdx.x == 0) {
            running_total = tile_base + tile_total;
        }

        __syncthreads();
    }
}

__global__ void BoundingBoxForTrianglesKernel(
    int* dev_bounding_box_triangles,
    int* dev_projected_triangles_snapped,
    int triangle_count,
    int width,
    int height) {
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < 4 * triangle_count) {
        int j = i % 4;
        if (j == 0) /* Bottom Left X */
        {
            int value = max(
                min(min(min(dev_projected_triangles_snapped[6 * (i / 4)],
                            dev_projected_triangles_snapped[6 * (i / 4) + 2]),
                        dev_projected_triangles_snapped[6 * (i / 4) + 4]),
                    width - 1),
                0);
            dev_bounding_box_triangles[i] = value;
        } else if (j == 1) /* Bottom Left Y */
        {
            int value = max(
                min(min(min(dev_projected_triangles_snapped[6 * (i / 4) + 1],
                            dev_projected_triangles_snapped[6 * (i / 4) + 3]),
                        dev_projected_triangles_snapped[6 * (i / 4) + 5]),
                    height - 1),
                0);
            dev_bounding_box_triangles[i] = value;
        } else if (j == 2) /* Top Right X */
        {
            int value = min(
                max(max(max(dev_projected_triangles_snapped[6 * (i / 4)],
                            dev_projected_triangles_snapped[6 * (i / 4) + 2]),
                        dev_projected_triangles_snapped[6 * (i / 4) + 4]),
                    0),
                width - 1);
            dev_bounding_box_triangles[i] = value;
        } else if (j == 3) /* Top Right Y */
        {
            int value = min(
                max(max(max(dev_projected_triangles_snapped[6 * (i / 4) + 1],
                            dev_projected_triangles_snapped[6 * (i / 4) + 3]),
                        dev_projected_triangles_snapped[6 * (i / 4) + 5]),
                    0),
                height - 1);
            dev_bounding_box_triangles[i] = value;
        }
    }
}

__global__ void BoundingBoxSizesKernel(
    int* dev_bounding_box_triangles,
    int* dev_bounding_box_triangles_sizes,
    int triangle_count,
    int* dev_bounding_box,
    bool* dev_backface) {
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < triangle_count) {
        int fourI = 4 * i;

        /*Load bounding box corners (LX, BY, RX, TY) to register memory*/
        int leftX = dev_bounding_box_triangles[fourI];
        int bottomY = dev_bounding_box_triangles[fourI + 1];
        int rightX = dev_bounding_box_triangles[fourI + 2];
        int topY = dev_bounding_box_triangles[fourI + 3];

        /*Backface*/
        if (dev_backface[i] == false) {
            dev_bounding_box_triangles_sizes[i] =
                (1 + rightX - leftX) * (1 + topY - bottomY);
        } else {
            dev_bounding_box_triangles_sizes[i] = 1;
            /*In "theory" should be 0, but this leads to fragments so make 1*/
        }

        /*Store Bounding Box on Image*/
        atomicMin(&dev_bounding_box[0], leftX);
        atomicMin(&dev_bounding_box[1], bottomY);
        atomicMax(&dev_bounding_box[2], rightX);
        atomicMax(&dev_bounding_box[3], topY);
    }
}

__global__ void BoundingBoxAndSizesKernel(
    int* dev_bounding_box_triangles,
    const int* dev_projected_triangles_snapped,
    int* dev_bounding_box_triangles_sizes,
    int triangle_count,
    int width,
    int height,
    int* dev_bounding_box,
    const bool* dev_backface) {
    /*
     * IMPORTANT:
     *
     * Preserve the original mapping:
     *
     *     4 CUDA threads per triangle
     *
     * j = 0 -> LX
     * j = 1 -> BY
     * j = 2 -> RX
     * j = 3 -> TY
     */
    const int i =
        (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    const bool valid = i < 4 * triangle_count;

    /*
     * Which bbox component does this thread own?
     */
    const int j = i & 3;

    const int triangle_index = i >> 2;

    int value = 0;

    if (valid) {
        const int projected_index = 6 * triangle_index;

        if (j == 0) {
            /* LX */
            value = max(
                min(min(min(dev_projected_triangles_snapped[projected_index],
                            dev_projected_triangles_snapped
                                [projected_index + 2]),
                        dev_projected_triangles_snapped[projected_index + 4]),
                    width - 1),
                0);

        } else if (j == 1) {
            /* BY */
            value = max(
                min(min(min(dev_projected_triangles_snapped
                                [projected_index + 1],
                            dev_projected_triangles_snapped
                                [projected_index + 3]),
                        dev_projected_triangles_snapped[projected_index + 5]),
                    height - 1),
                0);

        } else if (j == 2) {
            /* RX */
            value = min(
                max(max(max(dev_projected_triangles_snapped[projected_index],
                            dev_projected_triangles_snapped
                                [projected_index + 2]),
                        dev_projected_triangles_snapped[projected_index + 4]),
                    0),
                width - 1);

        } else {
            /* TY */
            value = min(
                max(max(max(dev_projected_triangles_snapped
                                [projected_index + 1],
                            dev_projected_triangles_snapped
                                [projected_index + 3]),
                        dev_projected_triangles_snapped[projected_index + 5]),
                    0),
                height - 1);
        }

        /*
         * FillTriangle still needs the bbox array.
         *
         * Since i == 4 * triangle_index + j, this is already
         * exactly the correct output location.
         */
        dev_bounding_box_triangles[i] = value;
    }

    /*
     * Groups of four threads are always contained inside a warp because
     * 256-thread blocks and 32-thread warps are both divisible by four.
     *
     * Get the active lanes for the final partially-filled warp.
     */
    const unsigned mask = __ballot_sync(0xffffffff, valid);

    if (!valid) {
        return;
    }

    const int lane = threadIdx.x & 31;

    const int group_lane = lane & ~3;

    /*
     * Each thread in the four-thread group receives all four bbox values
     * directly from registers in the other lanes.
     *
     * No reread from dev_bounding_box_triangles[].
     */
    const int left_x = __shfl_sync(mask, value, group_lane);

    const int bottom_y = __shfl_sync(mask, value, group_lane + 1);

    const int right_x = __shfl_sync(mask, value, group_lane + 2);

    const int top_y = __shfl_sync(mask, value, group_lane + 3);

    /*
     * Only thread 0 of each four-thread triangle group performs
     * size calculation + overall bbox atomics.
     */
    if (j == 0) {
        if (!dev_backface[triangle_index]) {
            dev_bounding_box_triangles_sizes[triangle_index] =
                (1 + right_x - left_x) * (1 + top_y - bottom_y);

        } else {
            /*
             * Preserve your existing "backfaces have size 1"
             * behavior.
             */
            dev_bounding_box_triangles_sizes[triangle_index] = 1;
        }

        atomicMin(&dev_bounding_box[0], left_x);

        atomicMin(&dev_bounding_box[1], bottom_y);

        atomicMax(&dev_bounding_box[2], right_x);

        atomicMax(&dev_bounding_box[3], top_y);
    }
}

__global__ void PrepareLaunchPacketKernel(
    int* dev_fragment_fill,
    int* dev_bounding_box_triangles_sizes,
    int* dev_bounding_box_triangles_sizes_prefix,
    int triangle_count) {
    dev_fragment_fill[0] =
        dev_bounding_box_triangles_sizes[triangle_count - 1] +
        dev_bounding_box_triangles_sizes_prefix[triangle_count - 1];
}

__global__ void StridePrefixKernel(
    int stride,
    int* dev_bounding_box_triangles_sizes,
    int* dev_bounding_box_triangles_sizes_prefix,
    int* dev_stride_prefixes,
    int triangle_count) {
    /*Should be slightly more then
    (boundingBoxTrianglesSizePrefix[triangleCount - 1] +
    boundingBoxTrianglesSize[triangleCount - 1] ) / stride
    */
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    int j = i * stride;

    if (j < dev_bounding_box_triangles_sizes_prefix[triangle_count - 1] +
            dev_bounding_box_triangles_sizes[triangle_count - 1]) {
        /*Get the index for the stride elements*/
        int low = 0;
        int high = triangle_count;
        int mid = 0;
        int strideIndex = -1;

        /*Binary Search Loop*/
        while (low != high) {
            /*Calculate Mid Index*/
            mid = (low + high) / 2;

            if (dev_bounding_box_triangles_sizes_prefix[mid] <= j) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        strideIndex = high - 1;
        dev_stride_prefixes[i] = strideIndex;
    }
}

__global__ void FillTriangleKernel_new(
    int* sizes,
    int* prefix,
    int* bounding_boxes,
    unsigned char* image,
    int triangle_count,
    int width,
    int height,
    float* projected) {
    __shared__ int start_triangle;
    __shared__ int reduced_prefix[256];

    const int total = prefix[triangle_count - 1] + sizes[triangle_count - 1];

    for (int chunk = blockIdx.x;; chunk += gridDim.x) {
        const int base = chunk * blockDim.x;
        /*
         * Uniform for the entire block, so every thread either
         * continues or breaks together.
         */
        if (base >= total) {
            break;
        }

        /*
         * Find the triangle containing the first fragment owned
         * by this block/chunk.
         */
        if (threadIdx.x == 0) {
            int low = 0;
            int high = triangle_count;

            while (low != high) {
                const int mid = (low + high) / 2;

                if (prefix[mid] <= base) {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }

            start_triangle = high - 1;
        }

        __syncthreads();

        /*
         * Cache the next <=256 prefix entries into shared memory.
         */
        const int prefix_idx = start_triangle + threadIdx.x;

        if (prefix_idx < triangle_count) {
            reduced_prefix[threadIdx.x] = prefix[prefix_idx];
        } else {
            reduced_prefix[threadIdx.x] = INT_MAX;
        }

        __syncthreads();

        const int i = base + threadIdx.x;

        if (i < total) {
            int low = 0;
            int high = min(
                static_cast<int>(blockDim.x), triangle_count - start_triangle);

            while (low != high) {
                const int mid = (low + high) / 2;

                if (reduced_prefix[mid] <= i) {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }

            const int triangle_index = high - 1 + start_triangle;

            const int triangle_index4 = 4 * triangle_index;

            const int lx = bounding_boxes[triangle_index4];
            const int by = bounding_boxes[triangle_index4 + 1];
            const int rx = bounding_boxes[triangle_index4 + 2];

            const int bbox_width = rx - lx + 1;

            const int inside_index = i - prefix[triangle_index];

            const int px_pixel = lx + inside_index % bbox_width;

            const int py_pixel = by + inside_index / bbox_width;

            const float px = static_cast<float>(px_pixel) + 0.5f;
            const float py = static_cast<float>(py_pixel) + 0.5f;

            const int triangle_index6 = 6 * triangle_index;

            const float x1 = projected[triangle_index6];
            const float y1 = projected[triangle_index6 + 1];
            const float x2 = projected[triangle_index6 + 2];
            const float y2 = projected[triangle_index6 + 3];
            const float x3 = projected[triangle_index6 + 4];
            const float y3 = projected[triangle_index6 + 5];

            const float denominator =
                (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);

            const float a = (y2 - y3) * (px - x3) + (x3 - x2) * (py - y3);

            if (denominator > 0.0f) {
                if (0.0f <= a && a <= denominator) {
                    const float b =
                        (y3 - y1) * (px - x3) + (x1 - x3) * (py - y3);

                    if (0.0f <= b && b <= denominator) {
                        const float c = denominator - a - b;

                        if (0.0f <= c && c <= denominator) {
                            image[py_pixel * width + px_pixel] = 255;
                        }
                    }
                }
            } else {
                if (0.0f >= a && a >= denominator) {
                    const float b =
                        (y3 - y1) * (px - x3) + (x1 - x3) * (py - y3);

                    if (0.0f >= b && b >= denominator) {
                        const float c = denominator - a - b;

                        if (0.0f >= c && c >= denominator) {
                            image[py_pixel * width + px_pixel] = 255;
                        }
                    }
                }
            }
        }

        /*
         * Required: no thread may begin the next chunk and overwrite
         * shared state while another thread is still using it.
         */
        __syncthreads();
    }
}

__global__ void FillTriangleKernel(
    int* sizes,
    int* prefix,
    int* bounding_boxes,
    unsigned char* iamges,
    int triangle_count,
    int width,
    int height,
    float* projected,
    int* dev_stride_prefixes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int stridedIndex;

    if (threadIdx.x == 0) {
        int j = blockIdx.x * blockDim.x;

        int low = 0;
        int high = triangle_count;

        while (low != high) {
            int mid = (low + high) / 2;

            if (prefix[mid] <= j) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        stridedIndex = high - 1;
    }

    __syncthreads();

    if (i < prefix[triangle_count - 1] + sizes[triangle_count - 1]) {
        /*Index of Triangle for the given stride (stride is of size 256 and the
         * stride group is blockIdx.x)*/
        // int stridedIndex = dev_stride_prefixes[blockIdx.x];

        /*Load [stridedIndex, stridedIndex + 255] at most (256) elements to
         * another shared memory (could hit upper bound)*/
        __shared__ int reducedPrefix[threads_per_block];
        if (threadIdx.x + stridedIndex < triangle_count) {
            reducedPrefix[threadIdx.x] = prefix[threadIdx.x + stridedIndex];
        }
        __syncthreads();

        /*Binary Search Loop Variables*/
        int low = 0;
        int high = min(blockDim.x, triangle_count - stridedIndex);
        int mid = 0;
        int triangleIndex = -1;

        while (low != high) {
            mid = (low + high) / 2;
            if (reducedPrefix[mid] <= i) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        triangleIndex = high - 1 + stridedIndex;

        /*Calculate the Pixel to Evaluate (Corresponding to Thread)*/
        int triangleIndex4 = 4 * triangleIndex;
        int Lx = bounding_boxes[triangleIndex4];
        int By = bounding_boxes[triangleIndex4 + 1];
        int Rx = bounding_boxes[triangleIndex4 + 2];
        int insideIndex = i - prefix[triangleIndex];
        int pxPixel = Lx + insideIndex % (Rx - Lx + 1);
        int pyPixel = By + insideIndex / (Rx - Lx + 1);
        float px = pxPixel + 0.5;
        float py = pyPixel + 0.5;

        /*Load in Triangle Coordinates*/
        int triangleIndex6 = 6 * triangleIndex;
        float x1 = projected[triangleIndex6];
        float y1 = projected[triangleIndex6 + 1];
        float x2 = projected[triangleIndex6 + 2];
        float y2 = projected[triangleIndex6 + 3];
        float x3 = projected[triangleIndex6 + 4];
        float y3 = projected[triangleIndex6 + 5];

        /*Use Barycentric Coordinates to Check if Point is In Triangle*/
        float denominator = ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3));
        float a = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3));

        if (denominator > 0) {
            if (0 <= a && a <= denominator) {
                float b = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3));
                if (0 <= b && b <= denominator) {
                    float c = denominator - a - b;
                    if (0 <= c && c <= denominator) {
                        iamges[(pyPixel)*width + pxPixel] = 255;
                    }
                }
            }
        } else {
            if (0 >= a && a >= denominator) {
                float b = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3));
                if (0 >= b && b >= denominator) {
                    float c = denominator - a - b;
                    if (0 >= c && c >= denominator) {
                        iamges[(pyPixel)*width + pxPixel] = 255;
                    }
                }
            }
        }
    }
}

/* U4: device-driven persistent worker kernels (fixed grid, chunk claiming via
 * atomicAdd) */
__global__ void OverflowCheckKernel(
    int* dev_fragment_fill,
    int* dev_overflowFlag,
    long long maxFragments) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Treat any signed overflow (negative fragment_fill) OR exceeding the
        // safe fragment budget as overflow.  maxFragments is long long to avoid
        // the int-wrapping bug (10,000,000 * 255 > INT_MAX).
        const int ff = dev_fragment_fill[0];
        dev_overflowFlag[0] =
            (ff < 0 || static_cast<long long>(ff) > maxFragments) ? 1 : 0;
    }
}

__global__ void StridePrefixPersistentKernel(
    int* dev_nextChunk,
    int chunkSize,
    int* dev_fragment_fill,
    int* dev_overflowFlag,
    int* dev_sizes,
    int* dev_prefix,
    int* dev_stride_prefixes,
    int triangle_count,
    int stride) {
    int total = dev_fragment_fill[0];
    if (*dev_overflowFlag) {
        return;
    }
    while (true) {
        int chunkStart = atomicAdd(dev_nextChunk, chunkSize);
        int jStart = chunkStart * stride;
        if (jStart >= total) {
            break;
        }
        int jEnd = min((chunkStart + chunkSize) * stride, total);
        // Process stride elements chunkStart .. chunkStart+chunkSize-1 but only
        // those with j < total
        for (int idx = chunkStart; idx < chunkStart + chunkSize; ++idx) {
            int j = idx * stride;
            if (j >= total) {
                break;
            }
            int low = 0, high = triangle_count, mid = 0;
            while (low != high) {
                mid = (low + high) / 2;
                if (dev_prefix[mid] <= j) {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            int strideIndex = high - 1;
            if (idx < maximum_stride_size) {
                dev_stride_prefixes[idx] = strideIndex;
            }
        }
        // Also need to handle jEnd unused
        (void)jEnd;
    }
}

__global__ void FillTrianglePersistentKernel(
    int* dev_nextCandidate,
    int chunkSize,
    int* dev_fragment_fill,
    int* dev_overflowFlag,
    int triangle_count,
    int* dev_bbox_triangles,
    int* dev_sizes,
    int* dev_prefix,
    unsigned char* dev_image,
    int width,
    int height,
    float* dev_projected_triangles,
    int* dev_stride_prefixes) {
    int total = dev_fragment_fill[0];
    if (*dev_overflowFlag) {
        return;
    }
    while (true) {
        int start = atomicAdd(dev_nextCandidate, chunkSize);
        if (start >= total) {
            break;
        }
        int end = start + chunkSize;
        if (end > total) {
            end = total;
        }
        for (int i = start; i < end; ++i) {
            // Find triangle index via global binary search on prefix
            // (device-driven, no shared memory)
            int low = 0, high = triangle_count, mid = 0;
            while (low != high) {
                mid = (low + high) / 2;
                if (dev_prefix[mid] <= i) {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            int triangleIndex = high - 1;
            if (triangleIndex < 0 || triangleIndex >= triangle_count) {
                continue;
            }
            int triangleIndex4 = 4 * triangleIndex;
            int Lx = dev_bbox_triangles[triangleIndex4];
            int By = dev_bbox_triangles[triangleIndex4 + 1];
            int Rx = dev_bbox_triangles[triangleIndex4 + 2];
            // Use stored sizes/prefix to compute insideIndex as original did
            int insideIndex = i - dev_prefix[triangleIndex];
            int denomX = Rx - Lx + 1;
            if (denomX <= 0) {
                continue;
            }
            int pxPixel = Lx + insideIndex % denomX;
            int pyPixel = By + insideIndex / denomX;
            if (pxPixel < 0 || pxPixel >= width || pyPixel < 0 ||
                pyPixel >= height) {
                continue;
            }
            float px = pxPixel + 0.5f;
            float py = pyPixel + 0.5f;
            int triangleIndex6 = 6 * triangleIndex;
            float x1 = dev_projected_triangles[triangleIndex6];
            float y1 = dev_projected_triangles[triangleIndex6 + 1];
            float x2 = dev_projected_triangles[triangleIndex6 + 2];
            float y2 = dev_projected_triangles[triangleIndex6 + 3];
            float x3 = dev_projected_triangles[triangleIndex6 + 4];
            float y3 = dev_projected_triangles[triangleIndex6 + 5];
            float denominator = ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3));
            float a = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3));
            if (denominator > 0) {
                if (0 <= a && a <= denominator) {
                    float b = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3));
                    if (0 <= b && b <= denominator) {
                        float c = denominator - a - b;
                        if (0 <= c && c <= denominator) {
                            dev_image[pyPixel * width + pxPixel] = 255;
                        }
                    }
                }
            } else {
                if (0 >= a && a >= denominator) {
                    float b = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3));
                    if (0 >= b && b >= denominator) {
                        float c = denominator - a - b;
                        if (0 >= c && c >= denominator) {
                            dev_image[pyPixel * width + pxPixel] = 255;
                        }
                    }
                }
            }
        }
    }
    // dev_stride_prefixes is read-only in this kernel; kept for interface
    // parity
    (void)dev_stride_prefixes;
}

__global__ void RasterizeTrianglesWarpKernel(
    const int* dev_projected_triangles_snapped,
    const float* dev_projected_triangles,
    const bool* dev_backface,
    unsigned char* dev_image,
    int* dev_bounding_box,
    int triangle_count,
    int width,
    int height) {
    /*
     * One warp owns one triangle.
     *
     * With a 256-thread block:
     *
     * warp 0 -> triangle N + 0
     * warp 1 -> triangle N + 1
     * ...
     * warp 7 -> triangle N + 7
     */
    const int warp_in_block = threadIdx.x / warpSize;

    const int lane = threadIdx.x % warpSize;

    const int warps_per_block = blockDim.x / warpSize;

    const int triangle_index = blockIdx.x * warps_per_block + warp_in_block;

    /*
     * Uniform across the entire warp, so returning here is safe.
     */
    if (triangle_index >= triangle_count) {
        return;
    }

    constexpr unsigned full_mask = 0xffffffffu;

    /*
     * Lane zero calculates triangle-level state once.
     */
    int left_x = 0;
    int bottom_y = 0;
    int right_x = 0;
    int top_y = 0;

    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float x3 = 0.0f;
    float y3 = 0.0f;

    int backface = 0;

    if (lane == 0) {
        const int snapped_index = 6 * triangle_index;

        /*
         * Load snapped coordinates for bounding box.
         */
        const int sx1 = dev_projected_triangles_snapped[snapped_index];

        const int sy1 = dev_projected_triangles_snapped[snapped_index + 1];

        const int sx2 = dev_projected_triangles_snapped[snapped_index + 2];

        const int sy2 = dev_projected_triangles_snapped[snapped_index + 3];

        const int sx3 = dev_projected_triangles_snapped[snapped_index + 4];

        const int sy3 = dev_projected_triangles_snapped[snapped_index + 5];

        /*
         * EXACT same bbox clamping rules as
         * BoundingBoxForTrianglesKernel.
         */
        left_x = max(min(min(min(sx1, sx2), sx3), width - 1), 0);

        bottom_y = max(min(min(min(sy1, sy2), sy3), height - 1), 0);

        right_x = min(max(max(max(sx1, sx2), sx3), 0), width - 1);

        top_y = min(max(max(max(sy1, sy2), sy3), 0), height - 1);

        /*
         * Preserve existing overall model bounding-box semantics.
         *
         * BoundingBoxSizesKernel currently updates the global bbox
         * for every triangle, including backfaces.
         */
        atomicMin(&dev_bounding_box[0], left_x);

        atomicMin(&dev_bounding_box[1], bottom_y);

        atomicMax(&dev_bounding_box[2], right_x);

        atomicMax(&dev_bounding_box[3], top_y);

        /*
         * Load unsnapped triangle coordinates used for the actual
         * point-in-triangle test.
         */
        const int projected_index = 6 * triangle_index;

        x1 = dev_projected_triangles[projected_index];

        y1 = dev_projected_triangles[projected_index + 1];

        x2 = dev_projected_triangles[projected_index + 2];

        y2 = dev_projected_triangles[projected_index + 3];

        x3 = dev_projected_triangles[projected_index + 4];

        y3 = dev_projected_triangles[projected_index + 5];

        backface = dev_backface[triangle_index] ? 1 : 0;
    }

    /*
     * Broadcast triangle state from lane zero to the whole warp.
     */
    left_x = __shfl_sync(full_mask, left_x, 0);

    bottom_y = __shfl_sync(full_mask, bottom_y, 0);

    right_x = __shfl_sync(full_mask, right_x, 0);

    top_y = __shfl_sync(full_mask, top_y, 0);

    x1 = __shfl_sync(full_mask, x1, 0);

    y1 = __shfl_sync(full_mask, y1, 0);

    x2 = __shfl_sync(full_mask, x2, 0);

    y2 = __shfl_sync(full_mask, y2, 0);

    x3 = __shfl_sync(full_mask, x3, 0);

    y3 = __shfl_sync(full_mask, y3, 0);

    backface = __shfl_sync(full_mask, backface, 0);

    const int bbox_width = right_x - left_x + 1;

    const int bbox_height = top_y - bottom_y + 1;

    if (bbox_width <= 0 || bbox_height <= 0) {
        return;
    }

    /*
     * IMPORTANT:
     *
     * Preserve the CURRENT renderer's backface behavior.
     *
     * BoundingBoxSizesKernel gives a backface triangle size=1.
     * FillTriangle_new therefore considers exactly one bbox fragment
     * for a backface, rather than the whole bbox.
     *
     * Keeping that here makes the comparison much closer to bit-identical.
     */
    const int fragment_count = backface ? 1 : bbox_width * bbox_height;

    /*
     * Precompute the denominator once per lane.
     *
     * These values are invariant for every pixel in the triangle.
     */
    const float denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);

    /*
     * Lane 0 processes fragment 0,
     * lane 1 processes fragment 1,
     * ...
     * lane 31 processes fragment 31,
     *
     * then each lane advances by 32.
     */
    for (int inside_index = lane; inside_index < fragment_count;
         inside_index += warpSize) {
        const int px_pixel = left_x + inside_index % bbox_width;

        const int py_pixel = bottom_y + inside_index / bbox_width;

        /*
         * Bbox is already clamped, but keep this guard while
         * validating the new rasterizer.
         */
        if (px_pixel < 0 || px_pixel >= width || py_pixel < 0 ||
            py_pixel >= height) {
            continue;
        }

        const float px = static_cast<float>(px_pixel) + 0.5f;

        const float py = static_cast<float>(py_pixel) + 0.5f;

        /*
         * EXACT same barycentric test used by FillTriangleKernel_new.
         */
        const float a = (y2 - y3) * (px - x3) + (x3 - x2) * (py - y3);

        if (denominator > 0.0f) {
            if (0.0f <= a && a <= denominator) {
                const float b = (y3 - y1) * (px - x3) + (x1 - x3) * (py - y3);

                if (0.0f <= b && b <= denominator) {
                    const float c = denominator - a - b;

                    if (0.0f <= c && c <= denominator) {
                        dev_image[py_pixel * width + px_pixel] = 255;
                    }
                }
            }

        } else {
            if (0.0f >= a && a >= denominator) {
                const float b = (y3 - y1) * (px - x3) + (x1 - x3) * (py - y3);

                if (0.0f >= b && b >= denominator) {
                    const float c = denominator - a - b;

                    if (0.0f >= c && c >= denominator) {
                        dev_image[py_pixel * width + px_pixel] = 255;
                    }
                }
            }
        }
    }
}

cudaError_t RenderEngine::Render() {
    /*Create Error Status*/
    cudaGetLastError();  // Resets Errors (MAYBE DELETE TO SAVE TIME?)

    /*Transform Points (Rotate then Translate) and Project to Screen and Snap*/
    WorldToPixelKernel<<<dim_grid_vertices_, threads_per_block>>>(
        dev_triangles_,
        dev_projected_triangles_,
        dev_projected_triangles_snapped_,
        3 * triangle_count_,
        dist_over_pix_pitch_,
        pix_conversion_x_,
        pix_conversion_y_,
        model_pose_.x_location_,
        model_pose_.y_location_,
        model_pose_.z_location_,
        model_rotation_mat_,
        dev_normals_,
        dev_backface_,
        use_backface_culling_,
        fx_,
        fy_,
        cx_,
        cy_,
        renderer_output_->GetDeviceImagePointer(),
        dev_bounding_box_,
        width_,
        height_);

    BoundingBoxForTrianglesKernel<<<
        dim_grid_bounding_box_,
        threads_per_block>>>(
        dev_bounding_box_triangles_,
        dev_projected_triangles_snapped_,
        triangle_count_,
        width_,
        height_);

    BoundingBoxSizesKernel<<<dim_grid_triangles_, threads_per_block>>>(
        dev_bounding_box_triangles_,
        dev_bounding_box_triangles_sizes_,
        triangle_count_,
        dev_bounding_box_,
        dev_backface_);

    /*Use CUB library to compute exlusive prefix sum of bound box sizes.*/
    cub::DeviceScan::ExclusiveSum(
        dev_cub_storage_,
        cub_storage_bytes_,
        dev_bounding_box_triangles_sizes_,
        dev_bounding_box_triangles_sizes_prefix_,
        triangle_count_);

    FillTriangleKernel_new<<<fill_triangle_grid_, threads_per_block>>>(
        dev_bounding_box_triangles_sizes_,
        dev_bounding_box_triangles_sizes_prefix_,
        dev_bounding_box_triangles_,
        renderer_output_->GetDeviceImagePointer(),
        triangle_count_,
        width_,
        height_,
        dev_projected_triangles_);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "FillTriangleKernel_new launch failed: "
                  << cudaGetErrorString(err) << '\n';
    }

    /*Check for Errors*/
    return cudaGetLastError();
}

bool RenderEngine::WriteImage(std::string file_name) {
    /*Check Initialized First*/
    if (!initialized_correctly_) {
        std::cout << "\nCUDA not Initialized for Render Engine - Cannot Write!";
        return false;
    }
    /*Array for Storing Device Image on Host*/
    auto host_image = static_cast<unsigned char*>(
        malloc(width_ * height_ * sizeof(unsigned char)));
    cudaMemcpy(
        host_image,
        renderer_output_->GetDeviceImagePointer(),
        width_ * height_ * sizeof(unsigned char),
        cudaMemcpyDeviceToHost);

    /*OpenCV Image Container/Write Function*/
    auto projection_mat =
        cv::Mat(height_, width_, CV_8UC1, host_image); /*Reverse before flip*/
    auto output_mat = cv::Mat(width_, height_, CV_8UC1);
    flip(projection_mat, output_mat, 0);
    bool result = imwrite(file_name, output_mat);

    /*Free Array*/
    free(host_image);
    return result;
}

cv::Mat RenderEngine::GetcvMatImage() {
    /*Check Initialized First*/
    if (!initialized_correctly_) {
        std::cout << "\nCUDA not Initialized for Render Engine - Cannot Write!";
        // return false;
    }

    /*Array for Storing Device Image on Host*/
    auto host_image = static_cast<unsigned char*>(
        malloc(width_ * height_ * sizeof(unsigned char)));
    cudaMemcpy(
        host_image,
        renderer_output_->GetDeviceImagePointer(),
        width_ * height_ * sizeof(unsigned char),
        cudaMemcpyDeviceToHost);

    /*OpenCV Image Container/Write Function*/
    auto projection_mat =
        cv::Mat(height_, width_, CV_8UC1, host_image); /*Reverse before flip*/
    auto output_mat = cv::Mat(width_, height_, CV_8UC1);
    flip(projection_mat, output_mat, 0);

    /*Free Array*/
    free(host_image);
    return output_mat;
}

bool RenderEngine::IsInitializedCorrectly() {
    return initialized_correctly_;
};

std::size_t RenderEngine::GetCubStorageBytes() const {
    return cub_storage_bytes_;
}

int RenderEngine::GetWidth() const {
    return width_;
}
int RenderEngine::GetHeight() const {
    return height_;
}
int RenderEngine::GetTriangleCount() const {
    return triangle_count_;
}
}  // namespace gpu_cost_function
