/*Render Engine Header*/
#include "gpu/render_engine.cuh"

/*Cub Library (CUDA)*/
#include "cub/cub.cuh"
#include "cub/device/device_scan.cuh"
#include "cub/util_allocator.cuh"

/*Standard Library*/
#include <iostream>

/*OpenCV 3.1 Library*/
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/*CUDA Custom Registration Namespace (Compiling as DLL)*/
namespace gpu_cost_function {

Pose::Pose(float x_location, float y_location, float z_location, float x_angle,
           float y_angle, float z_angle) {
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

RotationMatrix::RotationMatrix(float rotation_00, float rotation_01,
                               float rotation_02, float rotation_10,
                               float rotation_11, float rotation_12,
                               float rotation_20, float rotation_21,
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

RenderEngine::RenderEngine(int width, int height, int device,
                           bool use_backface_culling, float* triangles,
                           float* normals, int triangle_count,
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
        cx_ =
            static_cast<float>(width_) / 2.0f -
            camera_calibration_.principal_x_ / camera_calibration_.pixel_pitch_;
        cy_ =
            static_cast<float>(height) / 2.0f -
            camera_calibration_.principal_y_ / camera_calibration_.pixel_pitch_;
    } else if (camera_calibration_.type_ == "Denver") {
        fx_ = camera_calibration_.fx();
        fy_ = -camera_calibration_.fy();
        cx_ = camera_calibration_.cx();
        cy_ = height - camera_calibration_.cy();
    }
    pix_conversion_x_ =
        static_cast<float>(width_) / 2.0f -
        camera_calibration_.principal_x_ / camera_calibration_.pixel_pitch_;
    pix_conversion_y_ =
        static_cast<float>(height_) / 2.0f -
        camera_calibration_.principal_y_ / camera_calibration_.pixel_pitch_;
    dist_over_pix_pitch_ = -1.0f * camera_calibration_.principal_distance_ /
                           camera_calibration_.pixel_pitch_;

    /*Initialize Kernel Launch Sizes*/
    dim_grid_triangles_ =
        dim3(ceil(sqrt(static_cast<double>(triangle_count_) /
                       static_cast<double>(threads_per_block))),
             ceil(sqrt(static_cast<double>(triangle_count_) /
                       static_cast<double>(threads_per_block))));
    dim_grid_vertices_ =
        dim3(ceil(sqrt(3.0 * triangle_count_ /
                       static_cast<double>(threads_per_block))),
             ceil(sqrt(3.0 * triangle_count_ /
                       static_cast<double>(threads_per_block))));
    dim_grid_bounding_box_ =
        dim3(ceil(sqrt(4.0 * triangle_count_ /
                       static_cast<double>(threads_per_block))),
             ceil(sqrt(4.0 * triangle_count_ /
                       static_cast<double>(threads_per_block))));

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

    /*Free Host*/
    cudaFreeHost(fragment_fill_);
}

cudaError_t RenderEngine::InitializeCUDA(float* triangles, float* normals,
                                         int device) {
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
    cudaHostAlloc((void**)&fragment_fill_, 1 * sizeof(int),
                  cudaHostAllocDefault);

    /*Allocate GPU buffers for image, triangles.*/
    cudaMalloc((void**)&dev_z_line_values_, width_ * height_ * sizeof(float));

    cudaMalloc((void**)&dev_triangles_, triangle_count_ * 9 * sizeof(float));

    cudaMalloc((void**)&dev_normals_, triangle_count_ * 3 * sizeof(float));

    cudaMalloc((void**)&dev_backface_, triangle_count_ * sizeof(bool));

    cudaMalloc((void**)&dev_transf_vertex_zs_,
               triangle_count_ * 3 * sizeof(float));

    cudaMalloc((void**)&dev_tangent_triangle_, triangle_count_ * sizeof(bool));

    cudaMalloc((void**)&dev_projected_triangles_,
               triangle_count_ * 6 * sizeof(float));

    cudaMalloc((void**)&dev_projected_triangles_snapped_,
               triangle_count_ * 6 * sizeof(int));

    cudaMalloc((void**)&dev_bounding_box_triangles_,
               triangle_count_ * 4 * sizeof(int));

    cudaMalloc((void**)&dev_bounding_box_triangles_sizes_,
               triangle_count_ * sizeof(int));

    cudaMalloc((void**)&dev_bounding_box_triangles_sizes_prefix_,
               triangle_count_ * sizeof(int));

    cudaMalloc((void**)&dev_bounding_box_, 4 * sizeof(int));

    cudaMalloc((void**)&dev_fragment_fill_, 1 * sizeof(int));

    cudaMalloc((void**)&dev_stride_prefixes_,
               maximum_stride_size * sizeof(int));

    /*Check for Errors*/
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        initialized_correctly_ = false;
        FreeCuda();
        return cudaStatus;
    }

    /*Initialize the GPU Image*/
    renderer_output_ = new GPUImage(width_, height_, device);

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
        dev_cub_storage_, cub_storage_bytes_, dev_bounding_box_triangles_sizes_,
        dev_bounding_box_triangles_sizes_, triangle_count_);

    cudaMalloc(&dev_cub_storage_, cub_storage_bytes_);

    /*Check for Errors*/
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        initialized_correctly_ = false;
        FreeCuda();
        return cudaStatus;
    }

    /*Copy input from host memory to GPU.*/
    cudaMemcpy(dev_triangles_, triangles, triangle_count_ * 9 * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemcpy(dev_normals_, normals, triangle_count_ * 3 * sizeof(float),
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
    model_rotation_mat_ =
        RotationMatrix(cz * cy - sz * sx * sy, -1.0 * sz * cx,
                       cz * sy + sz * cy * sx, sz * cy + cz * sx * sy, cz * cx,
                       sz * sy - cz * cy * sx, -1.0 * cx * sy, sx, cx * cy);
}
void RenderEngine::SetRotationMatrix(RotationMatrix model_rotation_matrix) {
    model_rotation_mat_ = model_rotation_matrix;
}

GPUImage* RenderEngine::GetRenderOutput() { return renderer_output_; }

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
    float* dev_triangles, float* dev_projected_triangles,
    int* dev_projected_triangles_snapped, int vertex_count,
    float dist_over_pix_pitch, float pix_conversion_x, float pix_conversion_y,
    float x_location, float y_location, float z_location,
    RotationMatrix model_rotation_mat, float* dev_normals, bool* dev_backface,
    bool use_backface_culling, float fx, float fy, float cx, float cy) {
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

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
            if (dotProduct >= 0)
                dev_backface[i / 3] = true;
            else
                dev_backface[i / 3] = false;
            if (!use_backface_culling) dev_backface[i / 3] = false;
        }
        // Need to change this condition - it definitely can be higher than zero
        // if you are using a different calibration setup.
        if (tZ == 0)
            tZ = -.000001; /*Can't be above or at zero, so make very
                              small..should never happen*/

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

__global__ void BoundingBoxForTrianglesKernel(
    int* dev_bounding_box_triangles, int* dev_projected_triangles_snapped,
    int triangle_count, int width, int height) {
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

__global__ void BoundingBoxSizesKernel(int* dev_bounding_box_triangles,
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

__global__ void PrepareLaunchPacketKernel(
    int* dev_fragment_fill, int* dev_bounding_box_triangles_sizes,
    int* dev_bounding_box_triangles_sizes_prefix, int triangle_count) {
    dev_fragment_fill[0] =
        dev_bounding_box_triangles_sizes[triangle_count - 1] +
        dev_bounding_box_triangles_sizes_prefix[triangle_count - 1];
}

__global__ void StridePrefixKernel(int stride,
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

__global__ void FillTriangleKernel(int* dev_bounding_box_triangles_sizes,
                                   int* dev_bounding_box_triangles_sizes_prefix,
                                   int* dev_bounding_box_triangles,
                                   unsigned char* dev_image, int triangle_count,
                                   int width, int height,
                                   float* dev_projected_triangles,
                                   int* dev_stride_prefixes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < dev_bounding_box_triangles_sizes_prefix[triangle_count - 1] +
                dev_bounding_box_triangles_sizes[triangle_count - 1]) {
        /*Index of Triangle for the given stride (stride is of size 256 and the
         * stride group is blockIdx.x)*/
        int stridedIndex = dev_stride_prefixes[blockIdx.x];

        /*Load [stridedIndex, stridedIndex + 255] at most (256) elements to
         * another shared memory (could hit upper bound)*/
        __shared__ int reducedBoundingBoxTrianglesSizePrefix[threads_per_block];
        if (threadIdx.x + stridedIndex < triangle_count)
            reducedBoundingBoxTrianglesSizePrefix[threadIdx.x] =
                dev_bounding_box_triangles_sizes_prefix[threadIdx.x +
                                                        stridedIndex];
        __syncthreads();

        /*Binary Search Loop Variables*/
        int low = 0;
        int high = min(blockDim.x, triangle_count - stridedIndex);
        int mid = 0;
        int triangleIndex = -1;

        /*Binary Search Loop*/
        while (low != high) {
            /*Calculate Mid Index*/
            mid = (low + high) / 2;

            if (reducedBoundingBoxTrianglesSizePrefix[mid] <= i) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        triangleIndex = high - 1 + stridedIndex;

        /*Calculate the Pixel to Evaluate (Corresponding to Thread)*/
        int triangleIndex4 = 4 * triangleIndex;
        int Lx = dev_bounding_box_triangles[triangleIndex4];
        int By = dev_bounding_box_triangles[triangleIndex4 + 1];
        int Rx = dev_bounding_box_triangles[triangleIndex4 + 2];
        int insideIndex =
            i - dev_bounding_box_triangles_sizes_prefix[triangleIndex];
        int pxPixel = Lx + insideIndex % (Rx - Lx + 1);
        int pyPixel = By + insideIndex / (Rx - Lx + 1);
        float px = pxPixel + 0.5;
        float py = pyPixel + 0.5;

        /*Load in Triangle Coordinates*/
        int triangleIndex6 = 6 * triangleIndex;
        float x1 = dev_projected_triangles[triangleIndex6];
        float y1 = dev_projected_triangles[triangleIndex6 + 1];
        float x2 = dev_projected_triangles[triangleIndex6 + 2];
        float y2 = dev_projected_triangles[triangleIndex6 + 3];
        float x3 = dev_projected_triangles[triangleIndex6 + 4];
        float y3 = dev_projected_triangles[triangleIndex6 + 5];

        /*Use Barycentric Coordinates to Check if Point is In Triangle*/
        float denominator = ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3));
        float a = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3));

        if (denominator > 0) {
            if (0 <= a && a <= denominator) {
                float b = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3));
                if (0 <= b && b <= denominator) {
                    float c = denominator - a - b;
                    if (0 <= c && c <= denominator) {
                        dev_image[(pyPixel)*width + pxPixel] = 255;
                    }
                }
            }
        } else {
            if (0 >= a && a >= denominator) {
                float b = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3));
                if (0 >= b && b >= denominator) {
                    float c = denominator - a - b;
                    if (0 >= c && c >= denominator) {
                        dev_image[(pyPixel)*width + pxPixel] = 255;
                    }
                }
            }
        }
    }
}

cudaError_t RenderEngine::Render() {
    /*Create Error Status*/
    cudaGetLastError();  // Resets Errors (MAYBE DELETE TO SAVE TIME?)

    /*Clear Image*/
    cudaMemset(renderer_output_->GetDeviceImagePointer(), 0,
               width_ * height_ * sizeof(unsigned char));

    /*Reset Launch Packet*/
    ResetKernel<<<1, 1>>>(dev_bounding_box_, width_, height_);

    /*Transform Points (Rotate then Translate) and Project to Screen and Snap*/
    WorldToPixelKernel<<<dim_grid_vertices_, threads_per_block>>>(
        dev_triangles_, dev_projected_triangles_,
        dev_projected_triangles_snapped_, 3 * triangle_count_,
        dist_over_pix_pitch_, pix_conversion_x_, pix_conversion_y_,
        model_pose_.x_location_, model_pose_.y_location_,
        model_pose_.z_location_, model_rotation_mat_, dev_normals_,
        dev_backface_, use_backface_culling_, fx_, fy_, cx_, cy_);

    /*Calculate Bounding Boxes for Each Triangle*/
    BoundingBoxForTrianglesKernel<<<dim_grid_bounding_box_,
                                    threads_per_block>>>(
        dev_bounding_box_triangles_, dev_projected_triangles_snapped_,
        triangle_count_, width_, height_);

    /*Calculate Sizes of Bounding Boxes and Overall Bounding Box of Model*/
    BoundingBoxSizesKernel<<<dim_grid_triangles_, threads_per_block>>>(
        dev_bounding_box_triangles_, dev_bounding_box_triangles_sizes_,
        triangle_count_, dev_bounding_box_, dev_backface_);

    /*Use CUB library to compute exlusive prefix sum of bound box sizes.*/
    cub::DeviceScan::ExclusiveSum(
        dev_cub_storage_, cub_storage_bytes_, dev_bounding_box_triangles_sizes_,
        dev_bounding_box_triangles_sizes_prefix_, triangle_count_);

    /*Prepare Launch Packet and Send it to Host*/
    /*Contains bounding box on white pixels (LX,BY,RX,TY, and # of fragments to
    process (last element in dev_boundingBoxTrianglesSizePrefix and last element
    in dev_boundingBoxTrianglesSize)*/
    PrepareLaunchPacketKernel<<<1, 1>>>(
        dev_fragment_fill_, dev_bounding_box_triangles_sizes_,
        dev_bounding_box_triangles_sizes_prefix_, triangle_count_);

    cudaMemcpy(renderer_output_->GetBoundingBox(), dev_bounding_box_,
               4 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(fragment_fill_, dev_fragment_fill_, 1 * sizeof(int),
               cudaMemcpyDeviceToHost);

    /*Because doing a binary search for every fragement on the prefix search
    takes too long, we first do this on every 256th fragment, then load the
    above 256 prefix values into shared memory and binary search over those. The
    first part occurrs in the the StridePrefixKernel and the second occurs in
    the FillTriangleKernel. In the FillTriangleKernel we also shade (or not
    shade) the fragment based on the results of the point (center of pixel) in
    triangle test.*/

    /*Error check for too many fragments.*/
    if (static_cast<double>(fragment_fill_[0]) >
        static_cast<double>(maximum_stride_size) *
            static_cast<double>(threads_per_block - 1)) {
        fprintf(stderr,
                "Fragment overflow! Please shrink image and/or reduce model "
                "triangle count!");
        fragment_overflow_ = true;
        return cudaErrorMemoryAllocation;
    }

    StridePrefixKernel<<<ceil(static_cast<double>(fragment_fill_[0]) /
                              static_cast<double>(threads_per_block *
                                                  threads_per_block)),
                         threads_per_block>>>(
        threads_per_block, dev_bounding_box_triangles_sizes_,
        dev_bounding_box_triangles_sizes_prefix_, dev_stride_prefixes_,
        triangle_count_);

    FillTriangleKernel<<<ceil(static_cast<double>(fragment_fill_[0]) /
                              static_cast<double>(threads_per_block)),
                         threads_per_block>>>(
        dev_bounding_box_triangles_sizes_,
        dev_bounding_box_triangles_sizes_prefix_, dev_bounding_box_triangles_,
        renderer_output_->GetDeviceImagePointer(), triangle_count_, width_,
        height_, dev_projected_triangles_, dev_stride_prefixes_);

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
    cudaMemcpy(host_image, renderer_output_->GetDeviceImagePointer(),
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
    cudaMemcpy(host_image, renderer_output_->GetDeviceImagePointer(),
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

bool RenderEngine::IsInitializedCorrectly() { return initialized_correctly_; };
}  // namespace gpu_cost_function
