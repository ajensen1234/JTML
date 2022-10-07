/*Render Engine Header*/
#include "gpu/render_engine.cuh"

/*Cub Library (CUDA)*/
#include <cub/cub.cuh>
#include <cub/util_allocator.cuh>
#include <cub/device/device_scan.cuh>


/*CUDA Custom Registration Namespace (Compiling as DLL)*/
namespace gpu_cost_function {

	/*Kernels*/
	__global__ void DRR_ResetKernel(int* dev_bounding_box, int width, int height) {
		dev_bounding_box[0] = width - 1;
		/*Left Most X -> initialize with width - 1 (we are zero based) since can only be brought down*/
		dev_bounding_box[1] = height - 1;
		/*Bottom Most Y -> initialize with height - 1 (we are zero based) since can only be brought down*/
		dev_bounding_box[2] = 0; /*Right Most X -> initialize with zero since can only be brought up*/
		dev_bounding_box[3] = 0; /*Top Most Y -> initialize with zero since can only be brought up*/
	}

	__global__ void DRR_WorldToPixelKernel(float* dev_triangles, float* dev_projected_triangles,
	                                       int* dev_projected_triangles_snapped, float* dev_transf_vertex_zs,
	                                       int vertex_count, float dist_over_pix_pitch, float pix_conversion_x,
	                                       float pix_conversion_y,
	                                       float x_location, float y_location, float z_location,
	                                       RotationMatrix model_rotation_mat,
	                                       float* dev_normals, bool* dev_backface, bool* dev_tangent_triangle) {
		int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

		if (i < vertex_count) {
			/*Read in Vertices*/
			float vX = dev_triangles[(3 * i)];
			float vY = dev_triangles[(3 * i) + 1];
			float vZ = dev_triangles[(3 * i) + 2];

			/*Transform (Rotate then Translate) Vertices*/
			float tX = model_rotation_mat.rotation_00_ * vX + model_rotation_mat.rotation_01_ * vY +
				model_rotation_mat.rotation_02_ * vZ + x_location;
			float tY = model_rotation_mat.rotation_10_ * vX + model_rotation_mat.rotation_11_ * vY +
				model_rotation_mat.rotation_12_ * vZ + y_location;
			float tZ = model_rotation_mat.rotation_20_ * vX + model_rotation_mat.rotation_21_ * vY +
				model_rotation_mat.rotation_22_ * vZ + z_location;

			/*Store for DRR Computations*/
			dev_transf_vertex_zs[i] = tZ;

			/*Transform normal and compute dot product with vertex. Backface if >=  0. Only do on first vertex.*/
			if (i % 3 == 0) {
				float nX = dev_normals[3 * (i / 3)];
				float nY = dev_normals[3 * (i / 3) + 1];
				float nZ = dev_normals[3 * (i / 3) + 2];
				float dotProduct = (model_rotation_mat.rotation_00_ * nX + model_rotation_mat.rotation_01_ * nY +
						model_rotation_mat.rotation_02_ * nZ) * tX +
					(model_rotation_mat.rotation_10_ * nX + model_rotation_mat.rotation_11_ * nY +
						model_rotation_mat.rotation_12_ * nZ) * tY +
					(model_rotation_mat.rotation_20_ * nX + model_rotation_mat.rotation_21_ * nY +
						model_rotation_mat.rotation_22_ * nZ) * tZ;
				if (dotProduct >= 0) {
					dev_backface[i / 3] = true;
				}
				else {
					dev_backface[i / 3] = false;
				}
				/* Also in dev_tangent_triangle Store  true If Triangle Intersects the origin if extended into a plane
				(aka the normal is orthogonal to the vector from the triangle to the origin). Use in DRR computations later.*/
				if (dotProduct == 0) {
					dev_tangent_triangle[i / 3] = true;
				}
				else {
					dev_tangent_triangle[i / 3] = false;
				}


			}

			if (tZ >= 0)
				tZ = -.000001; /*Can't be above or at zero, so make very small..should never happen*/

			float sX = (tX / tZ) * dist_over_pix_pitch + pix_conversion_x;
			float sY = (tY / tZ) * dist_over_pix_pitch + pix_conversion_y;

			/*Store Projected Triangles Actual Location*/
			dev_projected_triangles[(2 * i)] = sX;
			dev_projected_triangles[(2 * i) + 1] = sY;


			/*Store Nearest Pixel of Projected Triangles (Round >= X.5 up to X + 1 and < X.5 down to X)*/
			dev_projected_triangles_snapped[(2 * i)] = static_cast<int>(floorf(sX + 0.5));
			dev_projected_triangles_snapped[(2 * i) + 1] = static_cast<int>(floorf(sY + 0.5));

		}
	}

	__global__ void DRR_BoundingBoxForTrianglesKernel(int* dev_bounding_box_triangles,
	                                                  int* dev_projected_triangles_snapped,
	                                                  int triangle_count, int width, int height) {
		int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

		if (i < 4 * triangle_count) {
			int j = i % 4;
			if (j == 0) /* Bottom Left X */
			{
				int value =
					max(
						min(
							min(
								min(
									dev_projected_triangles_snapped[6 * (i / 4)],
									dev_projected_triangles_snapped[6 * (i / 4) + 2]),
								dev_projected_triangles_snapped[6 * (i / 4) + 4]),
							width - 1),
						0);
				dev_bounding_box_triangles[i] = value;
			}
			else if (j == 1) /* Bottom Left Y */
			{
				int value =
					max(
						min(
							min(
								min(
									dev_projected_triangles_snapped[6 * (i / 4) + 1],
									dev_projected_triangles_snapped[6 * (i / 4) + 3]),
								dev_projected_triangles_snapped[6 * (i / 4) + 5]),
							height - 1),
						0);
				dev_bounding_box_triangles[i] = value;
			}
			else if (j == 2) /* Top Right X */
			{
				int value =
					min(
						max(
							max(
								max(
									dev_projected_triangles_snapped[6 * (i / 4)],
									dev_projected_triangles_snapped[6 * (i / 4) + 2]),
								dev_projected_triangles_snapped[6 * (i / 4) + 4]),
							0),
						width - 1);
				dev_bounding_box_triangles[i] = value;
			}
			else if (j == 3) /* Top Right Y */
			{
				int value =
					min(
						max(
							max(
								max(
									dev_projected_triangles_snapped[6 * (i / 4) + 1],
									dev_projected_triangles_snapped[6 * (i / 4) + 3]),
								dev_projected_triangles_snapped[6 * (i / 4) + 5]),
							0),
						height - 1);
				dev_bounding_box_triangles[i] = value;
			}
		}
	}

	__global__ void DRR_BoundingBoxSizesKernel(int* dev_bounding_box_triangles, int* dev_bounding_box_triangles_sizes,
	                                           int triangle_count, int* dev_bounding_box) {
		int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

		if (i < triangle_count) {
			int fourI = 4 * i;

			/*Load bounding box corners (LX, BY, RX, TY) to register memory*/
			int leftX = dev_bounding_box_triangles[fourI];
			int bottomY = dev_bounding_box_triangles[fourI + 1];
			int rightX = dev_bounding_box_triangles[fourI + 2];
			int topY = dev_bounding_box_triangles[fourI + 3];

			dev_bounding_box_triangles_sizes[i] = (1 + rightX - leftX) *
				(1 + topY - bottomY);

			/*Store Bounding Box on Image*/
			atomicMin(&dev_bounding_box[0], leftX);
			atomicMin(&dev_bounding_box[1], bottomY);
			atomicMax(&dev_bounding_box[2], rightX);
			atomicMax(&dev_bounding_box[3], topY);
		}
	}

	__global__ void DRR_PrepareLaunchPacketKernel(int* dev_fragment_fill, int* dev_bounding_box_triangles_sizes,
	                                              int* dev_bounding_box_triangles_sizes_prefix,
	                                              int triangle_count) {
		dev_fragment_fill[0] = dev_bounding_box_triangles_sizes[triangle_count - 1] +
			dev_bounding_box_triangles_sizes_prefix[triangle_count - 1];
	}

	__global__ void DRR_StridePrefixKernel(int stride, int* dev_bounding_box_triangles_sizes,
	                                       int* dev_bounding_box_triangles_sizes_prefix,
	                                       int* dev_stride_prefixes, int triangle_count) {
		/*Should be slightly more then
		(boundingBoxTrianglesSizePrefix[triangleCount - 1] + boundingBoxTrianglesSize[triangleCount - 1] ) / stride
		*/
		int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

		int j = i * stride;

		if (j < dev_bounding_box_triangles_sizes_prefix[triangle_count - 1] + dev_bounding_box_triangles_sizes[
			triangle_count - 1]) {
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
				}
				else {
					high = mid;
				}
			}
			strideIndex = high - 1;
			dev_stride_prefixes[i] = strideIndex;
		}
	}

	__global__ void DRR_FillTriangleKernel(int* dev_bounding_box_triangles_sizes,
	                                       int* dev_bounding_box_triangles_sizes_prefix,
	                                       int* dev_bounding_box_triangles, float* dev_z_line_values,
	                                       int triangle_count, int width, int height,
	                                       float* dev_projected_triangles, float* dev_transf_vertex_zs,
	                                       int* dev_stride_prefixes, bool* dev_backface, bool* dev_tangent_triangle) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		if (i < dev_bounding_box_triangles_sizes_prefix[triangle_count - 1] + dev_bounding_box_triangles_sizes[
			triangle_count - 1]) {
			/*Index of Triangle for the given stride (stride is of size 256 and the stride group is blockIdx.x)*/
			int stridedIndex = dev_stride_prefixes[blockIdx.x];

			/*Load [stridedIndex, stridedIndex + 255] at most (256) elements to another shared memory (could hit upper bound)*/
			__shared__ int reducedBoundingBoxTrianglesSizePrefix[threads_per_block];
			if (threadIdx.x + stridedIndex < triangle_count)
				reducedBoundingBoxTrianglesSizePrefix[threadIdx.x] = dev_bounding_box_triangles_sizes_prefix[threadIdx.x
					+ stridedIndex];
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
				}
				else {
					high = mid;
				}
			}
			triangleIndex = high - 1 + stridedIndex;

			/*Calculate the Pixel to Evaluate (Corresponding to Thread)*/
			int triangleIndex4 = 4 * triangleIndex;
			int Lx = dev_bounding_box_triangles[triangleIndex4];
			int By = dev_bounding_box_triangles[triangleIndex4 + 1];
			int Rx = dev_bounding_box_triangles[triangleIndex4 + 2];
			int insideIndex = i - dev_bounding_box_triangles_sizes_prefix[triangleIndex];
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

			/*Read in Original Triangles Vertices' Z Points*/
			int triangleIndex3 = 3 * triangleIndex;
			float tvz1 = dev_transf_vertex_zs[triangleIndex3];
			float tvz2 = dev_transf_vertex_zs[triangleIndex3 + 1];
			float tvz3 = dev_transf_vertex_zs[triangleIndex3 + 2];

			if (denominator > 0) {
				if (0 <= a && a <= denominator) {
					float b = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3));
					if (0 <= b && b <= denominator) {
						float c = denominator - a - b;
						if (0 <= c && c <= denominator) {
							if (dev_tangent_triangle[triangleIndex] == false) {
								if (dev_backface[triangleIndex] == false)
									atomicAdd(&dev_z_line_values[pyPixel * width + pxPixel],
									          (tvz1 * tvz2 * tvz3 * denominator) / (a * tvz2 * tvz3 + b * tvz1 * tvz3 +
										          c * tvz1 * tvz2));
								else
									atomicAdd(&dev_z_line_values[pyPixel * width + pxPixel], -1 *
									          (tvz1 * tvz2 * tvz3 * denominator) / (a * tvz2 * tvz3 + b * tvz1 * tvz3 +
										          c * tvz1 * tvz2));
							}
						}
					}
				}
			}
			else {
				if (0 >= a && a >= denominator) {
					float b = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3));
					if (0 >= b && b >= denominator) {
						float c = denominator - a - b;
						if (0 >= c && c >= denominator) {
							if (dev_tangent_triangle[triangleIndex] == false) {
								if (dev_backface[triangleIndex] == false)
									atomicAdd(&dev_z_line_values[pyPixel * width + pxPixel],
									          (tvz1 * tvz2 * tvz3 * denominator) / (a * tvz2 * tvz3 + b * tvz1 * tvz3 +
										          c * tvz1 * tvz2));
								else
									atomicAdd(&dev_z_line_values[pyPixel * width + pxPixel], -1 *
									          (tvz1 * tvz2 * tvz3 * denominator) / (a * tvz2 * tvz3 + b * tvz1 * tvz3 +
										          c * tvz1 * tvz2));
							}

						}
					}
				}
			}

		}

	}

	__global__ void ZToLineIntegralToDRRConversionKernel(unsigned char* dev_image, float* dev_z_line_values, int width,
	                                                     int height,
	                                                     int diff_kernel_left_x, int diff_kernel_bottom_y,
	                                                     int diff_kernel_cropped_width, float lower_bound,
	                                                     float upper_bound,
	                                                     float pix_conversion_x, float pix_conversion_y,
	                                                     float pixel_pitch, float principal_distance) {
		/*Global Thread*/
		int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

		/*Convert to Subsize*/
		i = (i / diff_kernel_cropped_width) * width + (i % diff_kernel_cropped_width) + diff_kernel_bottom_y * width +
			diff_kernel_left_x;


		/*If Correct Width and Height*/
		if (i < width * height) {
			/*Preserve Zeros*/
			if (dev_z_line_values[i] == 0) {
				dev_image[i] = 0;
			}
			else {
				/*Convert Z Value for Pixel to Line Integral*/
				float tx = ((i % width) + 0.5 - pix_conversion_x) * pixel_pitch;
				float ty = ((i / width) + 0.5 - pix_conversion_y) * pixel_pitch;
				float z_integral_ = dev_z_line_values[i];
				float line_integral = (z_integral_ / principal_distance) * sqrt(
					tx * tx + ty * ty + principal_distance * principal_distance);
				/*Convert Line Integral to DRR (Uchar)*/
				if (line_integral >= upper_bound)
					dev_image[i] = 255;
				else if (line_integral <= lower_bound)
					dev_image[i] = 0;
				else
					dev_image[i] = 255 * ((line_integral - lower_bound) / (upper_bound - lower_bound));
			}
		}

	}


	/*Render Engine for DRRs*/
	cudaError_t RenderEngine::RenderDRR(float lower_bound, float upper_bound) {
		/*Create Error Status*/
		cudaGetLastError(); //Resets Errors (MAYBE DELETE TO SAVE TIME?)

		/*Clear Image*/
		cudaMemset(renderer_output_->GetDeviceImagePointer(), 0, width_ * height_ * sizeof(unsigned char));

		/*Clear Line Integral Values*/
		cudaMemset(dev_z_line_values_, 0, width_ * height_ * sizeof(float));

		/*Reset Launch Packet*/
		DRR_ResetKernel << <1, 1 >> >(dev_bounding_box_, width_, height_);

		/*Transform Points (Rotate then Translate) and Project to Screen and Snap*/
		DRR_WorldToPixelKernel << <dim_grid_vertices_, threads_per_block >> >(dev_triangles_, dev_projected_triangles_,
		                                                                      dev_projected_triangles_snapped_,
		                                                                      dev_transf_vertex_zs_,
		                                                                      3 * triangle_count_,
		                                                                      dist_over_pix_pitch_, pix_conversion_x_,
		                                                                      pix_conversion_y_,
		                                                                      model_pose_.x_location_,
		                                                                      model_pose_.y_location_,
		                                                                      model_pose_.z_location_,
		                                                                      model_rotation_mat_, dev_normals_,
		                                                                      dev_backface_, dev_tangent_triangle_);

		/*Calculate Bounding Boxes for Each Triangle*/
		DRR_BoundingBoxForTrianglesKernel << <dim_grid_bounding_box_, threads_per_block >> >(
			dev_bounding_box_triangles_,
			dev_projected_triangles_snapped_, triangle_count_, width_, height_);

		/*Calculate Sizes of Bounding Boxes and Overall Bounding Box of Model*/
		DRR_BoundingBoxSizesKernel << <dim_grid_triangles_, threads_per_block >> >(
			dev_bounding_box_triangles_, dev_bounding_box_triangles_sizes_,
			triangle_count_, dev_bounding_box_);

		/*Use CUB library to compute exlusive prefix sum of bound box sizes.*/
		cub::DeviceScan::ExclusiveSum(dev_cub_storage_, cub_storage_bytes_, dev_bounding_box_triangles_sizes_,
		                              dev_bounding_box_triangles_sizes_prefix_, triangle_count_);

		/*Prepare Launch Packet and Send it to Host*/
		/*Contains bounding box on white pixels (LX,BY,RX,TY, and # of fragments to process
		(last element in dev_boundingBoxTrianglesSizePrefix and last element in dev_boundingBoxTrianglesSize)*/
		DRR_PrepareLaunchPacketKernel << <1, 1 >> >(dev_fragment_fill_, dev_bounding_box_triangles_sizes_,
		                                            dev_bounding_box_triangles_sizes_prefix_, triangle_count_);

		cudaMemcpy(renderer_output_->GetBoundingBox(), dev_bounding_box_, 4 * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(fragment_fill_, dev_fragment_fill_, 1 * sizeof(int), cudaMemcpyDeviceToHost);


		/*Because doing a binary search for every fragement on the prefix search takes too long,
		we first do this on every 256th fragment, then load the above 256 prefix values into shared memory
		and binary search over those. The first part occurrs in the the StridePrefixKernel and the second
		occurs in the FillTriangleKernel. In the FillTriangleKernel we also shade (or not shade) the fragment
		based on the results of the point (center of pixel) in triangle test.*/

		/*Error check for too many fragments.*/
		if (static_cast<double>(fragment_fill_[0]) > static_cast<double>(maximum_stride_size) * static_cast<double>(
			threads_per_block - 1)) {
			fprintf(stderr, "Fragment overflow! Please shrink image and/or reduce model triangle count!");
			fragment_overflow_ = true;
			return cudaErrorMemoryAllocation;
		}

		DRR_StridePrefixKernel << < ceil(
				static_cast<double>(fragment_fill_[0]) / static_cast<double>(threads_per_block * threads_per_block)),
			threads_per_block >> >(
				threads_per_block, dev_bounding_box_triangles_sizes_,
				dev_bounding_box_triangles_sizes_prefix_, dev_stride_prefixes_, triangle_count_);

		DRR_FillTriangleKernel << <ceil(static_cast<double>(fragment_fill_[0]) / static_cast<double>(threads_per_block))
			, threads_per_block >> >(
				dev_bounding_box_triangles_sizes_, dev_bounding_box_triangles_sizes_prefix_,
				dev_bounding_box_triangles_, dev_z_line_values_,
				triangle_count_, width_, height_, dev_projected_triangles_, dev_transf_vertex_zs_, dev_stride_prefixes_,
				dev_backface_, dev_tangent_triangle_);

		/* Compute launch parameters for Line Integral to DRR transformation. Want same size as sub image  */
		int* bounding_box = renderer_output_->GetBoundingBox();
		int diff_kernel_left_x = max(bounding_box[0], 0);
		int diff_kernel_bottom_y = max(bounding_box[1], 0);
		int diff_kernel_right_x = min(bounding_box[2], width_ - 1);
		int diff_kernel_top_y = min(bounding_box[3], height_ - 1);
		int diff_kernel_cropped_width = diff_kernel_right_x - diff_kernel_left_x + 1;
		int diff_kernel_cropped_height = diff_kernel_top_y - diff_kernel_bottom_y + 1;

		auto dim_grid_image_processing_ = dim3::dim3(
			ceil(static_cast<double>(diff_kernel_cropped_width) / sqrt(static_cast<double>(threads_per_block))),
			ceil(static_cast<double>(diff_kernel_cropped_height) / sqrt(static_cast<double>(threads_per_block))));


		/*Converts Z Values to Integrals and then Converts Further to DRR Image*/
		ZToLineIntegralToDRRConversionKernel << <dim_grid_image_processing_, threads_per_block >> >(
			renderer_output_->GetDeviceImagePointer(), dev_z_line_values_,
			width_, height_, diff_kernel_left_x, diff_kernel_bottom_y, diff_kernel_cropped_width, lower_bound,
			upper_bound,
			pix_conversion_x_, pix_conversion_y_, camera_calibration_.pixel_pitch_,
			camera_calibration_.principal_distance_);

		/*Check for Errors*/
		return cudaGetLastError();

	}
}
