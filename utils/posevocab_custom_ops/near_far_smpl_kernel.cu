#include <ATen/ATen.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38F
#endif

#define EPSILON_ABS_ZERO 1e-10
#define EPSILON_DIV_ZERO 1e-4


// for the older gpus atomicAdd with double arguments does not exist
//#if  __CUDA_ARCH__ < 600 and defined(__CUDA_ARCH__)
//static __inline__ __device__ double atomicAdd(double* address, double val) {
//    unsigned long long int* address_as_ull = (unsigned long long int*)address;
//    unsigned long long int old = *address_as_ull, assumed;
//    do {
//        assumed = old;
//        old = atomicCAS(address_as_ull, assumed,
//                __double_as_longlong(val + __longlong_as_double(assumed)));
//    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) } while (assumed != old);
//    } while (assumed != old);
//    return __longlong_as_double(old);
//}
//#endif

__device__ float fast_sqrt(float v) { return __fsqrt_rn(v); }
__device__ double fast_sqrt(double v) { return __dsqrt_rn(v); }

template<typename scalar_t>
__device__ bool intersect_ball(const scalar_t* ray_o, const scalar_t* ray_d, const scalar_t* center, const scalar_t radius, scalar_t* near, scalar_t* far){
    scalar_t a = ray_d[0] * ray_d[0] + ray_d[1] * ray_d[1] + ray_d[2] * ray_d[2];
    scalar_t b = 2 * (ray_d[0] * (ray_o[0] - center[0]) + ray_d[1] * (ray_o[1] - center[1]) + ray_d[2] * (ray_o[2] - center[2]));
    scalar_t c = (ray_o[0] - center[0]) * (ray_o[0] - center[0]) + (ray_o[1] - center[1]) * (ray_o[1] - center[1]) + (ray_o[2] - center[2]) * (ray_o[2] - center[2]) - radius * radius;
    scalar_t d = b * b - 4 * a * c;
    if (d < 1e-8){
        return false;
    } else{
        *near = (-b - fast_sqrt(d)) / (2 * a);
        *far = (-b + fast_sqrt(d)) / (2 * a);
        return true;
    }
}


template<typename scalar_t>
__global__ void near_far_smpl_kernel(
    const scalar_t* __restrict__ vertices,
    const scalar_t* __restrict__ ray_o,
    const scalar_t* __restrict__ ray_d,
    scalar_t* __restrict__ near,
    scalar_t* __restrict__ far,
    bool* __restrict__ intersect_flag,
    const scalar_t radius,
    int vertex_num,
    int ray_num)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ray_num) {
        return;
    }
    const scalar_t ray_o_[3] = {
        ray_o[3 * tid + 0], ray_o[3 * tid + 1], ray_o[3 * tid + 2]
    };
    const scalar_t ray_d_[3] = {
        ray_d[3 * tid + 0], ray_d[3 * tid + 1], ray_d[3 * tid + 2]
    };

    near[tid] = FLT_MAX;
    far[tid] = -1.0;
    bool intersect_once = false;
    for (int vi = 0; vi < vertex_num; vi++)
    {
        const scalar_t v[3] = {vertices[3 * vi + 0], vertices[3 * vi + 1], vertices[3 * vi + 2]};

        scalar_t near_ = 0.0, far_ = 0.0;
        bool intersect = intersect_ball(ray_o_, ray_d_, v, radius, &near_, &far_);
        if (!intersect){
            continue;
        }else{
            intersect_once = true;
            near[tid] = fminf(near[tid], near_);
            far[tid] = fmaxf(far[tid], far_);
            intersect_flag[tid] = true;
        }
    }
    if (!intersect_once){
        near[tid] = 0.0;
        far[tid] = 0.0;
        intersect_flag[tid] = false;
    }
}



void near_far_smpl(at::Tensor vertices, at::Tensor ray_o, at::Tensor ray_d, at::Tensor near, at::Tensor far, at::Tensor intersect_flag, const float radius)
{
    const auto vertex_num = vertices.size(0);
    const auto ray_num = ray_o.size(0);
    const int threads = 512;
    const dim3 blocks((ray_num - 1) / threads + 1);
    AT_DISPATCH_FLOATING_TYPES(vertices.scalar_type(), "near_far_smpl_kernel", ([&] {
        near_far_smpl_kernel<scalar_t><<<blocks, threads>>>(
            vertices.data<scalar_t>(),
            ray_o.data<scalar_t>(),
            ray_d.data<scalar_t>(),
            near.data<scalar_t>(),
            far.data<scalar_t>(),
            intersect_flag.data<bool>(),
            radius,
            vertex_num,
            ray_num);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error in sdf_kernel: %s\n", cudaGetErrorString(err));

    return;
}