#include <iostream>
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


namespace{

/*
    Distance Calculator
*/

__device__ void print_vec3f(const float* p, const char* string)
{
    printf("%s, %f %f %f\n", string, p[0], p[1], p[2]);
}

__device__ float calc_squared_dist(const float *p1, const float *p2) {
    const float x = p1[0] - p2[0];
    const float y = p1[1] - p2[1];
    const float z = p1[2] - p2[2];
    return x*x + y*y + z*z;
}

__device__ double calc_squared_dist(const double *p1, const double *p2) {
    const double x = p1[0] - p2[0];
    const double y = p1[1] - p2[1];
    const double z = p1[2] - p2[2];
    return x*x + y*y + z*z;
}

__device__ float fast_sqrt(float v) { return __fsqrt_rn(v); }
__device__ double fast_sqrt(double v) { return __dsqrt_rn(v); }
__device__ void my_swap(int *a, int *b) {
	int tmp = *a;
	*a = *b;
	*b = tmp;
}
__device__ void my_swap(double *a, double *b) {
	double tmp = *a;
	*a = *b;
	*b = tmp;
}
__device__ void my_swap(float *a, float *b) {
	float tmp = *a;
	*a = *b;
	*b = tmp;
}

template<typename scalar_t>
__device__ void cross(const scalar_t* v1, const scalar_t* v2, scalar_t* n)
{
    n[0] = v1[1] * v2[2] - v1[2] * v2[1];
    n[1] = v1[2] * v2[0] - v1[0] * v2[2];
    n[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

template<typename scalar_t>
__device__ scalar_t dot(const scalar_t* v1, const scalar_t* v2)
{
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

template<typename scalar_t>
__device__ scalar_t closest_pt_on_plane(const scalar_t* v0, const scalar_t* v1, const scalar_t* v2, const scalar_t* p, scalar_t* closest_pt){
    scalar_t vec01[3] = {v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]};
    scalar_t vec02[3] = {v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]};
    scalar_t plane_normal[3];
    cross(vec01, vec02, plane_normal);
    scalar_t norm = fast_sqrt(dot(plane_normal, plane_normal));
    plane_normal[0] /= norm;
    plane_normal[1] /= norm;
    plane_normal[2] /= norm;

    scalar_t vec0_p[3] = {p[0]-v0[0], p[1]-v0[1], p[2]-v0[2]};
    scalar_t min_dist = dot(vec0_p, plane_normal);
    closest_pt[0] = p[0] - min_dist * plane_normal[0];
    closest_pt[1] = p[1] - min_dist * plane_normal[1];
    closest_pt[2] = p[2] - min_dist * plane_normal[2];
    min_dist = fabsf(min_dist);
    return min_dist;
}

template<typename scalar_t>
__device__ bool pt_in_triangle(const scalar_t* v0, const scalar_t* v1, const scalar_t* v2, const scalar_t* p)
{
    scalar_t a[3] = {v0[0] - p[0], v0[1] - p[1], v0[2] - p[2]};
    scalar_t b[3] = {v1[0] - p[0], v1[1] - p[1], v1[2] - p[2]};
    scalar_t c[3] = {v2[0] - p[0], v2[1] - p[1], v2[2] - p[2]};

    scalar_t u[3], v[3], w[3];
    cross(a, b, u);
    cross(b, c, v);
    cross(c, a, w);

    // printf("v0 %f %f %f, v1 %f %f %f, v2 %f %f %f, p %f %f %f, a %f %f %f, b %f %f %f, c %f %f %f, u %f %f %f, v %f %f %f, w %f %f %f,\n",
    //     v0[0],v0[1],v0[2],v1[0],v1[1],v1[2],v2[0],v2[1],v2[2],p[0],p[1],p[2],a[0],a[1],a[2],b[0],b[1],b[2],c[0],c[1],c[2],u[0],u[1],u[2],v[0],v[1],v[2],w[0],w[1],w[2]);

    if (dot(u, v) < 0.f){
        return false;
    }

    if (dot(u, w) < 0.f){
        return false;
    }

    return true;
}

template<typename scalar_t>
__device__ scalar_t closest_pt_on_line(const scalar_t* a, const scalar_t* b, const scalar_t* c, scalar_t* closest_pt){
    scalar_t ac[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    scalar_t ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    scalar_t t = dot(ac, ab) / dot(ab, ab);
    t = fminf(0.f, t);
    t = fmaxf(t, 1.f);
    closest_pt[0] = a[0] + t * ab[0];
    closest_pt[1] = a[1] + t * ab[1];
    closest_pt[2] = a[2] + t * ab[2];
    return fast_sqrt(calc_squared_dist(closest_pt, c));
}

template<typename scalar_t>
__device__ float closest_pt_in_triangle(const scalar_t* v0, const scalar_t* v1, const scalar_t* v2, const scalar_t* p, scalar_t* closest_pt){
    scalar_t min_dist = closest_pt_on_plane(v0, v1, v2, p, closest_pt);
    scalar_t inside_flag = pt_in_triangle(v0, v1, v2, closest_pt);
    // printf("v0 %f %f %f, v1 %f %f %f, v2 %f %f %f, p %f %f %f, clsest_pt %f %f %f, inside flag: %d\n",
        // v0[0],v0[1],v0[2],v1[0],v1[1],v1[2],v2[0],v2[1],v2[2],p[0],p[1],p[2],closest_pt[0],closest_pt[1],closest_pt[2], inside_flag);

    // inside test
    if (inside_flag){
       return min_dist;
    }

    scalar_t closest_pt_tmp[3];
    min_dist = closest_pt_on_line(v0, v1, p, closest_pt_tmp);
    closest_pt[0] = closest_pt_tmp[0]; closest_pt[1] = closest_pt_tmp[1]; closest_pt[2] = closest_pt_tmp[2];
    scalar_t d2 = closest_pt_on_line(v1, v2, p, closest_pt_tmp);
    if (d2 < min_dist) {closest_pt[0] = closest_pt_tmp[0]; closest_pt[1] = closest_pt_tmp[1]; closest_pt[2] = closest_pt_tmp[2]; min_dist = d2;}
    scalar_t d3 = closest_pt_on_line(v2, v0, p, closest_pt_tmp);
    if (d3 < min_dist) {closest_pt[0] = closest_pt_tmp[0]; closest_pt[1] = closest_pt_tmp[1]; closest_pt[2] = closest_pt_tmp[2]; min_dist = d3;}

    return min_dist;
}

template<typename scalar_t>
__global__ void nearest_face_kernel(
    const scalar_t* __restrict__ vertices,
    const int* __restrict__ faces,
    const scalar_t* __restrict__ queries,
    scalar_t* __restrict__ dist,
    int* __restrict__ face_ids,
    scalar_t* __restrict__ nearest_pts,
    int face_num,
    int query_num) {
    const int chunk = 512;
    __shared__ scalar_t buffer[chunk * 3 * 3];
    for (int fi = 0; fi < face_num; fi += chunk){
        int chunk_size = min(face_num, fi + chunk) - fi;
        for (int j = threadIdx.x; j < chunk_size; j += blockDim.x){
            int vidx0 = faces[3 * (fi + j) + 0], vidx1 = faces[3 * (fi + j) + 1], vidx2 = faces[3 * (fi + j) + 2];
            buffer[9 * j + 0] = vertices[3 * vidx0 + 0];
            buffer[9 * j + 1] = vertices[3 * vidx0 + 1];
            buffer[9 * j + 2] = vertices[3 * vidx0 + 2];
            buffer[9 * j + 3] = vertices[3 * vidx1 + 0];
            buffer[9 * j + 4] = vertices[3 * vidx1 + 1];
            buffer[9 * j + 5] = vertices[3 * vidx1 + 2];
            buffer[9 * j + 6] = vertices[3 * vidx2 + 0];
            buffer[9 * j + 7] = vertices[3 * vidx2 + 1];
            buffer[9 * j + 8] = vertices[3 * vidx2 + 2];
        }
        __syncthreads();

//         for (int query_idx = threadIdx.x + blockIdx.x * blockDim.x; query_idx < query_num; query_idx += blockDim.x * gridDim.x)
        int query_idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (query_idx < query_num)
        {
            const scalar_t query_pt[3] = {
                queries[3 * query_idx + 0], queries[3 * query_idx + 1], queries[3 * query_idx + 2]
            };
            int nearest_face_id = 0;
            scalar_t nearest_dist = FLT_MAX;
            scalar_t nearest_pt[3];
            for (int k = 0; k < chunk_size; k++){
                scalar_t nearest_pt_tmp[3];
                scalar_t d = closest_pt_in_triangle(buffer + 9 * k + 0, buffer + 9 * k + 3, buffer + 9 * k + 6, query_pt, nearest_pt_tmp);
                if (nearest_dist > d || k == 0){
                    nearest_face_id = k + fi;
                    nearest_dist = d;
                    nearest_pt[0] = nearest_pt_tmp[0];
                    nearest_pt[1] = nearest_pt_tmp[1];
                    nearest_pt[2] = nearest_pt_tmp[2];
                }
            }
            if (fi == 0 || dist[query_idx] > nearest_dist){
                dist[query_idx] = nearest_dist;
                face_ids[query_idx] = nearest_face_id;
                nearest_pts[3 * query_idx + 0] = nearest_pt[0];
                nearest_pts[3 * query_idx + 1] = nearest_pt[1];
                nearest_pts[3 * query_idx + 2] = nearest_pt[2];
            }
        }
        __syncthreads();
    }
}

}

void nearest_face(at::Tensor vertices, at::Tensor faces, at::Tensor queries, at::Tensor dist, at::Tensor face_ids, at::Tensor nearest_pts)
{
    const auto vertex_num = vertices.size(0);
    const auto query_num = queries.size(0);
    const auto face_num = faces.size(0);
    const int threads = 512;
    const dim3 blocks((query_num - 1) / threads + 1);
//     const int blocks = 256;
//     const int threads = 256;
    AT_DISPATCH_FLOATING_TYPES(vertices.type(), "nearest_face_kernel", ([&] {
        nearest_face_kernel<scalar_t><<<blocks, threads>>>(
            vertices.data<scalar_t>(),
            faces.data<int>(),
            queries.data<scalar_t>(),
            dist.data<scalar_t>(),
            face_ids.data<int>(),
            nearest_pts.data<scalar_t>(),
            face_num, query_num);
    }));

    //cudaError_t err = cudaGetLastError();
    //if (err != cudaSuccess)
    //    printf("Error in nearest_face_kernel: %s\n", cudaGetErrorString(err));
}