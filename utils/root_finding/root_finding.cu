#include <iostream>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/tuple.h>

#include "vector_operations.hpp"
#include "utils.h"


__device__ void grid_sample_3d(
    const float* _volume,  // (X, Y, Z, C),
    const float3 _v,
    const float3 _min_v,
    const float3 _max_v,
    const int3 _res,
    const int _channel_num,
    float* _val
){
    float3 normalized_v;  // [0, 1]
    normalized_v.x = (_v.x - _min_v.x) / (_max_v.x - _min_v.x);
    normalized_v.y = (_v.y - _min_v.y) / (_max_v.y - _min_v.y);
    normalized_v.z = (_v.z - _min_v.z) / (_max_v.z - _min_v.z);

    normalized_v.x = fmaxf(fminf(normalized_v.x, 1.f), 0.f);
    normalized_v.y = fmaxf(fminf(normalized_v.y, 1.f), 0.f);
    normalized_v.z = fmaxf(fminf(normalized_v.z, 1.f), 0.f);

    int x = static_cast<int>(::round((_res.x - 1) * normalized_v.x));
    int y = static_cast<int>(::round((_res.y - 1) * normalized_v.y));
    int z = static_cast<int>(::round((_res.z - 1) * normalized_v.z));

    int start = x * (_res.y * _res.z * _channel_num) + y * (_res.z * _channel_num) + z * _channel_num;
    for(int i=0; i<_channel_num; i++){
        _val[i] = _volume[start + i];
    }
    
    // if (_channel_num == 55)
    //     printf("%d %d %d %f %f %f %f\n", x, y, z, _val[0], _val[1], _val[2], _val[3]);
}


__global__ void root_finding_kernel(
    const float* __restrict__ _weight_volume,  // (X, Y, Z, J)
    const float* __restrict__ _grad_volume, // (X, Y, Z, J*3)
    const float* __restrict__ _xt,  // (B, N, 3)
    const float* __restrict__ _xc_init,  // (B, N, 3)
    const float* __restrict__ _jnt_mats,  // (B, J, 4, 4)
    const float* __restrict__ _bounds, // (2, 3)
    const int* __restrict__ _res, // (3)
    float* __restrict__ _xc_opt,  // (B, N, 3)
    const int batch_size,
    const int point_num,
    const int joint_num,
    const float lambda,
    const int iter_num = 10
){
    int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_idx >= batch_size * point_num){
        return;
    }

    int batch_idx = thread_idx / point_num;
    int point_idx = thread_idx % point_num;

    float3 xc = make_float3(_xc_init[3*thread_idx + 0], _xc_init[3*thread_idx + 1], _xc_init[3*thread_idx + 2]);
    float3 xt = make_float3(_xt[3*thread_idx+0], _xt[3*thread_idx+1], _xt[3*thread_idx+2]);
    
    mat34 jnt_mats[55];
    for(int j=0; j<joint_num; j++){
        mat34& jnt_mat = jnt_mats[j];
        int start_idx = batch_idx * joint_num * 4 * 4 + j * 4 * 4;
        jnt_mat.rot.m00() = _jnt_mats[start_idx+0];
        jnt_mat.rot.m01() = _jnt_mats[start_idx+1];
        jnt_mat.rot.m02() = _jnt_mats[start_idx+2];
        jnt_mat.rot.m10() = _jnt_mats[start_idx+4];
        jnt_mat.rot.m11() = _jnt_mats[start_idx+5];
        jnt_mat.rot.m12() = _jnt_mats[start_idx+6];
        jnt_mat.rot.m20() = _jnt_mats[start_idx+8];
        jnt_mat.rot.m21() = _jnt_mats[start_idx+9];
        jnt_mat.rot.m22() = _jnt_mats[start_idx+10];

        jnt_mat.trans.x = _jnt_mats[start_idx+3];
        jnt_mat.trans.y = _jnt_mats[start_idx+7];
        jnt_mat.trans.z = _jnt_mats[start_idx+11];
    }

    float3 min_xyz = make_float3(_bounds[0], _bounds[1], _bounds[2]);
    float3 max_xyz = make_float3(_bounds[3], _bounds[4], _bounds[5]);
    int3 res = make_int3(_res[0], _res[1], _res[2]);

    for (int iter_idx = 0; iter_idx < 10; iter_idx++){
        
        // construct jacobian (part 1)
        float lbs[55];
        grid_sample_3d(
            _weight_volume,
            xc,
            min_xyz,
            max_xyz,
            res,
            joint_num,
            lbs
        );
        mat34 pt_mat = mat34::zeros();
        for(int j=0; j<joint_num; j++){
            pt_mat = pt_mat + jnt_mats[j] * lbs[j];
        }
        mat33 J1 = pt_mat.rot;

        // construct jacobian (part 2)
        float lbs_grad[55 * 3];
        grid_sample_3d(
            _grad_volume,
            xc,
            min_xyz,
            max_xyz,
            res,
            joint_num*3,
            lbs_grad
        );
        mat33 J2 = mat33::zero();
        float3 fwd_pt = make_float3(0.f, 0.f, 0.f);
        for(int j=0; j<joint_num; j++){
            float3 fwd_pt_sep = jnt_mats[j] * xc;
            J2.m00() += fwd_pt_sep.x * lbs_grad[3*j+0];
            J2.m01() += fwd_pt_sep.x * lbs_grad[3*j+1];
            J2.m02() += fwd_pt_sep.x * lbs_grad[3*j+2];
            J2.m10() += fwd_pt_sep.y * lbs_grad[3*j+0];
            J2.m11() += fwd_pt_sep.y * lbs_grad[3*j+1];
            J2.m12() += fwd_pt_sep.y * lbs_grad[3*j+2];
            J2.m20() += fwd_pt_sep.z * lbs_grad[3*j+0];
            J2.m21() += fwd_pt_sep.z * lbs_grad[3*j+1];
            J2.m22() += fwd_pt_sep.z * lbs_grad[3*j+2];

            fwd_pt = fwd_pt + lbs[j] * fwd_pt_sep;
        }

        // update
        mat33 J_inv = (J1 + J2 * lambda).inverse();
        // mat33 J_inv = J1.inverse();
        float3 delta = fwd_pt - xt;
        float3 update = J_inv * delta;
        update.x = fmaxf(fminf(update.x, 0.01), -0.01);
        update.y = fmaxf(fminf(update.y, 0.01), -0.01);
        update.z = fmaxf(fminf(update.z, 0.01), -0.01);

        xc = xc - update;
    }

    _xc_opt[3*thread_idx+0] = xc.x;
    _xc_opt[3*thread_idx+1] = xc.y;
    _xc_opt[3*thread_idx+2] = xc.z;
}


void root_finding(
    at::Tensor _weight_volume,  // (X, Y, Z, J)
    at::Tensor _grad_volume, // (X, Y, Z, J*3)
    at::Tensor _xt,  // (B, N, 3)
    at::Tensor _xc_init,  // (B, N, 3)
    at::Tensor _jnt_mats,  // (B, J, 4, 4)
    at::Tensor _bounds, // (2, 3)
    at::Tensor _res, // (3)
    at::Tensor _xc_opt,
    float _lambda,
    int _iter_num
)
{
    const int batch_size = _xt.size(0);
    const int point_num = _xt.size(1);
    const int joint_num = _weight_volume.size(3);

    const int threads = 512;
    const dim3 blocks((batch_size * point_num - 1) / threads + 1);

    CHECK_CONTIGUOUS_CUDA(_weight_volume);
    CHECK_CONTIGUOUS_CUDA(_grad_volume);
    CHECK_CONTIGUOUS_CUDA(_xt);
    CHECK_CONTIGUOUS_CUDA(_xc_init);
    CHECK_CONTIGUOUS_CUDA(_jnt_mats);
    CHECK_CONTIGUOUS_CUDA(_bounds);
    CHECK_CONTIGUOUS_CUDA(_res);
    CHECK_CONTIGUOUS_CUDA(_xc_opt);

    CHECK_IS_FLOAT(_weight_volume);
    CHECK_IS_FLOAT(_grad_volume);
    CHECK_IS_FLOAT(_xt);
    CHECK_IS_FLOAT(_xc_init);
    CHECK_IS_FLOAT(_jnt_mats);
    CHECK_IS_FLOAT(_bounds);
    CHECK_IS_INT(_res);
    CHECK_IS_FLOAT(_xc_opt);

    root_finding_kernel<<<blocks, threads>>>(
        _weight_volume.data<float>(),
        _grad_volume.data<float>(),
        _xt.data<float>(),
        _xc_init.data<float>(),
        _jnt_mats.data<float>(),
        _bounds.data<float>(),
        _res.data<int>(),
        _xc_opt.data<float>(),
        batch_size,
        point_num,
        joint_num,
        _lambda,
        _iter_num
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error in root_finding_kernel: %s\n", cudaGetErrorString(err));

    cudaDeviceSynchronize();

}


