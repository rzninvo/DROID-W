/* 
 * Copyright (c) 2021, Princeton Vision & Learning Lab (DROID-SLAM Authors)
 * All rights reserved.
 * 
 * This source code is licensed under the BSD 3-Clause License found in the
 * LICENSE file in the root directory of this source tree.
 * 
 * References:
 *   https://github.com/princeton-vl/DROID-SLAM
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

// #include "utils.cuh"

#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;
typedef std::vector<std::vector<long>> graph_t;
typedef std::vector<torch::Tensor> tensor_list_t;



#define MIN_DEPTH 0.25

#define THREADS 256
#define NUM_BLOCKS(batch_size) ((batch_size + THREADS - 1) / THREADS)


#define GPU_1D_KERNEL_LOOP(k, n) \
  for (size_t k = threadIdx.x; k<n; k += blockDim.x)

// Fast integer division: replace expensive k/wd and k%wd with multiply+subtract
__device__ __forceinline__ void fast_divmod(int k, int wd, int &i, int &j) {
  i = __float2int_rd(__fdividef((float)k, (float)wd));
  j = k - i * wd;
}


__device__ void warpReduce(volatile float *sdata, unsigned int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid +  8];
  sdata[tid] += sdata[tid +  4];
  sdata[tid] += sdata[tid +  2];
  sdata[tid] += sdata[tid +  1];
}

__device__ void blockReduce(volatile float *sdata) {
  unsigned int tid = threadIdx.x;
  __syncthreads();

  // if (threadIdx.x < 256) {sdata[tid] += sdata[tid + 256]; } __syncthreads();
  if (threadIdx.x < 128) {sdata[tid] += sdata[tid + 128]; } __syncthreads();
  if (threadIdx.x <  64) {sdata[tid] += sdata[tid +  64]; } __syncthreads();

  if (tid < 32) warpReduce(sdata, tid);
  __syncthreads();
}


__device__ void
actSO3(const float *q, const float *X, float *Y) {
  float uv[3];
  uv[0] = 2.0 * (q[1]*X[2] - q[2]*X[1]);
  uv[1] = 2.0 * (q[2]*X[0] - q[0]*X[2]);
  uv[2] = 2.0 * (q[0]*X[1] - q[1]*X[0]);

  Y[0] = X[0] + q[3]*uv[0] + (q[1]*uv[2] - q[2]*uv[1]);
  Y[1] = X[1] + q[3]*uv[1] + (q[2]*uv[0] - q[0]*uv[2]);
  Y[2] = X[2] + q[3]*uv[2] + (q[0]*uv[1] - q[1]*uv[0]);
}

__device__  void
actSE3(const float *t, const float *q, const float *X, float *Y) {
  actSO3(q, X, Y);
  Y[3] = X[3];
  Y[0] += X[3] * t[0];
  Y[1] += X[3] * t[1];
  Y[2] += X[3] * t[2];
}

__device__ void
adjSE3(const float *t, const float *q, const float *X, float *Y) {
  float qinv[4] = {-q[0], -q[1], -q[2], q[3]};
  actSO3(qinv, &X[0], &Y[0]);
  actSO3(qinv, &X[3], &Y[3]);

  float u[3], v[3];
  u[0] = t[2]*X[1] - t[1]*X[2];
  u[1] = t[0]*X[2] - t[2]*X[0];
  u[2] = t[1]*X[0] - t[0]*X[1];

  actSO3(qinv, u, v);
  Y[3] += v[0];
  Y[4] += v[1];
  Y[5] += v[2];
}

__device__ void 
relSE3(const float *ti, const float *qi, const float *tj, const float *qj, float *tij, float *qij) {
  qij[0] = -qj[3] * qi[0] + qj[0] * qi[3] - qj[1] * qi[2] + qj[2] * qi[1],
  qij[1] = -qj[3] * qi[1] + qj[1] * qi[3] - qj[2] * qi[0] + qj[0] * qi[2],
  qij[2] = -qj[3] * qi[2] + qj[2] * qi[3] - qj[0] * qi[1] + qj[1] * qi[0],
  qij[3] =  qj[3] * qi[3] + qj[0] * qi[0] + qj[1] * qi[1] + qj[2] * qi[2],

  actSO3(qij, ti, tij);
  tij[0] = tj[0] - tij[0];
  tij[1] = tj[1] - tij[1];
  tij[2] = tj[2] - tij[2];
}

  
__device__ void
expSO3(const float *phi, float* q) {
  // SO3 exponential map
  float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  float theta_p4 = theta_sq * theta_sq;

  float theta = sqrtf(theta_sq);
  float imag, real;

  if (theta_sq < 1e-8) {
    imag = 0.5 - (1.0/48.0)*theta_sq + (1.0/3840.0)*theta_p4;
    real = 1.0 - (1.0/ 8.0)*theta_sq + (1.0/ 384.0)*theta_p4;
  } else {
    imag = sinf(0.5 * theta) / theta;
    real = cosf(0.5 * theta);
  }

  q[0] = imag * phi[0];
  q[1] = imag * phi[1];
  q[2] = imag * phi[2];
  q[3] = real;

}

__device__ void
crossInplace(const float* a, float *b) {
  float x[3] = {
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0], 
  };

  b[0] = x[0];
  b[1] = x[1];
  b[2] = x[2];
}

__device__ void
expSE3(const float *xi, float* t, float* q) {
  // SE3 exponential map

  expSO3(xi + 3, q);
  float tau[3] = {xi[0], xi[1], xi[2]};
  float phi[3] = {xi[3], xi[4], xi[5]};

  float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  float theta = sqrtf(theta_sq);

  t[0] = tau[0]; 
  t[1] = tau[1]; 
  t[2] = tau[2];

  if (theta > 1e-4) {
    float a = (1 - cosf(theta)) / theta_sq;
    crossInplace(phi, tau);
    t[0] += a * tau[0];
    t[1] += a * tau[1];
    t[2] += a * tau[2];

    float b = (theta - sinf(theta)) / (theta * theta_sq);
    crossInplace(phi, tau);
    t[0] += b * tau[0];
    t[1] += b * tau[1];
    t[2] += b * tau[2];
  }
}
__global__ void projective_transform_kernel(
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> target,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> uncertainties,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Hs,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> vs,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Eii,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Eij,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Cii,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> bz,
    const bool enable_udba,
    const bool enable_bidirectional_uncer)
{
  const int block_id = blockIdx.x;        // every block handles one edge
  const int thread_id = threadIdx.x;      // every thread handles one pixel

  const int ht = disps.size(1);
  const int wd = disps.size(2);

  int ix = static_cast<int>(ii[block_id]);
  int jx = static_cast<int>(jj[block_id]);

  __shared__ float fx;
  __shared__ float fy;
  __shared__ float cx;
  __shared__ float cy;

  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];

  // load intrinsics from global memory
  if (thread_id == 0) {
    fx = intrinsics[0];
    fy = intrinsics[1];
    cx = intrinsics[2];
    cy = intrinsics[3];
  }

  __syncthreads();

  // stereo frames
  if (ix == jx) {
    if (thread_id == 0) {
      tij[0] =  -0.1;
      tij[1] =     0;
      tij[2] =     0;
      qij[0] =     0;
      qij[1] =     0;
      qij[2] =     0;
      qij[3] =     1;
    }
  }

  else {

    // load poses from global memory
    if (thread_id < 3) {
      ti[thread_id] = poses[ix][thread_id];
      tj[thread_id] = poses[jx][thread_id];
    }

    if (thread_id < 4) {
      qi[thread_id] = poses[ix][thread_id+3];
      qj[thread_id] = poses[jx][thread_id+3];
    }

    __syncthreads();

    if (thread_id == 0) {
      relSE3(ti, qi, tj, qj, tij, qij);
    }
  }

  __syncthreads();

  //points 
  float Xi[4];
  float Xj[4];

  // jacobians
  float Jx[12];
  float Jz;

  float* Ji = &Jx[0];
  float* Jj = &Jx[6];

  // hessians
  float hij[12*(12+1)/2];

  float vi[6], vj[6];

  int l;
  for (l=0; l<12*(12+1)/2; l++) {
    hij[l] = 0;
  }

  for (int n=0; n<6; n++) {
    vi[n] = 0;
    vj[n] = 0;
  }

  __syncthreads();

  GPU_1D_KERNEL_LOOP(k, ht*wd) {

    int i, j;
    fast_divmod(k, wd, i, j);

    const float u = static_cast<float>(j);
    const float v = static_cast<float>(i);
    
    // homogenous coordinates
    Xi[0] = (u - cx) / fx;
    Xi[1] = (v - cy) / fy;
    Xi[2] = 1;
    Xi[3] = disps[ix][i][j];  // homogenous coords [x,y,1,h]

    // transform homogenous point
    actSE3(tij, qij, Xi, Xj);   // frame i to frame j  Xj = (R * Xi * 1/h) + t = 1/h * (R * Xi + h * t)
    // h * Xj = R * Xi + h * t => final we get h * Xj

    const float x = Xj[0];
    const float y = Xj[1];
    const float h = Xj[3];    // disparity of Xi

    const float d = (Xj[2] < MIN_DEPTH) ? 0.0 : 1.0 / Xj[2];
    const float d2 = d * d;

    float wu = (Xj[2] < MIN_DEPTH) ? 0.0 : .001 * weight[block_id][0][i][j];
    float wv = (Xj[2] < MIN_DEPTH) ? 0.0 : .001 * weight[block_id][1][i][j];

    if (enable_udba) {
      float scale_uncer = fmaxf(45.0 * uncertainties[ix][i][j] - 35.0, 0.1f);
      float w_uncer = fmaxf(fminf(1.0 / scale_uncer, 1.0f), 0.0f);
      wu = wu * w_uncer;
      wv = wv * w_uncer;
      if (enable_bidirectional_uncer) {
        float target_x = target[block_id][0][i][j];
        float target_y = target[block_id][1][i][j];
        const bool inside = (target_x >= 0.0f && target_x <= (float)(wd - 1) &&
                            target_y >= 0.0f && target_y <= (float)(ht - 1));
        if (inside) {
          // interpolate the uncertainty from the neighboring pixels
          const int x0 = (int)floorf(target_x);
          const int y0 = (int)floorf(target_y);
          const int x1 = min(x0 + 1, int(wd) - 1);
          const int y1 = min(y0 + 1, int(ht) - 1);
    
          const float du = target_x - (float)x0;
          const float dv = target_y - (float)y0;
    
          const float w00 = (1.0f - du) * (1.0f - dv);
          const float w01 = du * (1.0f - dv);
          const float w10 = (1.0f - du) * dv;
          const float w11 = du * dv;
  
          float u_j00 = uncertainties[jx][y0][x0];  
          float u_j01 = uncertainties[jx][y0][x1];
          float u_j10 = uncertainties[jx][y1][x0];
          float u_j11 = uncertainties[jx][y1][x1];
          float uncer_target = w00 * u_j00 + w01 * u_j01 + w10 * u_j10 + w11 * u_j11;
          float scale_uncer_target = fmaxf(45.0 * uncer_target - 35.0, 0.1f);
          float w_uncer_target = fmaxf(fminf(1.0 / scale_uncer_target, 1.0f), 0.0f);
          wu = wu * w_uncer_target;
          wv = wv * w_uncer_target;
        }
      }
    }

    const float ru = target[block_id][0][i][j] - (fx * d * x + cx);   // compute residuals, fx * d * x + cx is the reprojection of the point in the second frame
    const float rv = target[block_id][1][i][j] - (fy * d * y + cy);

    // x - coordinate
    // u = fx * Xj_x / Xj_z + cx
    // u = fy * Xj_y / Xj_z + cy
    Jj[0] = fx * (h*d);         // ∂u/∂t_x (only relevant with x)
    Jj[1] = fx * 0;             // ∂u/∂t_y
    Jj[2] = fx * (-x*h*d2);     // ∂u/∂t_z (relevant with d)
    Jj[3] = fx * (-x*y*d2);     // ∂u/∂w_x 
    Jj[4] = fx * (1 + x*x*d2);  // ∂u/∂w_y
    Jj[5] = fx * (-y*d);        // ∂u/∂w_z
    // different J computation for the rotation: non-linear perturbation, but for the translation we use a linear approximation

    Jz = fx * (tij[0] * d - tij[2] * (x * d2));   // relevant with x & d, x = R * Xi + h * t[0], d = 1 / (R * Xi + h * t[2])
    // fx * (∂d/∂h * x + d * ∂x/∂h) = fx * ( - d2 * t[2] * x + d * t[0])
    Cii[block_id][k] = wu * Jz * Jz;      // block_id: the index of image pair
    bz[block_id][k] = wu * ru * Jz;

    if (ix == jx) wu = 0;

    adjSE3(tij, qij, Jj, Ji);
    for (int n=0; n<6; n++) Ji[n] *= -1;

    l=0;
    for (int n=0; n<12; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += wu * Jx[n] * Jx[m];
        l++;
      }
    }

    for (int n=0; n<6; n++) {
      vi[n] += wu * ru * Ji[n];
      vj[n] += wu * ru * Jj[n];

      Eii[block_id][n][k] = wu * Jz * Ji[n];    // wu: weight
      Eij[block_id][n][k] = wu * Jz * Jj[n];
    }

    // y - coordinate
    Jj[0] = fy * 0;
    Jj[1] = fy * (h*d);
    Jj[2] = fy * (-y*h*d2);
    Jj[3] = fy * (-1 - y*y*d2);
    Jj[4] = fy * (x*y*d2);
    Jj[5] = fy * (x*d);

    Jz = fy * (tij[1] * d - tij[2] * (y * d2));
    Cii[block_id][k] += wv * Jz * Jz;
    bz[block_id][k] += wv * rv * Jz;

    if (ix == jx) wv = 0;

    adjSE3(tij, qij, Jj, Ji);
    for (int n=0; n<6; n++) Ji[n] *= -1;

    l=0;
    for (int n=0; n<12; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += wv * Jx[n] * Jx[m];     // Jx = [Ji, Jj]
        l++;
      }
    }

    for (int n=0; n<6; n++) {
      vi[n] += wv * rv * Ji[n];
      vj[n] += wv * rv * Jj[n];

      Eii[block_id][n][k] += wv * Jz * Ji[n];
      Eij[block_id][n][k] += wv * Jz * Jj[n];
    }


  }

  __syncthreads();

  __shared__ float sdata[THREADS];
  for (int n=0; n<6; n++) {
    sdata[threadIdx.x] = vi[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      vs[0][block_id][n] = sdata[0];
    }

    __syncthreads();

    sdata[threadIdx.x] = vj[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      vs[1][block_id][n] = sdata[0];
    }

  }

  l=0;
  for (int n=0; n<12; n++) {
    for (int m=0; m<=n; m++) {
      sdata[threadIdx.x] = hij[l];
      blockReduce(sdata);

      if (threadIdx.x == 0) {
        if (n<6 && m<6) {
          Hs[0][block_id][n][m] = sdata[0];
          Hs[0][block_id][m][n] = sdata[0];
        }
        else if (n >=6 && m<6) {
          Hs[1][block_id][m][n-6] = sdata[0];
          Hs[2][block_id][n-6][m] = sdata[0];
        }
        else {
          Hs[3][block_id][n-6][m-6] = sdata[0];
          Hs[3][block_id][m-6][n-6] = sdata[0];
        }
      }

      l++;
    }
  }
}


__global__ void projmap_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> coords,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> valid)
{

  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;

  const int ht = disps.size(1);
  const int wd = disps.size(2);

  __shared__ int ix;
  __shared__ int jx;

  __shared__ float fx;
  __shared__ float fy;
  __shared__ float cx;
  __shared__ float cy;

  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];

  // load intrinsics from global memory
  if (thread_id == 0) {
    ix = static_cast<int>(ii[block_id]);
    jx = static_cast<int>(jj[block_id]);
    fx = intrinsics[0];
    fy = intrinsics[1];
    cx = intrinsics[2];
    cy = intrinsics[3];
  }

  __syncthreads();

  // load poses from global memory
  if (thread_id < 3) {
    ti[thread_id] = poses[ix][thread_id];
    tj[thread_id] = poses[jx][thread_id];
  }

  if (thread_id < 4) {
    qi[thread_id] = poses[ix][thread_id+3];
    qj[thread_id] = poses[jx][thread_id+3];
  }

  __syncthreads();

  if (thread_id == 0) {
    relSE3(ti, qi, tj, qj, tij, qij);
  }

  //points 
  float Xi[4];
  float Xj[4];

  __syncthreads();

  GPU_1D_KERNEL_LOOP(k, ht*wd) {
    int i, j;
    fast_divmod(k, wd, i, j);

    const float u = static_cast<float>(j);
    const float v = static_cast<float>(i);
    
    // homogenous coordinates
    Xi[0] = (u - cx) / fx;
    Xi[1] = (v - cy) / fy;
    Xi[2] = 1;
    Xi[3] = disps[ix][i][j];

    // transform homogenous point
    actSE3(tij, qij, Xi, Xj);

    coords[block_id][i][j][0] = u;
    coords[block_id][i][j][1] = v;

    if (Xj[2] > 0.01) {
      coords[block_id][i][j][0] = fx * (Xj[0] / Xj[2]) + cx;
      coords[block_id][i][j][1] = fy * (Xj[1] / Xj[2]) + cy;
    }

    valid[block_id][i][j][0] = (Xj[2] > MIN_DEPTH) ? 1.0 : 0.0;

  }
}

__global__ void frame_distance_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> dist,
    const float beta) {

  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;

  const int ht = disps.size(1);
  const int wd = disps.size(2);

  __shared__ int ix;
  __shared__ int jx;

  __shared__ float fx;
  __shared__ float fy;
  __shared__ float cx;
  __shared__ float cy;

  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];

  // load intrinsics from global memory
  if (thread_id == 0) {
    ix = static_cast<int>(ii[block_id]);
    jx = static_cast<int>(jj[block_id]);
    fx = intrinsics[0];
    fy = intrinsics[1];
    cx = intrinsics[2];
    cy = intrinsics[3];
  }

  __syncthreads();


  //points 
  float Xi[4];
  float Xj[4];

  __shared__ float accum[THREADS]; accum[thread_id] = 0;
  __shared__ float valid[THREADS]; valid[thread_id] = 0;
  __shared__ float total[THREADS]; total[thread_id] = 0;

  __syncthreads();

  for (int n=0; n<1; n++) {

    if (thread_id < 3) {
      ti[thread_id] = poses[ix][thread_id];
      tj[thread_id] = poses[jx][thread_id];
    }

    if (thread_id < 4) {
      qi[thread_id] = poses[ix][thread_id+3];
      qj[thread_id] = poses[jx][thread_id+3];
    }

    __syncthreads();


    relSE3(ti, qi, tj, qj, tij, qij);

    float d, du, dv;

    GPU_1D_KERNEL_LOOP(k, ht*wd) {
      const int i = k / wd;
      const int j = k % wd;

      const float u = static_cast<float>(j);
      const float v = static_cast<float>(i);


      // if (disps[ix][i][j] < 0.01) {
      //   continue;
      // }
      
      // homogenous coordinates
      Xi[0] = (u - cx) / fx;
      Xi[1] = (v - cy) / fy;
      Xi[2] = 1;
      Xi[3] = disps[ix][i][j];

      // transform homogenous point
      actSE3(tij, qij, Xi, Xj);

      du = fx * (Xj[0] / Xj[2]) + cx - u;
      dv = fy * (Xj[1] / Xj[2]) + cy - v;
      d = sqrtf(du*du + dv*dv);

      total[threadIdx.x] += beta;
      
      if (Xj[2] > MIN_DEPTH) {
        accum[threadIdx.x] += beta * d;
        valid[threadIdx.x] += beta;
      }

      Xi[0] = (u - cx) / fx;
      Xi[1] = (v - cy) / fy;
      Xi[2] = 1;
      Xi[3] = disps[ix][i][j];

      Xj[0] = Xi[0] + Xi[3] * tij[0];
      Xj[1] = Xi[1] + Xi[3] * tij[1];
      Xj[2] = Xi[2] + Xi[3] * tij[2];

      du = fx * (Xj[0] / Xj[2]) + cx - u;
      dv = fy * (Xj[1] / Xj[2]) + cy - v;
      d = sqrtf(du*du + dv*dv);

      total[threadIdx.x] += (1 - beta);
      
      if (Xj[2] > MIN_DEPTH) {
        accum[threadIdx.x] += (1 - beta) * d;
        valid[threadIdx.x] += (1 - beta);
      }
    }

    if (threadIdx.x == 0) {
      int tmp = ix;
      ix = jx;
      jx = tmp;
    }

    __syncthreads();

  }
  __syncthreads(); blockReduce(accum);
  __syncthreads(); blockReduce(total);
  __syncthreads(); blockReduce(valid);

  __syncthreads();

  if (thread_id == 0) {
    dist[block_id] = (valid[0] / (total[0] + 1e-8) < 0.75) ? 1000.0 : accum[0] / valid[0];
  }
}



__global__ void depth_filter_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> inds,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> thresh,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> counter)
{

  const int block_id = blockIdx.x;
  const int neigh_id = blockIdx.y;
  const int index = blockIdx.z * blockDim.x + threadIdx.x;

  // if (threadIdx.x == 0) {
  //   printf("%d %d %d %d\n", blockIdx.x, blockIdx.y, blockDim.x, threadIdx.x);
  // }

  const int num = disps.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  __shared__ int ix;
  __shared__ int jx;

  __shared__ float fx;
  __shared__ float fy;
  __shared__ float cx;
  __shared__ float cy;

  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];

  if (threadIdx.x == 0) {
    ix = static_cast<int>(inds[block_id]);
    jx = (neigh_id < 3) ? ix - neigh_id - 1 : ix + neigh_id;
    fx = intrinsics[0];
    fy = intrinsics[1];
    cx = intrinsics[2];
    cy = intrinsics[3];
  }

  __syncthreads();

  if (jx < 0 || jx >= num) {
    return;
  }

  const float t = thresh[block_id];

  // load poses from global memory
  if (threadIdx.x < 3) {
    ti[threadIdx.x] = poses[ix][threadIdx.x];
    tj[threadIdx.x] = poses[jx][threadIdx.x];
  }

  if (threadIdx.x < 4) {
    qi[threadIdx.x] = poses[ix][threadIdx.x+3];
    qj[threadIdx.x] = poses[jx][threadIdx.x+3];
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    relSE3(ti, qi, tj, qj, tij, qij);
  }

  //points 
  float Xi[4];
  float Xj[4];

  __syncthreads();

  if (index < ht*wd) {
    int i, j;
    fast_divmod(index, wd, i, j);

    const float ui = static_cast<float>(j);
    const float vi = static_cast<float>(i);
    const float di = disps[ix][i][j];
    
    // homogenous coordinates
    Xi[0] = (ui - cx) / fx;
    Xi[1] = (vi - cy) / fy;
    Xi[2] = 1;
    Xi[3] = di;

    // transform homogenous point
    actSE3(tij, qij, Xi, Xj);

    const float uj = fx * (Xj[0] / Xj[2]) + cx;
    const float vj = fy * (Xj[1] / Xj[2]) + cy;
    const float dj = Xj[3] / Xj[2];

    const int u0 = static_cast<int>(floor(uj));
    const int v0 = static_cast<int>(floor(vj));

    if (u0 >= 0 && v0 >= 0 && u0 < wd-1 && v0 < ht-1) {
      const float wx = ceil(uj) - uj;
      const float wy = ceil(vj) - vj;

      const float d00 = disps[jx][v0+0][u0+0];
      const float d01 = disps[jx][v0+0][u0+1];
      const float d10 = disps[jx][v0+1][u0+0];
      const float d11 = disps[jx][v0+1][u0+1];

      const float dj_hat = wy*wx*d00 + wy*(1-wx)*d01 + (1-wy)*wx*d10 + (1-wy)*(1-wx)*d11;

      const float err = abs(1.0/dj - 1.0/dj_hat);
      if       (abs(1.0/dj - 1.0/d00) < t) atomicAdd(&counter[block_id][i][j], 1.0f);
      else if  (abs(1.0/dj - 1.0/d01) < t) atomicAdd(&counter[block_id][i][j], 1.0f);
      else if  (abs(1.0/dj - 1.0/d10) < t) atomicAdd(&counter[block_id][i][j], 1.0f);
      else if  (abs(1.0/dj - 1.0/d11) < t) atomicAdd(&counter[block_id][i][j], 1.0f);
    }
  }
}



__global__ void iproj_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> points)

{

  const int block_id = blockIdx.x;
  const int index = blockIdx.y * blockDim.x + threadIdx.x;


  const int num = disps.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  __shared__ float fx;
  __shared__ float fy;
  __shared__ float cx;
  __shared__ float cy;

  __shared__ float t[3];
  __shared__ float q[4];

  if (threadIdx.x == 0) {
    fx = intrinsics[0];
    fy = intrinsics[1];
    cx = intrinsics[2];
    cy = intrinsics[3];
  }

  __syncthreads();


  // load poses from global memory
  if (threadIdx.x < 3) {
    t[threadIdx.x] = poses[block_id][threadIdx.x];
  }

  if (threadIdx.x < 4) {
    q[threadIdx.x] = poses[block_id][threadIdx.x+3];
  }

  __syncthreads();

  //points 
  float Xi[4];
  float Xj[4];

  if (index < ht*wd) {
    int i, j;
    fast_divmod(index, wd, i, j);

    const float ui = static_cast<float>(j);
    const float vi = static_cast<float>(i);
    const float di = disps[block_id][i][j];
    
    // homogenous coordinates
    Xi[0] = (ui - cx) / fx;
    Xi[1] = (vi - cy) / fy;
    Xi[2] = 1;
    Xi[3] = di;

    // transform homogenous point
    actSE3(t, q, Xi, Xj);

    points[block_id][i][j][0] = Xj[0] / Xj[3];
    points[block_id][i][j][1] = Xj[1] / Xj[3];
    points[block_id][i][j][2] = Xj[2] / Xj[3];

  }
}



__global__ void accum_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> inps,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ptrs,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> idxs,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> outs)
{
  
  const int block_id = blockIdx.x;
  const int D = inps.size(2);

  const int start = ptrs[block_id];
  const int end = ptrs[block_id+1];

  for (int k=threadIdx.x; k<D; k+=blockDim.x) {
    float x = 0;
    for (int i=start; i<end; i++) {
      x += inps[idxs[i]][k];
    }
    outs[block_id][k] = x;
  }  
}


__device__ void
retrSE3(const float *xi, const float* t, const float* q, float* t1, float* q1) {
  // retraction on SE3 manifold

  float dt[3] = {0, 0, 0};
  float dq[4] = {0, 0, 0, 1};
  
  expSE3(xi, dt, dq);

  q1[0] = dq[3] * q[0] + dq[0] * q[3] + dq[1] * q[2] - dq[2] * q[1];
  q1[1] = dq[3] * q[1] + dq[1] * q[3] + dq[2] * q[0] - dq[0] * q[2];
  q1[2] = dq[3] * q[2] + dq[2] * q[3] + dq[0] * q[1] - dq[1] * q[0];
  q1[3] = dq[3] * q[3] - dq[0] * q[0] - dq[1] * q[1] - dq[2] * q[2];

  actSO3(dq, t, t1);
  t1[0] += dt[0];
  t1[1] += dt[1];
  t1[2] += dt[2];
}


__global__ void pose_retr_kernel(
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> dx,
    const int t0, const int t1) 
{

  for (int k=t0+threadIdx.x; k<t1; k+=blockDim.x) {
    float xi[6], q[4], q1[4], t[3], t1[3];

    t[0] = poses[k][0];
    t[1] = poses[k][1];
    t[2] = poses[k][2];

    q[0] = poses[k][3];
    q[1] = poses[k][4];
    q[2] = poses[k][5];
    q[3] = poses[k][6];
    
    for (int n=0; n<6; n++) {
      xi[n] = dx[k-t0][n];
    }

    retrSE3(xi, t, q, t1, q1);

    poses[k][0] = t1[0];
    poses[k][1] = t1[1];
    poses[k][2] = t1[2];

    poses[k][3] = q1[0];
    poses[k][4] = q1[1];
    poses[k][5] = q1[2];
    poses[k][6] = q1[3];
  }
}

__global__ void disp_retr_kernel(
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> dz,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> inds) 
{
  const int i = inds[blockIdx.x];
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  for (int k=threadIdx.x; k<ht*wd; k+=blockDim.x) {
    float d = disps[i][k/wd][k%wd] + dz[blockIdx.x][k];
    disps[i][k/wd][k%wd] = d;
  }
}

__global__ void uncertainty_retr_kernel(
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> uncertainties,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> du,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> inds)
{
  const int i = inds[blockIdx.x];
  const int ht = uncertainties.size(1);
  const int wd = uncertainties.size(2);

  for (int k=threadIdx.x; k<ht*wd; k+=blockDim.x) {
    float uncer = uncertainties[i][k/wd][k%wd] + du[blockIdx.x][k];
    uncer = fmaxf(uncer, 1e-1f);            // if uncer is too small, data term will not help gradient descent as the denom is larger than the nominator
    uncer = fminf(uncer, 2.0);            // if uncer is too large, it will cause instability, as the prior will not work
    uncertainties[i][k/wd][k%wd] = uncer;
  }
}

__global__ void uncertainty_update_kernel(
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> uncertainties,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> temp_y_cdot,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> dino_feats,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> affine_weights,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> inds)
{
  const int idx = inds[blockIdx.x];
  const int ht = uncertainties.size(1);
  const int wd = uncertainties.size(2);
  const int dim_feat = dino_feats.size(1);
  for (int k=threadIdx.x; k<ht*wd; k+=blockDim.x) {   // blockDim.x should be 256
    float tmp = 0.0;
    for (int c = 0; c < dim_feat; ++c) {
      tmp += affine_weights[c] * dino_feats[idx][c][k/wd][k%wd];
    }
    tmp += affine_weights[dim_feat];
    temp_y_cdot[idx][k/wd][k%wd] = tmp;
    float uncer = logf(1.1 + expf(tmp));  // softplus to ensure positivity + log(1.1) to constrain min(uncer) > 1e-1
    uncertainties[idx][k/wd][k%wd] = uncer;
  }
}

torch::Tensor accum_cuda(torch::Tensor data, torch::Tensor ix, torch::Tensor jx) {
  torch::Tensor ix_cpu = ix.to(torch::kCPU);
  torch::Tensor jx_cpu = jx.to(torch::kCPU);
  torch::Tensor inds = torch::argsort(ix_cpu);

  long* ix_data = ix_cpu.data_ptr<long>();
  long* jx_data = jx_cpu.data_ptr<long>();
  long* kx_data = inds.data_ptr<long>();

  int count = jx.size(0);
  std::vector<int> cols;

  torch::Tensor ptrs_cpu = torch::zeros({count+1}, 
    torch::TensorOptions().dtype(torch::kInt64));
  
  long* ptrs_data = ptrs_cpu.data_ptr<long>();
  ptrs_data[0] = 0;

  int i = 0;
  for (int j=0; j<count; j++) {
    while (i < ix.size(0) && ix_data[kx_data[i]] <= jx_data[j]) {
      if (ix_data[kx_data[i]] == jx_data[j])
        cols.push_back(kx_data[i]);
      i++;
    }
    ptrs_data[j+1] = cols.size();
  }

  torch::Tensor idxs_cpu = torch::zeros({long(cols.size())}, 
    torch::TensorOptions().dtype(torch::kInt64));

  long* idxs_data = idxs_cpu.data_ptr<long>();

  for (int i=0; i<cols.size(); i++) {
    idxs_data[i] = cols[i];
  }

  torch::Tensor ptrs = ptrs_cpu.to(torch::kCUDA);
  torch::Tensor idxs = idxs_cpu.to(torch::kCUDA);

  torch::Tensor out = torch::zeros({jx.size(0), data.size(1)},
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  accum_kernel<<<count, THREADS>>>(
    data.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    ptrs.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    idxs.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    out.packed_accessor32<float,2,torch::RestrictPtrTraits>());

  return out;
}


__global__ void EEt6x6_kernel(
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> E,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Q,
    const torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> idx,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> S)
{

  // indicices
  const int ix = idx[blockIdx.x][0];
  const int jx = idx[blockIdx.x][1];
  const int kx = idx[blockIdx.x][2];

  const int D = E.size(2);

  float dS[6][6];
  float ei[6];
  float ej[6];

  for (int i=0; i<6; i++) {
    for (int j=0; j<6; j++) {
      dS[i][j] = 0;
    }
  }

  for (int k=threadIdx.x; k<D; k+=blockDim.x) {
    const float q = Q[kx][k];
      
    // coalesced memory read
    for (int n=0; n<6; n++) {
      ei[n] = E[ix][n][k] * q;
      ej[n] = E[jx][n][k];
    }

    // block EEt
    for (int n=0; n<6; n++) {
      for (int m=0; m<6; m++) {
        dS[n][m] += ei[n] * ej[m];
      }
    }
  }

  __syncthreads();
  __shared__ float sdata[THREADS];

  for (int n=0; n<6; n++) {
    for (int m=0; m<6; m++) {
      sdata[threadIdx.x] = dS[n][m];

      blockReduce(sdata);

      if (threadIdx.x == 0) {
        S[blockIdx.x][n][m] = sdata[0];
      }
    }
  }
}


__global__ void Ev6x1_kernel(
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> E,
    const torch::PackedTensorAccessor32<float, 2,torch::RestrictPtrTraits> Q,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> w,
    const torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> idx,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> v)
{
  const int D = E.size(2);
  const int kx = idx[blockIdx.x][0];

  float b[6];
  for (int n=0; n<6; n++) {
    b[n] = 0.0;
  }

  for (int k=threadIdx.x; k<D; k+=blockDim.x) {
    const float q_w = Q[kx][k] * w[kx][k];

    for (int n=0; n<6; n++) {
      b[n] += q_w * E[blockIdx.x][n][k];
    }
  }

  __syncthreads();
  __shared__ float sdata[THREADS];

  for (int n=0; n<6; n++) {
    sdata[threadIdx.x] = b[n];
    blockReduce(sdata);

    if (threadIdx.x == 0) {
      v[blockIdx.x][n] += sdata[0];
    }
  }
}

__global__ void EvT6x1_kernel(
  const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> E,
  const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> x,
  const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> idx,
  torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> w)
{

  const int D = E.size(2);
  const int ix = idx[blockIdx.x];

  if (idx[blockIdx.x] <= 0 || idx[blockIdx.x] >= x.size(0))
    return;

  for (int k=threadIdx.x; k<D; k+=blockDim.x) {
    float dw = 0;
    for (int n=0; n<6; n++) {
      dw += E[blockIdx.x][n][k] * x[ix][n];
    }
    w[blockIdx.x][k] = dw;
  }
}

class SparseBlock {
  public:

    Eigen::SparseMatrix<double> A;
    Eigen::VectorX<double> b;

    SparseBlock(int N, int M) : N(N), M(M) {
      A = Eigen::SparseMatrix<double>(N*M, N*M);
      b = Eigen::VectorXd::Zero(N*M);
    }

    SparseBlock(Eigen::SparseMatrix<double> const& A, Eigen::VectorX<double> const& b, 
        int N, int M) : A(A), b(b), N(N), M(M) {}

    void update_lhs(torch::Tensor As, torch::Tensor ii, torch::Tensor jj) {

      auto As_cpu = As.to(torch::kCPU).to(torch::kFloat64);
      auto ii_cpu = ii.to(torch::kCPU).to(torch::kInt64);
      auto jj_cpu = jj.to(torch::kCPU).to(torch::kInt64);

      auto As_acc = As_cpu.accessor<double,3>();
      auto ii_acc = ii_cpu.accessor<long,1>();
      auto jj_acc = jj_cpu.accessor<long,1>();

      std::vector<T> tripletList;
      for (int n=0; n<ii.size(0); n++) {
        const int i = ii_acc[n];
        const int j = jj_acc[n];

        if (i >= 0 && j >= 0) {
          for (int k=0; k<M; k++) {
            for (int l=0; l<M; l++) {
              double val = As_acc[n][k][l];
              tripletList.push_back(T(M*i + k, M*j + l, val));
            }
          }
        }
      }
      A.setFromTriplets(tripletList.begin(), tripletList.end());
    }

    void update_rhs(torch::Tensor bs, torch::Tensor ii) {
      auto bs_cpu = bs.to(torch::kCPU).to(torch::kFloat64);
      auto ii_cpu = ii.to(torch::kCPU).to(torch::kInt64);

      auto bs_acc = bs_cpu.accessor<double,2>();
      auto ii_acc = ii_cpu.accessor<long,1>();

      for (int n=0; n<ii.size(0); n++) {
        const int i = ii_acc[n];
        if (i >= 0) {
          for (int j=0; j<M; j++) {
            b(i*M + j) += bs_acc[n][j];
          }
        }
      }
    }

    SparseBlock operator-(const SparseBlock& S) {
      return SparseBlock(A - S.A, b - S.b, N, M);
    }

    std::tuple<torch::Tensor, torch::Tensor> get_dense() {
      Eigen::MatrixXd Ad = Eigen::MatrixXd(A);

      torch::Tensor H = torch::from_blob(Ad.data(), {N*M, N*M}, torch::TensorOptions()
        .dtype(torch::kFloat64)).to(torch::kCUDA).to(torch::kFloat32);

      torch::Tensor v = torch::from_blob(b.data(), {N*M, 1}, torch::TensorOptions()
        .dtype(torch::kFloat64)).to(torch::kCUDA).to(torch::kFloat32);

      return std::make_tuple(H, v);

    }

    torch::Tensor solve(const float lm=0.0001, const float ep=0.1) {

      torch::Tensor dx;

      Eigen::SparseMatrix<double> L(A);
      L.diagonal().array() += ep + lm * L.diagonal().array();

      Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
      solver.compute(L);

      if (solver.info() == Eigen::Success) {
        Eigen::VectorXd x = solver.solve(b);
        dx = torch::from_blob(x.data(), {N, M}, torch::TensorOptions()
          .dtype(torch::kFloat64)).to(torch::kCUDA).to(torch::kFloat32);
      }
      else {
        dx = torch::zeros({N, M}, torch::TensorOptions()
          .device(torch::kCUDA).dtype(torch::kFloat32));
      }
      
      return dx;
    }

  private:
    const int N;
    const int M;

};


SparseBlock schur_block(torch::Tensor E,
                        torch::Tensor Q,
                        torch::Tensor w,
                        torch::Tensor ii,
                        torch::Tensor jj,
                        torch::Tensor kk,
                        const int t0,
                        const int t1)
{

  torch::Tensor ii_cpu = ii.to(torch::kCPU);
  torch::Tensor jj_cpu = jj.to(torch::kCPU);
  torch::Tensor kk_cpu = kk.to(torch::kCPU);

  const int P = t1 - t0;
  const long* ii_data = ii_cpu.data_ptr<long>();
  const long* jj_data = jj_cpu.data_ptr<long>();
  const long* kk_data = kk_cpu.data_ptr<long>();

  std::vector<std::vector<long>> graph(P);
  std::vector<std::vector<long>> index(P);

  for (int n=0; n<ii_cpu.size(0); n++) {
    const int j = jj_data[n];
    const int k = kk_data[n];

    if (j >= t0 && j <= t1) {
      const int t = j - t0;
      graph[t].push_back(k);
      index[t].push_back(n);
    }
  }

  std::vector<long> ii_list, jj_list, idx, jdx;

  for (int i=0; i<P; i++) {
    for (int j=0; j<P; j++) {
      for (int k=0; k < graph[i].size(); k++) {
        for (int l=0; l < graph[j].size(); l++) {
          if (graph[i][k] == graph[j][l]) {
            ii_list.push_back(i);
            jj_list.push_back(j);

            idx.push_back(index[i][k]);
            idx.push_back(index[j][l]);
            idx.push_back(graph[i][k]);
          }
        }
      }
    }
  }

  torch::Tensor ix_cuda = torch::from_blob(idx.data(), {long(idx.size())}, 
    torch::TensorOptions().dtype(torch::kInt64)).to(torch::kCUDA).view({-1, 3});

  torch::Tensor jx_cuda = torch::stack({kk_cpu}, -1)
    .to(torch::kCUDA).to(torch::kInt64);

  torch::Tensor ii2_cpu = torch::from_blob(ii_list.data(), {long(ii_list.size())}, 
    torch::TensorOptions().dtype(torch::kInt64)).view({-1});

  torch::Tensor jj2_cpu = torch::from_blob(jj_list.data(), {long(jj_list.size())}, 
    torch::TensorOptions().dtype(torch::kInt64)).view({-1});

  torch::Tensor S = torch::zeros({ix_cuda.size(0), 6, 6}, 
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  torch::Tensor v = torch::zeros({jx_cuda.size(0), 6},
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

  EEt6x6_kernel<<<ix_cuda.size(0), THREADS>>>(
    E.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    Q.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    ix_cuda.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
    S.packed_accessor32<float,3,torch::RestrictPtrTraits>());

  Ev6x1_kernel<<<jx_cuda.size(0), THREADS>>>(
    E.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    Q.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    w.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    jx_cuda.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
    v.packed_accessor32<float,2,torch::RestrictPtrTraits>());

  // schur block
  SparseBlock A(P, 6);
  A.update_lhs(S, ii2_cpu, jj2_cpu);
  A.update_rhs(v, jj_cpu - t0);

  return A;
}


__device__ inline float bilinear_at(
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits>& feat, // [N,C,H,W]
    int b, int c, float u, float v) {

  const int H = feat.size(2);
  const int W = feat.size(3);

  // set boundary
  if (u < 0.0f) u = 0.0f;
  if (v < 0.0f) v = 0.0f;
  if (u > (float)(W - 1)) u = (float)(W - 1);
  if (v > (float)(H - 1)) v = (float)(H - 1);

  const int x0 = (int)floorf(u);
  const int y0 = (int)floorf(v);
  const int x1 = min(x0 + 1, W - 1);
  const int y1 = min(y0 + 1, H - 1);

  const float du = u - (float)x0;
  const float dv = v - (float)y0;

  const float w00 = (1.0f - du) * (1.0f - dv);
  const float w10 = du * (1.0f - dv);
  const float w01 = (1.0f - du) * dv;
  const float w11 = du * dv;

  const float f00 = feat[b][c][y0][x0];
  const float f10 = feat[b][c][y0][x1];
  const float f01 = feat[b][c][y1][x0];
  const float f11 = feat[b][c][y1][x1];

  return w00 * f00 + w10 * f10 + w01 * f01 + w11 * f11, w00, w10, w01, w11;
}


__global__ void dino_feats_projective_transform_kernel(
  const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> target,
  const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> uncertainties,
  const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> dino_feats,
  const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
  const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
  const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsics,
  const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
  const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
  torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Jii_data,
  torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Jjj_data,
  torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> kf_count,
  torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> loss_data,
  const bool debug)
{
  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;
  
  const int ht = disps.size(1);
  const int wd = disps.size(2);
  const int C = dino_feats.size(1);

  int ix = static_cast<int>(ii[block_id]);
  int jx = static_cast<int>(jj[block_id]);

  // only when the distance between the two frames is larger than or equal to 2, we update the uncertainty
  if (abs(ix - jx) < 2) {
    return;
  }

  if (thread_id == 0) {
    atomicAdd(&kf_count[ix], 1.0f);
    atomicAdd(&kf_count[jx], 1.0f);
  }
  __syncthreads();

  __shared__ float fx;
  __shared__ float fy;
  __shared__ float cx;
  __shared__ float cy;

  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];

  // load intrinsics from global memory
  if (thread_id == 0) {
    fx = intrinsics[0];
    fy = intrinsics[1];
    cx = intrinsics[2];
    cy = intrinsics[3];
  }

  __syncthreads();

  // stereo frames
  if (ix == jx) {
    if (thread_id == 0) {
      tij[0] =  -0.1;
      tij[1] =     0;
      tij[2] =     0;
      qij[0] =     0;
      qij[1] =     0;
      qij[2] =     0;
      qij[3] =     1;
    }
  }

  else {

    // load poses from global memory
    if (thread_id < 3) {
      ti[thread_id] = poses[ix][thread_id];
      tj[thread_id] = poses[jx][thread_id];
    }

    if (thread_id < 4) {
      qi[thread_id] = poses[ix][thread_id+3];
      qj[thread_id] = poses[jx][thread_id+3];
    }

    __syncthreads();

    if (thread_id == 0) {
      relSE3(ti, qi, tj, qj, tij, qij);
    }
  }

  __syncthreads();

  //points 
  float Xi[4];
  float Xj[4];

  float vi[6], vj[6];

  for (int n=0; n<6; n++) {
    vi[n] = 0;
    vj[n] = 0;
  }

  __syncthreads();

  GPU_1D_KERNEL_LOOP(k, ht*wd) {

    int i, j;
    fast_divmod(k, wd, i, j);

    const float u = static_cast<float>(j);
    const float v = static_cast<float>(i);
    
    // homogenous coordinates
    Xi[0] = (u - cx) / fx;
    Xi[1] = (v - cy) / fy;
    Xi[2] = 1;
    Xi[3] = disps[ix][i][j];  // homogenous coords [x,y,1,h]

    // transform homogenous point
    actSE3(tij, qij, Xi, Xj);   // frame i to frame j  Xj = (R * Xi * 1/h) + t = 1/h * (R * Xi + h * t)
    // h * Xj = R * Xi + h * t => final we get h * Xj

    const float x = Xj[0];
    const float y = Xj[1];
    const float h = Xj[3];    // disparity of Xi

    const float d = (Xj[2] < MIN_DEPTH) ? 0.0 : 1.0 / Xj[2];

    const float u_i = fx * d * x + cx;
    const float v_i = fy * d * y + cy;

    const bool inside = (u_i >= 0.0f && u_i <= (float)(wd - 1) &&
                        v_i >= 0.0f && v_i <= (float)(ht - 1));
    const bool depth_ok = (Xj[2] >= MIN_DEPTH);

    if (!inside || !depth_ok) {
      continue;
    }

    float dot_sim = 0.0f;

    float fi_norm = 0.0f;
    float fj_norm = 0.0f;

    const int x0 = (int)floorf(u_i);
    const int y0 = (int)floorf(v_i);
    const int x1 = min(x0 + 1, int(wd) - 1);
    const int y1 = min(y0 + 1, int(ht) - 1);
  
    const float du = u_i - (float)x0;
    const float dv = v_i - (float)y0;
  
    const float w00 = (1.0f - du) * (1.0f - dv);
    const float w01 = du * (1.0f - dv);
    const float w10 = (1.0f - du) * dv;
    const float w11 = du * dv;

    for (int c = 0; c < C; ++c) {
      float f00 = dino_feats[jx][c][y0][x0];
      float f01 = dino_feats[jx][c][y0][x1];
      float f10 = dino_feats[jx][c][y1][x0];
      float f11 = dino_feats[jx][c][y1][x1];
      const float fj = w00 * f00 + w01 * f01 + w10 * f10 + w11 * f11;
      const float fi = dino_feats[ix][c][i][j];
      // compute cosine similarity
      dot_sim += fi * fj;
      fi_norm += fi * fi;
      fj_norm += fj * fj;
    }

    float uncer_i = uncertainties[ix][i][j];
    float uncer_j00 = uncertainties[jx][y0][x0];
    float uncer_j01 = uncertainties[jx][y0][x1];
    float uncer_j10 = uncertainties[jx][y1][x0];
    float uncer_j11 = uncertainties[jx][y1][x1];
    float uncer_reproj_j = w00 * uncer_j00 + w01 * uncer_j01 + w10 * uncer_j10 + w11 * uncer_j11;

    dot_sim = dot_sim / (sqrtf(fi_norm) * sqrtf(fj_norm) + 1e-6);
    dot_sim = fmaxf(fminf(dot_sim, 1.0f), 0.0f); // clamp to [0,1]
    // clamp to 0 if less than 0.5
    dot_sim = dot_sim > 0.5 ? dot_sim : 0.0;
    
    bool uncer_decoupling = true;
    if (uncer_decoupling) {
      // loss_data = (1 - dot_sim) / (uncer_i * uncer_reproj_j)
      float res = (1 - dot_sim);
      float denom = (uncer_i * uncer_reproj_j) * (uncer_i * uncer_reproj_j);
      float Ji = - res * uncer_reproj_j / denom;
      float Jj = - res * uncer_i / denom;
      float Jj_00 = Jj * w00;
      float Jj_01 = Jj * w01;
      float Jj_10 = Jj * w10;
      float Jj_11 = Jj * w11;

      Jii_data[block_id][k] = Ji;
      // use atomicAdd to add the value to the J_data
      const int k00 = y0 * wd + x0;
      const int k01 = y0 * wd + x1;
      const int k10 = y1 * wd + x0;
      const int k11 = y1 * wd + x1;  // TODO: check if this is correct

      atomicAdd(&Jjj_data[block_id][k00], Jj_00);
      atomicAdd(&Jjj_data[block_id][k01], Jj_01);
      atomicAdd(&Jjj_data[block_id][k10], Jj_10);
      atomicAdd(&Jjj_data[block_id][k11], Jj_11);
      // computer loss_data for debugging
      if (debug) {
        atomicAdd(&loss_data[block_id], res / (uncer_i * uncer_reproj_j));
      }
    }
    else {
      // loss_data_no_decoupling = (1 - dot_sim) / (uncer_i ** 2)
      float res = (1 - dot_sim);
      float Ji_no_decoupling = - 2 * res / (powf(uncer_i, 3));
      Jii_data[block_id][k] = Ji_no_decoupling;
    }
  }

  __syncthreads();
}

__device__ float dino_feats_similarity_kernel(
  const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> dino_feats,
  const int idx1,
  const int idx2,
  const int i,
  const int j
)
{
  const int C = dino_feats.size(1);

  float feat_similarity = 0.0;
  float f1_norm = 0.0;
  float f2_norm = 0.0;
  for (int c = 0; c < C; ++c) {
    float f1 = dino_feats[idx1][c][i][j];
    float f2 = dino_feats[idx2][c][i][j];
    feat_similarity += f1 * f2;
    f1_norm += f1 * f1;
    f2_norm += f2 * f2;
  }
  return feat_similarity / (sqrtf(f1_norm) * sqrtf(f2_norm) + 1e-6);
}

__global__ void prior_regularization_kernel(
  const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> uncertainties,
  torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> J_prior,
  const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> kx,
  torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> loss_prior,
  const bool debug)
{
  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;

  const int idx = kx[block_id];

  const int ht = uncertainties.size(1);
  const int wd = uncertainties.size(2);
  const int num_blocks = gridDim.x;

  const float similarity_threshold = 0.80;

  __syncthreads();

  GPU_1D_KERNEL_LOOP(k, ht*wd) {

    int i, j;
    fast_divmod(k, wd, i, j);
    
    // prior loss
    float dLp_duncer = 1.0 / (uncertainties[idx][i][j] + 1.0);
    J_prior[block_id][k] = dLp_duncer;

    if (debug) {
      float loss_reg = logf(uncertainties[idx][i][j] + 1.0);
      atomicAdd(&loss_prior[block_id], loss_reg);
    }
  }
  __syncthreads();
}

__global__ void linear_transform_kernel(
  const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> J_total,
  torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> J_linear,
  const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> dino_feats,
  const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> uncertainties,
  const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> temp_y_cdot,
  const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> kx)
{
  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;

  const int ht = uncertainties.size(1);
  const int wd = uncertainties.size(2);
  const int dim_feat = dino_feats.size(1);

  const int idx = kx[block_id];

  GPU_1D_KERNEL_LOOP(k, ht*wd) {
    int i, j;
    fast_divmod(k, wd, i, j);
    float uncer = uncertainties[idx][i][j];
    float y_cdot = temp_y_cdot[idx][i][j];
    float duncer_dy = 1.0 / (1.0 + 1.1 * expf(-y_cdot));
    // feature terms
    float dL2dy = J_total[block_id][k] * duncer_dy;
    for (int c = 0; c < dim_feat; ++c) {
      float dL2dw = dL2dy * dino_feats[idx][c][i][j];
      atomicAdd(&J_linear[block_id][c], dL2dw);
    }
    // bias term
    atomicAdd(&J_linear[block_id][dim_feat], dL2dy);
  }
  __syncthreads();
  GPU_1D_KERNEL_LOOP(i, dim_feat + 1) {
    J_linear[block_id][i] /= (ht * wd);
  }
}

std::vector<torch::Tensor> ba_cuda(
    torch::Tensor poses,        // [buffer, 7]
    torch::Tensor disps,        // [buffer, height, width]
    torch::Tensor intrinsics,
    torch::Tensor disps_sens,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor uncertainties,
    torch::Tensor temp_y_cdot,
    torch::Tensor dino_feats,
    torch::Tensor affine_weights,
    torch::Tensor eta,          // used for ragularizing of estimated disparity
    torch::Tensor ii,
    torch::Tensor jj,
    const int t0,
    const int t1,
    const int iterations,
    const float lm,             // Levenberg-Marquardt damping factor
    const float ep,             // 
    const float gamma_data,
    const float gamma_prior,
    const float gamma_depth,
    const float lr,
    const float weight_decay,
    const bool motion_only,
    const bool depth_only,
    const bool enable_update_uncer,
    const bool enable_udba,
    const bool enable_affine_transform,
    const bool enable_bidirectional_uncer,
    const bool debug)
{
  auto opts = poses.options();
  const int num = ii.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  torch::Tensor ts = torch::arange(t0, t1).to(torch::kCUDA);
  torch::Tensor ii_exp = torch::cat({ts, ii}, 0);
  torch::Tensor jj_exp = torch::cat({ts, jj}, 0);

  std::tuple<torch::Tensor, torch::Tensor> kuniq = 
    torch::_unique(ii_exp, true, true);

  torch::Tensor kx = std::get<0>(kuniq);
  torch::Tensor kk_exp = std::get<1>(kuniq);

  const int num_kf = kx.size(0);
  const int dim_feats = dino_feats.size(1);
    
  torch::Tensor dx;         // pose update
  torch::Tensor dz;         // disparity update

  // initialize buffers
  torch::Tensor Hs = torch::zeros({4, num, 6, 6}, opts);    // H_ii, H_ij, H_ji, H_jj (6 params per pose)
  torch::Tensor vs = torch::zeros({2, num, 6}, opts);       // v_i, v_j (6 params per pose)
  torch::Tensor Eii = torch::zeros({num, 6, ht*wd}, opts);  // Jacobian of residual w.r.t. pose i
  torch::Tensor Eij = torch::zeros({num, 6, ht*wd}, opts);  // Jacobian of residual w.r.t. pose j
  torch::Tensor Cii = torch::zeros({num, ht*wd}, opts);     // per-pixel squared error
  torch::Tensor wi = torch::zeros({num, ht*wd}, opts);      // per-pixel weights

  torch::Tensor Jii_data = torch::zeros({num, ht*wd}, opts);
  torch::Tensor Jjj_data = torch::zeros({num, ht*wd}, opts);
  torch::Tensor J_prior = torch::zeros({num_kf, ht*wd}, opts);
  // linear model for uncertainty: y = w^T * x + b, dimension of w is dim_feats
  torch::Tensor J_linear = torch::zeros({num_kf, dim_feats+1}, opts);

  torch::Tensor kf_count = torch::zeros({t1}, opts);

  torch::Tensor loss_data = torch::zeros({num}, opts);
  torch::Tensor loss_prior = torch::zeros({num_kf}, opts);
  torch::Tensor loss_total = torch::zeros({num_kf}, opts);

  for (int itr=0; itr<iterations; itr++) {

    projective_transform_kernel<<<num, THREADS>>>(
      targets.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      weights.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      uncertainties.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
      ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      Hs.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      vs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Eii.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Eij.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Cii.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      wi.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      enable_udba,
      enable_bidirectional_uncer);


    // pose x pose block
    SparseBlock A(t1 - t0, 6);

    A.update_lhs(Hs.reshape({-1, 6, 6}), 
        torch::cat({ii, ii, jj, jj}) - t0, 
        torch::cat({ii, jj, ii, jj}) - t0);

    A.update_rhs(vs.reshape({-1, 6}), 
        torch::cat({ii, jj}) - t0);

    if (motion_only) {
      dx = A.solve(lm, ep);

      // update poses
      pose_retr_kernel<<<1, THREADS>>>(
        poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(), t0, t1);
    }
    
    else {
      // add depth residual if there are depth sensor measurements
      const float alpha = gamma_depth;
      torch::Tensor m = (disps_sens.index({kx, "..."}) > 0).to(torch::TensorOptions().dtype(torch::kFloat32)).view({-1, ht*wd});
      torch::Tensor C = accum_cuda(Cii, ii, kx) + m * alpha + (1 - m) * eta.view({-1, ht*wd});
      torch::Tensor w = accum_cuda(wi, ii, kx) - m * alpha * (disps.index({kx, "..."}) - disps_sens.index({kx, "..."})).view({-1, ht*wd});
      torch::Tensor Q = 1.0 / C;

      torch::Tensor Ei = accum_cuda(Eii.view({num, 6*ht*wd}), ii, ts).view({t1-t0, 6, ht*wd});
      torch::Tensor E = torch::cat({Ei, Eij}, 0);

      SparseBlock S = schur_block(E, Q, w, ii_exp, jj_exp, kk_exp, t0, t1);   // simulataneouly construct the lhs and rhs of the linear system
      dx = (A - S).solve(lm, ep);     // compute delta_pose

      torch::Tensor ix = jj_exp - t0;
      torch::Tensor dw = torch::zeros({ix.size(0), ht*wd}, opts);

      EvT6x1_kernel<<<ix.size(0), THREADS>>>(
        E.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        ix.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
        dw.packed_accessor32<float,2,torch::RestrictPtrTraits>());

      dz = Q * (w - accum_cuda(dw, ii_exp, kx));      // compute delta_d

      if (!depth_only){
        // update poses
        pose_retr_kernel<<<1, THREADS>>>(
          poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
          dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(), t0, t1);
      }

      // update disparity maps
      disp_retr_kernel<<<kx.size(0), THREADS>>>(
        disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        dz.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        kx.packed_accessor32<long,1,torch::RestrictPtrTraits>());         // index of the keyframes
    }

    if (enable_update_uncer)
    {
      dino_feats_projective_transform_kernel<<<num, THREADS>>>(
        targets.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        uncertainties.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        dino_feats.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
        jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
        Jii_data.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        Jjj_data.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        kf_count.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        loss_data.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        debug);

      torch::Tensor coeff_data = 1.0f / (torch::index_select(kf_count, 0, kx) + 1e-6f) * gamma_data;

      prior_regularization_kernel<<<kx.size(0), THREADS>>>(
        uncertainties.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        J_prior.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        kx.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
        loss_prior.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        debug);

      torch::Tensor J_total = coeff_data.unsqueeze(1) * (accum_cuda(Jii_data, ii, kx) + accum_cuda(Jjj_data, jj, kx)) + gamma_prior * J_prior;
      if (enable_affine_transform) {
        linear_transform_kernel<<<kx.size(0), THREADS>>>(
          J_total.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
          J_linear.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
          dino_feats.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
          uncertainties.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
          temp_y_cdot.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
          kx.packed_accessor32<long,1,torch::RestrictPtrTraits>());
        // get the mean of J_linear for the dimension of 0
        torch::Tensor du = - lr * J_linear.mean(0);
        affine_weights += (du - weight_decay * affine_weights);  // L2 regularization

        uncertainty_update_kernel<<<kx.size(0), THREADS>>>(
          uncertainties.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
          temp_y_cdot.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
          dino_feats.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
          affine_weights.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
          kx.packed_accessor32<long,1,torch::RestrictPtrTraits>());
        
        if (debug) {
          for (int k=0; k<num; k++) {
            int ix = ii.index({k}).item<int>();
            int jx = jj.index({k}).item<int>();
            int idx_kf = ix - t0;
            loss_total[idx_kf] += loss_data[k] * coeff_data[idx_kf];
          }
          loss_total += loss_prior * gamma_prior;
          float loss = (loss_total).mean().item<float>() / (ht * wd);
          printf("total_loss: %f\n", loss);
          // printf mean of absolute value of du and variance of du
          printf("mean of absolute value of du: %f\n", du.abs().mean().item<float>());
        }
      }
      
      else {
        torch::Tensor du = - lr * J_total;
        // update mask
        uncertainty_retr_kernel<<<kx.size(0), THREADS>>>(
          uncertainties.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
          du.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
          kx.packed_accessor32<long,1,torch::RestrictPtrTraits>());
      }

      // clear Jii_data and Jjj_data due to the accumulation in the kernel
      Jjj_data.zero_();
      kf_count.zero_();
      J_linear.zero_();
      if (debug) {
        loss_data.zero_();
        loss_prior.zero_();
        loss_total.zero_();
      }
    }
  }

  return {dx, dz};
}



torch::Tensor frame_distance_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ii,
    torch::Tensor jj,
    const float beta)
{
  auto opts = poses.options();
  const int num = ii.size(0);

  torch::Tensor dist = torch::zeros({num}, opts);

  frame_distance_kernel<<<num, THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    dist.packed_accessor32<float,1,torch::RestrictPtrTraits>(), beta);

  return dist;
}


std::vector<torch::Tensor> projmap_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ii,
    torch::Tensor jj)
{
  auto opts = poses.options();
  const int num = ii.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  torch::Tensor coords = torch::zeros({num, ht, wd, 3}, opts);
  torch::Tensor valid = torch::zeros({num, ht, wd, 1}, opts);

  projmap_kernel<<<num, THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    coords.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    valid.packed_accessor32<float,4,torch::RestrictPtrTraits>());

  return {coords, valid};
}


torch::Tensor depth_filter_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ix,
    torch::Tensor thresh)
{
  const int num = ix.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  torch::Tensor counter = torch::zeros({num, ht, wd}, disps.options());

  dim3 blocks(num, 6, NUM_BLOCKS(ht * wd));

  depth_filter_kernel<<<blocks, THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    ix.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    thresh.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    counter.packed_accessor32<float,3,torch::RestrictPtrTraits>());

  return counter;
}


torch::Tensor iproj_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics)
{

  const int nm = disps.size(0);
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  auto opts = disps.options();
  torch::Tensor points = torch::zeros({nm, ht, wd, 3}, opts);

  dim3 blocks(nm, NUM_BLOCKS(ht * wd));

  iproj_kernel<<<blocks, THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    points.packed_accessor32<float,4,torch::RestrictPtrTraits>());

  return points;

}
