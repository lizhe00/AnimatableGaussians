// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#include <float.h>
#include <math.h>
#include <cstdio>

// helper WarpReduce used in .cu files

template <typename scalar_t>
__device__ void WarpReduce(
    volatile scalar_t* min_dists,
    volatile int64_t* min_idxs,
    const size_t tid) {
  // s = 32
  if (min_dists[tid] > min_dists[tid + 32]) {
    min_idxs[tid] = min_idxs[tid + 32];
    min_dists[tid] = min_dists[tid + 32];
  }
  // s = 16
  if (min_dists[tid] > min_dists[tid + 16]) {
    min_idxs[tid] = min_idxs[tid + 16];
    min_dists[tid] = min_dists[tid + 16];
  }
  // s = 8
  if (min_dists[tid] > min_dists[tid + 8]) {
    min_idxs[tid] = min_idxs[tid + 8];
    min_dists[tid] = min_dists[tid + 8];
  }
  // s = 4
  if (min_dists[tid] > min_dists[tid + 4]) {
    min_idxs[tid] = min_idxs[tid + 4];
    min_dists[tid] = min_dists[tid + 4];
  }
  // s = 2
  if (min_dists[tid] > min_dists[tid + 2]) {
    min_idxs[tid] = min_idxs[tid + 2];
    min_dists[tid] = min_dists[tid + 2];
  }
  // s = 1
  if (min_dists[tid] > min_dists[tid + 1]) {
    min_idxs[tid] = min_idxs[tid + 1];
    min_dists[tid] = min_dists[tid + 1];
  }
}


template <typename scalar_t>
__device__ void WarpReduce2(
    volatile scalar_t* min_dists,
    volatile int64_t* min_idxs,
    volatile scalar_t* mw0,
    volatile scalar_t* mw1,
    volatile scalar_t* mw2,
    const size_t tid) {
  // s = 32
  if (min_dists[tid] > min_dists[tid + 32]) {
    min_idxs[tid] = min_idxs[tid + 32];
    min_dists[tid] = min_dists[tid + 32];
    mw0[tid] = mw0[tid + 32];
    mw1[tid] = mw1[tid + 32];
    mw2[tid] = mw2[tid + 32];
  }
  // s = 16
  if (min_dists[tid] > min_dists[tid + 16]) {
    min_idxs[tid] = min_idxs[tid + 16];
    min_dists[tid] = min_dists[tid + 16];
    mw0[tid] = mw0[tid + 16];
    mw1[tid] = mw1[tid + 16];
    mw2[tid] = mw2[tid + 16];
  }
  // s = 8
  if (min_dists[tid] > min_dists[tid + 8]) {
    min_idxs[tid] = min_idxs[tid + 8];
    min_dists[tid] = min_dists[tid + 8];
    mw0[tid] = mw0[tid + 8];
    mw1[tid] = mw1[tid + 8];
    mw2[tid] = mw2[tid + 8];
  }
  // s = 4
  if (min_dists[tid] > min_dists[tid + 4]) {
    min_idxs[tid] = min_idxs[tid + 4];
    min_dists[tid] = min_dists[tid + 4];
    mw0[tid] = mw0[tid + 4];
    mw1[tid] = mw1[tid + 4];
    mw2[tid] = mw2[tid + 4];
  }
  // s = 2
  if (min_dists[tid] > min_dists[tid + 2]) {
    min_idxs[tid] = min_idxs[tid + 2];
    min_dists[tid] = min_dists[tid + 2];
    mw0[tid] = mw0[tid + 2];
    mw1[tid] = mw1[tid + 2];
    mw2[tid] = mw2[tid + 2];
  }
  // s = 1
  if (min_dists[tid] > min_dists[tid + 1]) {
    min_idxs[tid] = min_idxs[tid + 1];
    min_dists[tid] = min_dists[tid + 1];
    mw0[tid] = mw0[tid + 1];
    mw1[tid] = mw1[tid + 1];
    mw2[tid] = mw2[tid + 1];
  }
}
