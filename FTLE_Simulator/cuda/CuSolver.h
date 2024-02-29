#ifndef CU_SOLVER_H
#define CU_SOLVER_H

#pragma once

#include <vector>
#include "CUDACheck.h"
#include "CUDABuffer.h"

struct CuSolverSvdParams
{
  cusolverDnHandle_t handle;
  cusolverEigMode_t jobz;
  gesvdjInfo_t params;
  int batchSize;
  int m;
  int n;
  float *A;
  int lda;
  float *S;
  int ldu;
  float *U;
  int ldv;
  float *V;
  int lwork;
  float *d_work = nullptr;
  std::vector<int> info;
  int *d_info = nullptr;

  CuSolverSvdParams()
      : handle(nullptr),
        jobz(CUSOLVER_EIG_MODE_NOVECTOR),
        params(nullptr),
        batchSize(0),
        m(0),
        n(0),
        A(nullptr),
        lda(0),
        S(nullptr),
        ldu(0),
        U(nullptr),
        ldv(0),
        V(nullptr),
        lwork(0),
        d_work(nullptr),
        info(0),
        d_info(nullptr)
  {
    // Initialize the handle and params
    CUSOLVER_CHECK(cusolverDnCreate(&handle));
    CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&params));
  }

  ~CuSolverSvdParams()
  {
    // Destroy the handle and params
    if (handle != nullptr)
    {
      CUSOLVER_CHECK(cusolverDnDestroy(handle));
    }
    if (params != nullptr)
    {
      CUSOLVER_CHECK(cusolverDnDestroyGesvdjInfo(params));
    }
    if (d_work != nullptr)
    {
      CUDA_CHECK(cudaFree(d_work));
    }
    if (d_info != nullptr)
    {
      CUDA_CHECK(cudaFree(d_info));
    }
  }
};

#endif // CU_SOLVER_H
