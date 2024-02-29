#ifndef CUDA_CHECK_H
#define CUDA_CHECK_H

#pragma once

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <curand_kernel.h>

#define CUDA_CHECK(call)                                                             \
    {                                                                                \
        cudaError_t err = call;                                                      \
        if (err != cudaSuccess)                                                      \
        {                                                                            \
            fprintf(stderr, "CUDA Error: %s:%d, ", __FILE__, __LINE__);              \
            fprintf(stderr, "code: %d, reason: %s\n", err, cudaGetErrorString(err)); \
            exit(1);                                                                 \
        }                                                                            \
    }

#define CUSOLVER_CHECK(call)                                                \
    {                                                                       \
        cusolverStatus_t err = call;                                        \
        if (err != CUSOLVER_STATUS_SUCCESS)                                 \
        {                                                                   \
            fprintf(stderr, "CUSOLVER Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d\n", err);                             \
            exit(1);                                                        \
        }                                                                   \
    }

#endif // CUDA_CHECK_H
