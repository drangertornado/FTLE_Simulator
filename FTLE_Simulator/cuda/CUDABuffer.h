#ifndef CUDA_BUFFER_H
#define CUDA_BUFFER_H

#pragma once

#include "CUDACheck.h"
#include <assert.h>
#include <vector>

// Adapted from: https://github.com/ingowald/optix7course

struct CUDABuffer
{
    // Device pointer
    void *d_ptr = nullptr;
    // Size in bytes
    size_t sizeInBytes = 0;

    // Helper functions for optix
    /*inline CUdeviceptr d_pointer() const
    {
        return (CUdeviceptr)d_ptr;
    }*/

    // Re-size buffer to given number of bytes
    void resize(size_t size)
    {
        if (d_ptr)
            free();
        alloc(size);
    }

    // Allocate to given number of bytes
    void alloc(size_t size)
    {
        assert(d_ptr == nullptr);
        this->sizeInBytes = size;
        CUDA_CHECK(cudaMalloc((void **)&d_ptr, sizeInBytes));
    }

    // Free allocated memory (d_ptr)
    void free()
    {
        CUDA_CHECK(cudaFree(d_ptr));
        d_ptr = nullptr;
        sizeInBytes = 0;
    }

    // Allocate and transfer data from host to device memory (d_ptr)
    template <typename T>
    void allocAndUpload(const std::vector<T> &vector)
    {
        alloc(vector.size() * sizeof(T));
        upload((const T *)vector.data(), vector.size());
    }

    // Transfer data from host memory (t) to device memory (d_ptr)
    template <typename T>
    void upload(const T *t, size_t count)
    {
        assert(d_ptr != nullptr);
        assert(sizeInBytes == count * sizeof(T));
        CUDA_CHECK(cudaMemcpy(d_ptr, (void *)t,
                              count * sizeof(T),
                              cudaMemcpyHostToDevice));
    }

    // Transfer data from device memory (d_ptr) to host memory (t)
    template <typename T>
    void download(T *t, size_t count)
    {
        assert(d_ptr != nullptr);
        assert(sizeInBytes == count * sizeof(T));
        CUDA_CHECK(cudaMemcpy((void *)t, d_ptr,
                              count * sizeof(T),
                              cudaMemcpyDeviceToHost));
    }
};

#endif // CUDA_BUFFER_H
