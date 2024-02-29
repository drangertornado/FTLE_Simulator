#ifndef RENDERER_H
#define RENDERER_H

#pragma once

#include <vector>
#include "CUDABuffer.h"
#include "Grid.h"
#include "Camera.h"
#include "Settings.h"

class Renderer
{
public:
    Renderer(Grid *grid, Settings *settings);
    ~Renderer();    

    void render();

    unsigned char *getPixels();
    unsigned char *getPixels_d_ptr();
    unsigned int getPixelsCount();
    unsigned int getPixelChannelsCount();

    curandState *getCurandStates_d_ptr();
    unsigned int getCurandStatesCount();

    Camera *getCamera();

private:
    void initializeCamera(unsigned int width, unsigned int height);

    void initializeBuffers(unsigned int width, unsigned int height);

    void drawGradient(unsigned int width, unsigned int height);

    cudaStream_t stream = nullptr;

    std::vector<unsigned char> pixels;
    CUDABuffer pixelsBuffer;
    unsigned int pixelsCount;
    unsigned int pixelChannelsCount;

    CUDABuffer curandStatesBuffer;
    unsigned int curandStatesCount;

    Camera camera;
    Grid *grid = nullptr;
    Settings *settings = nullptr;
};

#endif // RENDERER_H
