#ifndef RENDERER_CPP
#define RENDERER_CPP

#include "Renderer.h"
#include <iostream>
#include "VolumeRender.cuh"

// Constructor
Renderer::Renderer(Grid *grid, Settings *settings) : grid(grid),
                                                     settings(settings)
{
    std::cout << "\nInitializing renderer..." << std::endl;
    std::cout << "\nSetting up camera..." << std::endl;

    // Setup camera
    initializeCamera(settings->screenWidth, settings->screenHeight);
    // Initialize buffers on the device
    initializeBuffers(settings->screenWidth, settings->screenHeight);
}

// Destructor
Renderer::~Renderer()
{
    // Perform cleanup
    pixelsBuffer.free();
    curandStatesBuffer.free();

    // Destroy stream
    cudaStreamDestroy(stream);
}

// Function to setup the initial camera parameters
void Renderer::initializeCamera(unsigned int width, unsigned int height)
{
    float offset = 10.0f;
    glm::vec3 cameraPosition(0.0f, 0.0f, grid->getAABB()->max.z + offset);
    glm::vec3 cameraFront(0.0f, 0.0f, -1.0f);
    glm::vec3 cameraUp(0.0f, 1.0f, 0.0f);
    camera = Camera(cameraPosition, cameraFront, cameraUp, static_cast<float>(width), static_cast<float>(height));
}

// Function to initialize the buffers
void Renderer::initializeBuffers(unsigned int width, unsigned int height)
{
    // Create CUDA stream
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Allocate memory for pixels
    pixelsCount = width * height;
    pixelChannelsCount = 4;
    pixels = std::vector<unsigned char>(pixelsCount * pixelChannelsCount);
    pixelsBuffer.allocAndUpload(pixels);

    // Allocate memory to Curand states for generating random numbers
    curandStatesCount = width * height;
    curandStatesBuffer.alloc(curandStatesCount * sizeof(curandState));
    VolumeRenderKernel::initializeCurand(this, settings, stream);
}

// Function to render the pixels
void Renderer::render()
{
    // For testing the renderer
    // drawGradient(settings->screenWidth, settings->screenHeight);

    // Call CUDA kernel to render
    VolumeRenderKernel::renderPixels(this, grid, settings, stream);
    pixelsBuffer.download(pixels.data(), pixels.size());
}

// Function to create a gradient image
void Renderer::drawGradient(unsigned int width, unsigned int height)
{
    pixels.resize(width * height * 4);

    // Calculate step size for the gradient
    float stepR = 255.0f / (width - 1);
    float stepG = 255.0f / (height - 1);

    // Iterate over each pixel in the image and set RGBA values
    for (unsigned int y = 0; y < height; y++)
    {
        for (unsigned int x = 0; x < width; x++)
        {
            unsigned int index = (y * width + x) * 4;

            // Set RGBA values based on the gradient
            pixels[index] = static_cast<unsigned char>(x * stepR);
            pixels[index + 1] = static_cast<unsigned char>(y * stepG);
            pixels[index + 2] = 0;
            pixels[index + 3] = 255;
        }
    }
}

unsigned char *Renderer::getPixels()
{
    return pixels.data();
}

unsigned char *Renderer::getPixels_d_ptr()
{
    return static_cast<unsigned char *>(pixelsBuffer.d_ptr);
}

unsigned int Renderer::getPixelsCount()
{
    return pixelsCount;
}

unsigned int Renderer::getPixelChannelsCount()
{
    return pixelChannelsCount;
}

curandState *Renderer::getCurandStates_d_ptr()
{
    return static_cast<curandState *>(curandStatesBuffer.d_ptr);
}

unsigned int Renderer::getCurandStatesCount()
{
    return curandStatesCount;
}

Camera *Renderer::getCamera()
{
    return &camera;
}

#endif // RENDERER_CPP
