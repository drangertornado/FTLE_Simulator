#ifndef VOLUME_RENDER_CU
#define VOLUME_RENDER_CU

#include "VolumeRender.cuh"

// Function to initialize the curand states
__global__ void setupCurandStates(curandState *curandStates, unsigned int curandStatesCount)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Not sure if there is a better way seed
    unsigned long long seed = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < curandStatesCount)
    {
        curand_init(seed, threadIdx.x, 0, &curandStates[idx]);
    }
}

// Function to render the volume
__global__ void volumeRender(unsigned char *pixels, unsigned int pixelsCount, unsigned int channelsPerPixel, Point *points, unsigned int gridResolution, float gridSpacing, AABB gridAABB, Camera camera, float rayStepSize, TransferFunction tfForward, TransferFunction tfBackward)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pixelsCount)
    {
        // Compute normalized device coordinates of the pixel with origin at the center
        glm::vec2 normalizedDeviceCoords = normalizePixels(idx, camera.viewportSize);

        // Generate ray
        Ray ray(normalizedDeviceCoords, camera);

        // Default pixel color to black
        glm::vec4 pixelColor(0.0f);

        // Compute ray intersections on the volume's axis aligned bounding box (AABB)
        float tmin, tmax;
        if (gridAABB.intersectRay(ray, tmin, tmax))
        {
            // Ray march through the bounding volume based on step size
            for (float t = tmin; t <= tmax; t += rayStepSize)
            {
                glm::vec3 samplePoint = ray.origin + t * ray.direction;

                glm::vec2 interpolatedScalar = interpolateScalar(samplePoint, points, gridResolution, gridSpacing);

                // Use transfer function to add color to pixelColor variable
                glm::vec4 interpolatedColorForward = tfForward.getColor(interpolatedScalar.x);
                glm::vec4 interpolatedColorBackward = tfBackward.getColor(interpolatedScalar.y);

                // Use additive blending
                glm::vec4 interpolatedColorBlended = interpolatedColorForward + interpolatedColorBackward;
                // glm::vec4 interpolatedColorBlended = (interpolatedColorForward + interpolatedColorBackward) * 0.5f;

                // Use over blending for forward composting
                pixelColor = overBlend(pixelColor, interpolatedColorBlended);

                // Early ray termination if opacity is 1.0f
                if (pixelColor.a >= 0.999999f)
                {
                    break;
                }
            }
        }

        // Save pixel color
        pixels[idx * channelsPerPixel] = static_cast<unsigned char>(glm::clamp(pixelColor.r * 255.0f, 0.0f, 255.0f));
        pixels[idx * channelsPerPixel + 1] = static_cast<unsigned char>(glm::clamp(pixelColor.g * 255.0f, 0.0f, 255.0f));
        pixels[idx * channelsPerPixel + 2] = static_cast<unsigned char>(glm::clamp(pixelColor.b * 255.0f, 0.0f, 255.0f));
        pixels[idx * channelsPerPixel + 3] = static_cast<unsigned char>(glm::clamp(pixelColor.a * 255.0f, 0.0f, 255.0f));
    }
}

// Function to render the volume with anti-aliasing
__global__ void volumeRenderAA(unsigned char *pixels, unsigned int pixelsCount, unsigned int channelsPerPixel, Point *points, unsigned int gridResolution, float gridSpacing, AABB gridAABB, Camera camera, unsigned int raysCount, float rayStepSize, float antiAliasingIntensity, TransferFunction tfForward, TransferFunction tfBackward, curandState *curandStates)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pixelsCount)
    {
        // Compute normalized device coordinates of the pixel with origin at the center
        glm::vec2 normalizedDeviceCoords = normalizePixels(idx, camera.viewportSize);

        // Default pixel color to black
        glm::vec4 pixelColor(0.0f);

        // Generate rays
        for (unsigned int rayIdx = 0; rayIdx < raysCount; rayIdx++)
        {
            // Generate ray
            Ray ray(normalizedDeviceCoords, camera, antiAliasingIntensity, curandStates[idx]);

            // Compute ray intersections on the volume's axis aligned bounding box (AABB)
            float tmin, tmax;
            if (gridAABB.intersectRay(ray, tmin, tmax))
            {
                glm::vec4 pixelColorPerRay(0.0f);

                // Ray march through the bounding volume based on step size
                for (float t = tmin; t <= tmax; t += rayStepSize)
                {
                    glm::vec3 samplePoint = ray.origin + t * ray.direction;

                    glm::vec2 interpolatedScalar = interpolateScalar(samplePoint, points, gridResolution, gridSpacing);

                    // Use transfer function to add color to pixelColor variable
                    glm::vec4 interpolatedColorForward = tfForward.getColor(interpolatedScalar.x);
                    glm::vec4 interpolatedColorBackward = tfBackward.getColor(interpolatedScalar.y);

                    // Use additive blending
                    glm::vec4 interpolatedColorBlended = interpolatedColorForward + interpolatedColorBackward;
                    // glm::vec4 interpolatedColorBlended = (interpolatedColorForward + interpolatedColorBackward) * 0.5f;

                    // Use over blending for forward composting
                    pixelColorPerRay = overBlend(pixelColorPerRay, interpolatedColorBlended);

                    // Early ray termination if opacity is 1.0f
                    if (pixelColorPerRay.a >= 0.999999f)
                    {
                        break;
                    }
                }

                if (rayIdx > 0)
                {
                    pixelColor = (pixelColor + pixelColorPerRay) * 0.5f;
                }
                else
                {
                    pixelColor = pixelColorPerRay;
                }
            }
        }

        // Save pixel color
        pixels[idx * channelsPerPixel] = static_cast<unsigned char>(glm::clamp(pixelColor.r * 255.0f, 0.0f, 255.0f));
        pixels[idx * channelsPerPixel + 1] = static_cast<unsigned char>(glm::clamp(pixelColor.g * 255.0f, 0.0f, 255.0f));
        pixels[idx * channelsPerPixel + 2] = static_cast<unsigned char>(glm::clamp(pixelColor.b * 255.0f, 0.0f, 255.0f));
        pixels[idx * channelsPerPixel + 3] = static_cast<unsigned char>(glm::clamp(pixelColor.a * 255.0f, 0.0f, 255.0f));
    }
}

// Function to normalize the pixels to device coordinates
__device__ glm::vec2 normalizePixels(const unsigned int &pixelIndex, const glm::vec2 &viewportSize)
{
    float normalizedDeviceCoordsX = (pixelIndex % static_cast<unsigned int>(viewportSize.x)) / static_cast<float>(viewportSize.x) - 0.5f;
    float normalizedDeviceCoordsY = (pixelIndex / viewportSize.x) / static_cast<float>(viewportSize.y) - 0.5f;

    return glm::vec2(normalizedDeviceCoordsX, normalizedDeviceCoordsY);
}

// Function to interpolate the scalar values
__device__ glm::vec2 interpolateScalar(const glm::vec3 &samplePoint, const Point *points, const unsigned int &resolution, const float &spacing)
{
    // Apply inverse transformation
    float halfSize = 0.5f * (resolution - 1) * spacing;
    float gridSpacingInverse = 1 / spacing;
    glm::vec3 transformedSamplePoint = (samplePoint + halfSize) * gridSpacingInverse;

    // Coordinates of left (x), bottom (y) and front (z) neighbour point
    unsigned int xmin = static_cast<unsigned int>(transformedSamplePoint.x);
    unsigned int ymin = static_cast<unsigned int>(transformedSamplePoint.y);
    unsigned int zmin = static_cast<unsigned int>(transformedSamplePoint.z);

    // Linearized index of 8 neighbouring points within the cell
    unsigned int frontBottomLeftIdx = linearizeIndex(xmin, ymin, zmin, resolution);
    unsigned int frontBottomRightIdx = linearizeIndex(xmin + 1, ymin, zmin, resolution);
    unsigned int frontTopLeftIdx = linearizeIndex(xmin, ymin + 1, zmin, resolution);
    unsigned int frontTopRightIdx = linearizeIndex(xmin + 1, ymin + 1, zmin, resolution);
    unsigned int backBottomLeftIdx = linearizeIndex(xmin, ymin, zmin + 1, resolution);
    unsigned int backBottomRightIdx = linearizeIndex(xmin + 1, ymin, zmin + 1, resolution);
    unsigned int backTopLeftIdx = linearizeIndex(xmin, ymin + 1, zmin + 1, resolution);
    unsigned int backTopRightIdx = linearizeIndex(xmin + 1, ymin + 1, zmin + 1, resolution);

    // Calculate fractional distances along each axis
    glm::vec3 fractionalDistances = transformedSamplePoint - glm::vec3(xmin, ymin, zmin);

    // Perform linear interpolation along the x-axis
    glm::vec2 interpolatedValueXFrontBottom = glm::mix(
        glm::vec2(points[frontBottomLeftIdx].ftleExponentForward,
                  points[frontBottomLeftIdx].ftleExponentBackward),
        glm::vec2(points[frontBottomRightIdx].ftleExponentForward,
                  points[frontBottomRightIdx].ftleExponentBackward),
        fractionalDistances.x);

    glm::vec2 interpolatedValueXFrontTop = glm::mix(
        glm::vec2(points[frontTopLeftIdx].ftleExponentForward,
                  points[frontTopLeftIdx].ftleExponentBackward),
        glm::vec2(points[frontTopRightIdx].ftleExponentForward,
                  points[frontTopRightIdx].ftleExponentBackward),
        fractionalDistances.x);

    glm::vec2 interpolatedValueXBackBottom = glm::mix(
        glm::vec2(points[backBottomLeftIdx].ftleExponentForward,
                  points[backBottomLeftIdx].ftleExponentBackward),
        glm::vec2(points[backBottomRightIdx].ftleExponentForward,
                  points[backBottomRightIdx].ftleExponentBackward),
        fractionalDistances.x);

    glm::vec2 interpolatedValueXBackTop = glm::mix(
        glm::vec2(points[backTopLeftIdx].ftleExponentForward,
                  points[backTopLeftIdx].ftleExponentBackward),
        glm::vec2(points[backTopRightIdx].ftleExponentForward,
                  points[backTopRightIdx].ftleExponentBackward),
        fractionalDistances.x);

    // Perform linear interpolation along the y-axis
    glm::vec2 interpolatedValueYFront = glm::mix(
        interpolatedValueXFrontBottom,
        interpolatedValueXFrontTop,
        fractionalDistances.y);

    glm::vec2 interpolatedValueYBack = glm::mix(
        interpolatedValueXBackBottom,
        interpolatedValueXBackTop,
        fractionalDistances.y);

    // Perform linear interpolation along the z-axis
    glm::vec2 interpolatedScalarValue = glm::mix(
        interpolatedValueYFront,
        interpolatedValueYBack,
        fractionalDistances.z);

    return interpolatedScalarValue;
}

// Function to linearize the coordinates
__device__ unsigned int linearizeIndex(const int &x, const int &y, const int &z, const unsigned int &resolution)
{
    return x + y * resolution + z * resolution * resolution;
}

// Function to blend foreground over background
__device__ glm::vec4 overBlend(const glm::vec4 &foreground, const glm::vec4 &background)
{
    glm::vec4 blendedColor(0.0f);

    if (background.a >= 0.001f)
    {
        // Blend colors
        blendedColor.r = foreground.a * foreground.r + (1.0f - foreground.a) * background.r;
        blendedColor.g = foreground.a * foreground.g + (1.0f - foreground.a) * background.g;
        blendedColor.b = foreground.a * foreground.b + (1.0f - foreground.a) * background.b;
        blendedColor.a = foreground.a + (1.0f - foreground.a) * background.a;
    }
    else
    {
        return foreground;
    }

    return blendedColor;
}

// External C++ interface
namespace VolumeRenderKernel
{
    __host__ void initializeCurand(Renderer *renderer, Settings *settings, cudaStream_t &stream)
    {
        unsigned int numBlocks = (renderer->getCurandStatesCount() + settings->cudaBlockSize - 1) / settings->cudaBlockSize;

        // Initialize curand states
        setupCurandStates<<<numBlocks, settings->cudaBlockSize>>>(
            renderer->getCurandStates_d_ptr(),
            renderer->getCurandStatesCount());

        cudaStreamSynchronize(stream);
    }

    __host__ void renderPixels(Renderer *renderer, Grid *grid, Settings *settings, cudaStream_t &stream)
    {
        unsigned int numBlocks = (renderer->getPixelsCount() + settings->cudaBlockSize - 1) / settings->cudaBlockSize;

        // Render the image
        if (settings->antiAliasingEnabled)
        {
            volumeRenderAA<<<numBlocks, settings->cudaBlockSize, 0, stream>>>(
                renderer->getPixels_d_ptr(),
                renderer->getPixelsCount(),
                renderer->getPixelChannelsCount(),
                grid->getPoints_d_ptr(),
                grid->getResolution(),
                grid->getSpacing(),
                *grid->getAABB(),
                *renderer->getCamera(),
                settings->raysCount,
                settings->rayStepSize,
                settings->antiAliasingIntensity,
                settings->tfForward,
                settings->tfBackward,
                renderer->getCurandStates_d_ptr());
        }
        else
        {
            volumeRender<<<numBlocks, settings->cudaBlockSize, 0, stream>>>(
                renderer->getPixels_d_ptr(),
                renderer->getPixelsCount(),
                renderer->getPixelChannelsCount(),
                grid->getPoints_d_ptr(),
                grid->getResolution(),
                grid->getSpacing(),
                *grid->getAABB(),
                *renderer->getCamera(),
                settings->rayStepSize,
                settings->tfForward,
                settings->tfBackward);
        }

        // Wait for pixels to be rendered
        cudaStreamSynchronize(stream);
    }
}

#endif // VOLUME_RENDER_CU
