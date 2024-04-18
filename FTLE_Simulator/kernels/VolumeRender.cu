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
__global__ void volumeRender(unsigned char *pixels, unsigned int pixelsCount, unsigned int channelsPerPixel, Camera camera, Point *points, unsigned int gridResolution, float gridSpacing, AABB gridAABB, float rayStepSize, unsigned int flowDirectionPreference, TransferFunction tfForward, TransferFunction tfBackward, bool lightingEnabled, PointLight light, PhongLighting lighting)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pixelsCount)
    {
        // Compute normalized device coordinates of the pixel with origin at the center
        glm::vec2 normalizedDeviceCoords = normalizePixels(idx, camera.viewportSize);

        // Generate ray
        Ray ray(normalizedDeviceCoords, camera);

        // Integrate colors by ray marching
        glm::vec4 pixelColor = rayMarch(ray, rayStepSize, points, gridResolution, gridSpacing, gridAABB, flowDirectionPreference, tfForward, tfBackward, lightingEnabled, light, lighting);

        // Save pixel color
        pixels[idx * channelsPerPixel] = static_cast<unsigned char>(glm::clamp(pixelColor.r * 255.0f, 0.0f, 255.0f));
        pixels[idx * channelsPerPixel + 1] = static_cast<unsigned char>(glm::clamp(pixelColor.g * 255.0f, 0.0f, 255.0f));
        pixels[idx * channelsPerPixel + 2] = static_cast<unsigned char>(glm::clamp(pixelColor.b * 255.0f, 0.0f, 255.0f));
    }
}

// Function to render the volume with anti-aliasing
__global__ void volumeRenderAA(unsigned char *pixels, unsigned int pixelsCount, unsigned int channelsPerPixel, Camera camera, Point *points, unsigned int gridResolution, float gridSpacing, AABB gridAABB, unsigned int raysCount, float rayStepSize, unsigned int flowDirectionPreference, TransferFunction tfForward, TransferFunction tfBackward, bool lightingEnabled, PointLight light, PhongLighting lighting, curandState *curandStates, float antiAliasingIntensity)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pixelsCount)
    {
        // Compute normalized device coordinates of the pixel with origin at the center
        glm::vec2 normalizedDeviceCoords = normalizePixels(idx, camera.viewportSize);

        // Generate ray
        Ray ray(normalizedDeviceCoords, camera, antiAliasingIntensity, curandStates[idx]);

        // Integrate colors by ray marching
        glm::vec4 pixelColor = rayMarchAA(ray, rayStepSize, points, gridResolution, gridSpacing, gridAABB, flowDirectionPreference, tfForward, tfBackward, lightingEnabled, light, lighting, curandStates[idx], antiAliasingIntensity);

        // Generate rays
        for (unsigned int rayIdx = 0; rayIdx < raysCount - 1; rayIdx++)
        {
            // Generate ray
            Ray ray(normalizedDeviceCoords, camera, antiAliasingIntensity, curandStates[idx]);

            // Integrate colors by ray marching
            glm::vec4 pixelColorPerRay = rayMarchAA(ray, rayStepSize, points, gridResolution, gridSpacing, gridAABB, flowDirectionPreference, tfForward, tfBackward, lightingEnabled, light, lighting, curandStates[idx], antiAliasingIntensity);

            // Mix colors
            pixelColor = (pixelColor + pixelColorPerRay) * 0.5f;
        }

        // Save pixel color
        pixels[idx * channelsPerPixel] = static_cast<unsigned char>(glm::clamp(pixelColor.r * 255.0f, 0.0f, 255.0f));
        pixels[idx * channelsPerPixel + 1] = static_cast<unsigned char>(glm::clamp(pixelColor.g * 255.0f, 0.0f, 255.0f));
        pixels[idx * channelsPerPixel + 2] = static_cast<unsigned char>(glm::clamp(pixelColor.b * 255.0f, 0.0f, 255.0f));
    }
}

__device__ glm::vec4 rayMarch(const Ray &ray, const float &rayStepSize, const Point *points, const unsigned int &gridResolution, const float &gridSpacing, AABB &gridAABB, const unsigned int &flowDirectionPreference, TransferFunction &tfForward, TransferFunction &tfBackward, bool &lightingEnabled, PointLight &light, PhongLighting &lighting)
{
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

            // Early ray termination if opacity is 1.0f
            if (pixelColor.a >= 0.9999f)
            {
                break;
            }

            // Interpolate scalar values (x - forward, y - backward)
            glm::vec2 interpolatedScalars;
            // Interpolated normals
            glm::vec3 interpolatedNormalsForward, interpolatedNormalsBackward;
            // Perform trilinear interpolation on the sample point
            interpolateParameters(samplePoint, points, gridResolution, gridSpacing, interpolatedScalars, interpolatedNormalsForward, interpolatedNormalsBackward);

            // Use transfer function to set color
            glm::vec4 interpolatedColorForward = tfForward.getColor(interpolatedScalars.x);
            glm::vec4 interpolatedColorBackward = tfBackward.getColor(interpolatedScalars.y);

            // Apply lighting to the colors
            glm::vec3 colorForward, colorBackward;
            if (lightingEnabled)
            {
                colorForward = lighting.computeLighting(samplePoint, interpolatedColorForward, interpolatedNormalsForward, light, ray);
                colorBackward = lighting.computeLighting(samplePoint, interpolatedColorBackward, interpolatedNormalsBackward, light, ray);
            }
            else
            {
                colorForward = interpolatedColorForward;
                colorBackward = interpolatedColorBackward;
            }

            // Blend colors
            glm::vec4 interpolatedColorBlended;
            if (flowDirectionPreference == 0)
            {
                // Blend forward and backward color via additive blending
                interpolatedColorBlended = glm::vec4(colorForward + colorBackward, (interpolatedColorForward.a + interpolatedColorBackward.a) * 0.5f);
            }
            else if (flowDirectionPreference == 1)
            {
                // Set forward color
                interpolatedColorBlended = glm::vec4(colorForward, interpolatedColorForward.a);
            }
            else if (flowDirectionPreference == 2)
            {
                // Set backward color
                interpolatedColorBlended = glm::vec4(colorBackward, interpolatedColorBackward.a);
            }

            // Use over blending for forward composting
            pixelColor = overBlend(pixelColor, interpolatedColorBlended);
        }

        // Blend with black color if alpha is not 1.0f
        if (pixelColor.a < 0.9999f)
        {
            pixelColor = overBlend(pixelColor, glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        }
    }

    return pixelColor;
}

__device__ glm::vec4 rayMarchAA(const Ray &ray, const float &rayStepSize, const Point *points, const unsigned int &gridResolution, const float &gridSpacing, AABB &gridAABB, const unsigned int &flowDirectionPreference, TransferFunction &tfForward, TransferFunction &tfBackward, bool &lightingEnabled, PointLight &light, PhongLighting &lighting, curandState &curandState, const float &antiAliasingIntensity)
{
    // Default pixel color to black
    glm::vec4 pixelColor(0.0f);

    // Compute ray intersections on the volume's axis aligned bounding box (AABB)
    float tmin, tmax;
    if (gridAABB.intersectRay(ray, tmin, tmax))
    {
        // Ray march through the bounding volume based on step size
        for (float t = tmin; t <= tmax; t += rayStepSize + (0.5f - static_cast<float>(curand_uniform(&curandState))) * antiAliasingIntensity)
        {
            glm::vec3 samplePoint = ray.origin + t * ray.direction;

            // Early ray termination if opacity is 1.0f
            if (pixelColor.a >= 0.9999f)
            {
                break;
            }

            // Interpolate scalar values (x - forward, y - backward)
            glm::vec2 interpolatedScalars;
            // Interpolated normals
            glm::vec3 interpolatedNormalsForward, interpolatedNormalsBackward;
            // Perform trilinear interpolation on the sample point
            interpolateParameters(samplePoint, points, gridResolution, gridSpacing, interpolatedScalars, interpolatedNormalsForward, interpolatedNormalsBackward);

            // Use transfer function to set color
            glm::vec4 interpolatedColorForward = tfForward.getColor(interpolatedScalars.x);
            glm::vec4 interpolatedColorBackward = tfBackward.getColor(interpolatedScalars.y);

            // Apply lighting to the colors
            glm::vec3 colorForward, colorBackward;
            if (lightingEnabled)
            {
                colorForward = lighting.computeLighting(samplePoint, interpolatedColorForward, interpolatedNormalsForward, light, ray);
                colorBackward = lighting.computeLighting(samplePoint, interpolatedColorBackward, interpolatedNormalsBackward, light, ray);
            }
            else
            {
                colorForward = interpolatedColorForward;
                colorBackward = interpolatedColorBackward;
            }

            // Blend colors
            glm::vec4 interpolatedColorBlended;
            if (flowDirectionPreference == 0)
            {
                // Blend forward and backward color via additive blending
                interpolatedColorBlended = glm::vec4(colorForward + colorBackward, (interpolatedColorForward.a + interpolatedColorBackward.a) * 0.5f);
            }
            else if (flowDirectionPreference == 1)
            {
                // Set forward color
                interpolatedColorBlended = glm::vec4(colorForward, interpolatedColorForward.a);
            }
            else if (flowDirectionPreference == 2)
            {
                // Set backward color
                interpolatedColorBlended = glm::vec4(colorBackward, interpolatedColorBackward.a);
            }

            // Use over blending for forward composting
            pixelColor = overBlend(pixelColor, interpolatedColorBlended);
        }

        // Blend with black color if alpha is not 1.0f
        if (pixelColor.a < 0.9999f)
        {
            pixelColor = overBlend(pixelColor, glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        }
    }

    return pixelColor;
}

// Function to normalize the pixels to device coordinates
__device__ glm::vec2 normalizePixels(const unsigned int &pixelIndex, const glm::vec2 &viewportSize)
{
    float normalizedDeviceCoordsX = (pixelIndex % static_cast<unsigned int>(viewportSize.x)) / static_cast<float>(viewportSize.x) - 0.5f;
    float normalizedDeviceCoordsY = (pixelIndex / viewportSize.x) / static_cast<float>(viewportSize.y) - 0.5f;

    return glm::vec2(normalizedDeviceCoordsX, normalizedDeviceCoordsY);
}

// Function to interpolate the parameters (FTLE values and normal vectors)
__device__ void interpolateParameters(const glm::vec3 &samplePoint, const Point *points, const unsigned int &resolution, const float &spacing, glm::vec2 &interpolatedScalars, glm::vec3 &interpolatedNormalsForward, glm::vec3 &interpolatedNormalsBackward)
{
    // Apply inverse transformation
    float halfSize = 0.5f * (resolution - 1) * spacing;
    float gridSpacingInverse = 1 / spacing;
    glm::vec3 transformedSamplePoint = (samplePoint + halfSize) * gridSpacingInverse;

    // Coordinates of left (x), bottom (y) and front (z) neighbour point
    int xmin = static_cast<int>(transformedSamplePoint.x);
    int ymin = static_cast<int>(transformedSamplePoint.y);
    int zmin = static_cast<int>(transformedSamplePoint.z);

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
    // Interpolate scalars
    glm::vec2 interpolatedScalarXFrontBottom = glm::mix(
        glm::vec2(points[frontBottomLeftIdx].ftleExponentForward,
                  points[frontBottomLeftIdx].ftleExponentBackward),
        glm::vec2(points[frontBottomRightIdx].ftleExponentForward,
                  points[frontBottomRightIdx].ftleExponentBackward),
        fractionalDistances.x);

    glm::vec2 interpolatedScalarXFrontTop = glm::mix(
        glm::vec2(points[frontTopLeftIdx].ftleExponentForward,
                  points[frontTopLeftIdx].ftleExponentBackward),
        glm::vec2(points[frontTopRightIdx].ftleExponentForward,
                  points[frontTopRightIdx].ftleExponentBackward),
        fractionalDistances.x);

    glm::vec2 interpolatedScalarXBackBottom = glm::mix(
        glm::vec2(points[backBottomLeftIdx].ftleExponentForward,
                  points[backBottomLeftIdx].ftleExponentBackward),
        glm::vec2(points[backBottomRightIdx].ftleExponentForward,
                  points[backBottomRightIdx].ftleExponentBackward),
        fractionalDistances.x);

    glm::vec2 interpolatedScalarXBackTop = glm::mix(
        glm::vec2(points[backTopLeftIdx].ftleExponentForward,
                  points[backTopLeftIdx].ftleExponentBackward),
        glm::vec2(points[backTopRightIdx].ftleExponentForward,
                  points[backTopRightIdx].ftleExponentBackward),
        fractionalDistances.x);

    // Interpolate normals
    glm::vec3 interpolatedNormalXFrontBottomForward = glm::mix(
        points[frontBottomLeftIdx].normalVectorForward,
        points[frontBottomRightIdx].normalVectorForward,
        fractionalDistances.x);

    glm::vec3 interpolatedNormalXFrontBottomBackward = glm::mix(
        points[frontBottomLeftIdx].normalVectorBackward,
        points[frontBottomRightIdx].normalVectorBackward,
        fractionalDistances.x);

    glm::vec3 interpolatedNormalXFrontTopForward = glm::mix(
        points[frontTopLeftIdx].normalVectorForward,
        points[frontTopRightIdx].normalVectorForward,
        fractionalDistances.x);

    glm::vec3 interpolatedNormalXFrontTopBackward = glm::mix(
        points[frontTopLeftIdx].normalVectorBackward,
        points[frontTopRightIdx].normalVectorBackward,
        fractionalDistances.x);

    glm::vec3 interpolatedNormalXBackBottomForward = glm::mix(
        points[backBottomLeftIdx].normalVectorForward,
        points[backBottomRightIdx].normalVectorForward,
        fractionalDistances.x);

    glm::vec3 interpolatedNormalXBackBottomBackward = glm::mix(
        points[backBottomLeftIdx].normalVectorBackward,
        points[backBottomRightIdx].normalVectorBackward,
        fractionalDistances.x);

    glm::vec3 interpolatedNormalXBackTopForward = glm::mix(
        points[backTopLeftIdx].normalVectorForward,
        points[backTopRightIdx].normalVectorForward,
        fractionalDistances.x);

    glm::vec3 interpolatedNormalXBackTopBackward = glm::mix(
        points[backTopLeftIdx].normalVectorBackward,
        points[backTopRightIdx].normalVectorBackward,
        fractionalDistances.x);

    // Perform linear interpolation along the y-axis
    // Interpolate scalars
    glm::vec2 interpolatedScalarYFront = glm::mix(
        interpolatedScalarXFrontBottom,
        interpolatedScalarXFrontTop,
        fractionalDistances.y);

    glm::vec2 interpolatedScalarYBack = glm::mix(
        interpolatedScalarXBackBottom,
        interpolatedScalarXBackTop,
        fractionalDistances.y);

    // Interpolate normals
    glm::vec3 interpolatedNormalYFrontForward = glm::mix(
        interpolatedNormalXFrontBottomForward,
        interpolatedNormalXFrontTopForward,
        fractionalDistances.y);

    glm::vec3 interpolatedNormalYFrontBackward = glm::mix(
        interpolatedNormalXFrontBottomBackward,
        interpolatedNormalXFrontTopBackward,
        fractionalDistances.y);

    glm::vec3 interpolatedNormalYBackForward = glm::mix(
        interpolatedNormalXBackBottomForward,
        interpolatedNormalXBackTopForward,
        fractionalDistances.y);

    glm::vec3 interpolatedNormalYBackBackward = glm::mix(
        interpolatedNormalXBackBottomBackward,
        interpolatedNormalXBackTopBackward,
        fractionalDistances.y);

    // Perform linear interpolation along the z-axis
    interpolatedScalars = glm::mix(
        interpolatedScalarYFront,
        interpolatedScalarYBack,
        fractionalDistances.z);

    interpolatedNormalsForward = glm::mix(
        interpolatedNormalYFrontForward,
        interpolatedNormalYBackForward,
        fractionalDistances.z);

    interpolatedNormalsBackward = glm::mix(
        interpolatedNormalYFrontBackward,
        interpolatedNormalYBackBackward,
        fractionalDistances.z);
}

// Function to linearize the coordinates
__device__ unsigned int linearizeIndex(const int &x, const int &y, const int &z, const unsigned int &resolution)
{
    return x + y * resolution + z * resolution * resolution;
}

// Function to blend foreground over background
__device__ glm::vec4 overBlend(const glm::vec4 &foreground, const glm::vec4 &background)
{
    // If the background is very transparent there is no point in blending
    if (background.a <= 0.0001f)
    {
        return foreground;
    }

    // Blend colors
    glm::vec4 blendedColor(0.0f);
    blendedColor.r = foreground.a * foreground.r + (1.0f - foreground.a) * background.r;
    blendedColor.g = foreground.a * foreground.g + (1.0f - foreground.a) * background.g;
    blendedColor.b = foreground.a * foreground.b + (1.0f - foreground.a) * background.b;
    blendedColor.a = foreground.a + (1.0f - foreground.a) * background.a;

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

        // Light moves along with the camera
        settings->light.position = renderer->getCamera()->position;

        // Render the image
        if (settings->antiAliasingEnabled)
        {
            volumeRenderAA<<<numBlocks, settings->cudaBlockSize, 0, stream>>>(
                renderer->getPixels_d_ptr(),
                renderer->getPixelsCount(),
                renderer->getPixelChannelsCount(),
                *renderer->getCamera(),
                grid->getPoints_d_ptr(),
                grid->getResolution(),
                grid->getSpacing(),
                *grid->getAABB(),
                settings->raysCount,
                settings->rayStepSize,
                settings->flowDirectionPreference,
                settings->tfForward,
                settings->tfBackward,
                settings->lightingEnabled,
                settings->light,
                settings->lighting,
                renderer->getCurandStates_d_ptr(),
                settings->antiAliasingIntensity);
        }
        else
        {
            volumeRender<<<numBlocks, settings->cudaBlockSize, 0, stream>>>(
                renderer->getPixels_d_ptr(),
                renderer->getPixelsCount(),
                renderer->getPixelChannelsCount(),
                *renderer->getCamera(),
                grid->getPoints_d_ptr(),
                grid->getResolution(),
                grid->getSpacing(),
                *grid->getAABB(),
                settings->rayStepSize,
                settings->flowDirectionPreference,
                settings->tfForward,
                settings->tfBackward,
                settings->lightingEnabled,
                settings->light,
                settings->lighting);
        }

        // Wait for pixels to be rendered
        cudaStreamSynchronize(stream);
    }
}

#endif // VOLUME_RENDER_CU
