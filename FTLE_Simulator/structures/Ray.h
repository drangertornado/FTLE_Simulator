#ifndef RAY_H
#define RAY_H

#pragma once

#include <curand_kernel.h>
#include "glm.hpp"
#include "Camera.h"

struct Ray
{
  glm::vec3 origin;
  glm::vec3 direction;

  __device__ Ray(const glm::vec2 &normalizedDeviceCoords, const Camera &camera)
  {
    origin = camera.position;
    direction = glm::normalize(camera.front + normalizedDeviceCoords.x * camera.right * camera.aspectRatio + normalizedDeviceCoords.y * camera.up);
  }

  __device__ Ray(const glm::vec2 &normalizedDeviceCoords, const Camera &camera, float antiAliasingIntensity, curandState &curandState)
  {
    origin = camera.position;
    direction = glm::normalize(camera.front + normalizedDeviceCoords.x * camera.right * camera.aspectRatio + normalizedDeviceCoords.y * camera.up);

    // Randomly jitter the ray direction
    direction.x += (0.5f - static_cast<float>(curand_uniform(&curandState))) * antiAliasingIntensity;
    direction.y += (0.5f - static_cast<float>(curand_uniform(&curandState))) * antiAliasingIntensity;
    direction.z += (0.5f - static_cast<float>(curand_uniform(&curandState))) * antiAliasingIntensity;

    // Normalize the direction after jittering
    direction = glm::normalize(direction);
  }

  __device__ Ray(const glm::vec3 &rayOrigin, const glm::vec3 &rayDirection)
      : origin(rayOrigin),
        direction(rayDirection) {}

  __device__ Ray()
      : origin(0.0f),
        direction(0.0f) {}
};

#endif // RAY_H
