#ifndef AABB_H
#define AABB_H

#pragma once

#include "glm.hpp"
#include "Ray.h"

struct AABB
{
  glm::vec3 min;
  glm::vec3 max;

  __host__ AABB(glm::vec3 minBound, glm::vec3 maxBound)
      : min(minBound),
        max(maxBound) {}

  __host__ AABB()
      : min(0.0f),
        max(0.0f) {}

  // Adapted from: https://tavianator.com
  __device__ bool intersectRay(const Ray &ray, float &tmin, float &tmax)
  {
    glm::vec3 invDirection = 1.0f / ray.direction;

    tmin = (min.x - ray.origin.x) * invDirection.x;
    tmax = (max.x - ray.origin.x) * invDirection.x;

    if (tmin > tmax)
    {
      float temp = tmin;
      tmin = tmax;
      tmax = temp;
    }

    float tymin = (min.y - ray.origin.y) * invDirection.y;
    float tymax = (max.y - ray.origin.y) * invDirection.y;

    if (tymin > tymax)
    {
      float temp = tymin;
      tymin = tymax;
      tymax = temp;
    }

    if ((tmin > tymax) || (tymin > tmax))
      return false;

    if (tymin > tmin)
      tmin = tymin;
    if (tymax < tmax)
      tmax = tymax;

    float tzmin = (min.z - ray.origin.z) * invDirection.z;
    float tzmax = (max.z - ray.origin.z) * invDirection.z;

    if (tzmin > tzmax)
    {
      float temp = tzmin;
      tzmin = tzmax;
      tzmax = temp;
    }

    if ((tmin > tzmax) || (tzmin > tmax))
      return false;

    if (tzmin > tmin)
      tmin = tzmin;
    if (tzmax < tmax)
      tmax = tzmax;

    // To prevent 2 volumes from appearing
    if (tmin < 0.0f && tmax < 0.0f)
      return false;
    
    // To render correctly when camera is inside AABB
    tmin = abs(tmin);

    return true;
  }
};

#endif // AABB_H
