#ifndef VOLUME_RENDER_CUH
#define VOLUME_RENDER_CUH

#pragma once

#include "CUDABuffer.h"
#include "Camera.h"
#include "Ray.h"
#include "Light.h"
#include "PhongLighting.h"
#include "Point.h"
#include "AABB.h"
#include "Grid.h"
#include "TransferFunction.h"
#include "Renderer.h"
#include "Settings.h"

__global__ void setupCurandStates(curandState *curandStates, unsigned int curandStatesCount);

__global__ void volumeRender(unsigned char *pixels,
							 unsigned int pixelsCount,
							 unsigned int channelsPerPixel,
							 Camera camera,
							 Point *points,
							 unsigned int gridResolution,
							 float gridSpacing,
							 AABB gridAABB,
							 float rayStepSize,
							 unsigned int flowDirectionPreference,
							 TransferFunction tfForward,
							 TransferFunction tfBackward,
							 bool lightingEnabled,
							 PointLight light,
							 PhongLighting lighting);

__global__ void volumeRenderAA(unsigned char *pixels,
							   unsigned int pixelsCount,
							   unsigned int channelsPerPixel,
							   Camera camera,
							   Point *points,
							   unsigned int gridResolution,
							   float gridSpacing,
							   AABB gridAABB,
							   unsigned int raysCount,
							   float rayStepSize,
							   unsigned int flowDirectionPreference,
							   TransferFunction tfForward,
							   TransferFunction tfBackward,
							   bool lightingEnabled,
							   PointLight light,
							   PhongLighting lighting,
							   curandState *curandStates,
							   float antiAliasingIntensity);

__device__ glm::vec4 rayMarch(const Ray &ray,
							  const float &rayStepSize,
							  const Point *points,
							  const unsigned int &gridResolution,
							  const float &gridSpacing,
							  AABB &gridAABB,
							  const unsigned int &flowDirectionPreference,
							  TransferFunction &tfForward,
							  TransferFunction &tfBackward,
							  bool &lightingEnabled,
							  PointLight &light,
							  PhongLighting &lighting);

__device__ glm::vec4 rayMarchAA(const Ray &ray,
								const float &rayStepSize,
								const Point *points,
								const unsigned int &gridResolution,
								const float &gridSpacing,
								AABB &gridAABB,
								const unsigned int &flowDirectionPreference,
								TransferFunction &tfForward,
								TransferFunction &tfBackward,
								bool &lightingEnabled,
								PointLight &light,
								PhongLighting &lighting,
								curandState &curandState,
								const float &antiAliasingIntensity);

__device__ glm::vec2 normalizePixels(const unsigned int &pixelIndex, const glm::vec2 &viewportSize);

__device__ void interpolateParameters(const glm::vec3 &samplePoint,
									  const Point *points,
									  const unsigned int &resolution,
									  const float &spacing,
									  glm::vec2 &interpolatedScalars,
									  glm::vec3 &interpolatedNormalsForward,
									  glm::vec3 &interpolatedNormalsBackward);

__device__ unsigned int linearizeIndex(const int &x, const int &y, const int &z, const unsigned int &resolution);

__device__ glm::vec4 overBlend(const glm::vec4 &foreground, const glm::vec4 &background);

namespace VolumeRenderKernel
{
	__host__ void initializeCurand(Renderer *renderer, Settings *settings, cudaStream_t &stream);

	__host__ void renderPixels(Renderer *renderer, Grid *grid, Settings *settings, cudaStream_t &stream);
}

#endif // VOLUME_RENDER_CUH
