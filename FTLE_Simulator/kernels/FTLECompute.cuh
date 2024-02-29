#ifndef FTLE_COMPUTE_CUH
#define FTLE_COMPUTE_CUH

#pragma once

#include "CUDABuffer.h"
#include "CuSolver.h"
#include "Point.h"
#include "Grid.h"
#include "Settings.h"

__global__ void computeFtleExponent(Point *d_points,
									unsigned int pointsCount,
									float *d_singularValues,
									float integrationDuration);

__global__ void computeFlowMapTensor(Point *d_points,
									 unsigned int pointsCount,
									 float gridSpacing,
									 float *d_flowMapTensorMatrices);

__global__ void integratePositionForward(Point *d_points,
										 unsigned int pointsCount,
										 float integrationStartTime,
										 float integrationEndTime,
										 float stepSize);

__global__ void integratePositionBackward(Point *d_points,
										  unsigned int pointsCount,
										  float integrationStartTime,
										  float integrationEndTime,
										  float stepSize);

__device__ glm::vec3 rk4Integration(const glm::vec3 &startPosition,
									const float &startTime,
									const float &endTime,
									const bool &integrateForward = true,
									const float &stepSize = 0.05f);

__device__ glm::vec3 odeAbcFlow(const glm::vec3 &position, const float &time);

__host__ void computeSingularValues(CuSolverSvdParams *cuSolverSvdParams, cudaStream_t stream);

namespace FTLEComputeKernel
{
	__host__ void computeFtle(Grid *grid,
							  Settings *settings,
							  float integrationStartTime,
							  float integrationDuration,
							  cudaStream_t stream);
}

#endif // FTLE_COMPUTE_CUH
