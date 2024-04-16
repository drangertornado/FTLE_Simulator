#ifndef FTLE_COMPUTE_CU
#define FTLE_COMPUTE_CU

#include "FTLECompute.cuh"
#include "glm.hpp"

// Function to compute the normal vector
__global__ void computeNormalVector(Point *d_points, unsigned int pointsCount)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pointsCount)
    {
        // Initialize neighbouring point FTLE values
       float neighbourFtleLeftForward,
            neighbourFtleLeftBackward,
            neighbourFtleRightForward,
            neighbourFtleRightBackward,
            neighbourFtleUpForward,
            neighbourFtleUpBackward,
            neighbourFtleDownForward,
            neighbourFtleDownBackward,
            neighbourFtleFrontForward,
            neighbourFtleFrontBackward,
            neighbourFtleBackForward,
            neighbourFtleBackBackward;

        // Determine neighbouring FTLE values from index
        // Left neighbour FTLE value
        neighbourFtleLeftForward = d_points[d_points[idx].neighbourIndex.left].ftleExponentForward;
        neighbourFtleLeftBackward = d_points[d_points[idx].neighbourIndex.left].ftleExponentBackward;

        // Right neighbour FTLE value
        neighbourFtleRightForward = d_points[d_points[idx].neighbourIndex.right].ftleExponentForward;
        neighbourFtleRightBackward = d_points[d_points[idx].neighbourIndex.right].ftleExponentBackward;

        // Top neighbour FTLE value
        neighbourFtleUpForward = d_points[d_points[idx].neighbourIndex.up].ftleExponentForward;
        neighbourFtleUpBackward = d_points[d_points[idx].neighbourIndex.up].ftleExponentBackward;

        // Bottom neighbour FTLE value
        neighbourFtleDownForward = d_points[d_points[idx].neighbourIndex.down].ftleExponentForward;
        neighbourFtleDownBackward = d_points[d_points[idx].neighbourIndex.down].ftleExponentBackward;

        // Front neighbour FTLE value
        neighbourFtleFrontForward = d_points[d_points[idx].neighbourIndex.front].ftleExponentForward;
        neighbourFtleFrontBackward = d_points[d_points[idx].neighbourIndex.front].ftleExponentBackward;

        // Back neighbour FTLE value
        neighbourFtleBackForward = d_points[d_points[idx].neighbourIndex.back].ftleExponentForward;
        neighbourFtleBackBackward = d_points[d_points[idx].neighbourIndex.back].ftleExponentBackward;

        // Compute the gradient
        glm::vec3 gradientForward = glm::vec3(neighbourFtleRightForward - neighbourFtleLeftForward, neighbourFtleUpForward - neighbourFtleDownForward, neighbourFtleBackForward - neighbourFtleFrontForward);
        glm::vec3 gradientBackward = glm::vec3(neighbourFtleRightBackward - neighbourFtleLeftBackward, neighbourFtleUpBackward - neighbourFtleDownBackward, neighbourFtleBackBackward - neighbourFtleFrontBackward);

        // Save the normal vector
        d_points[idx].normalVectorForward = glm::normalize(gradientForward);
        d_points[idx].normalVectorBackward = glm::normalize(gradientBackward);
    }
}

// Function to compute the FTLE exponent
__global__ void computeFtleExponent(Point *d_points, unsigned int pointsCount, float *d_singularValues, float integrationDuration)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pointsCount)
    {
        // Square roots of the eigenvalues are equal to the singular values for real symmetric positive semi-definite matrix
        float singularValueForward = d_singularValues[idx * 6];
        float singularValueBackward = d_singularValues[idx * 6 + 3];

        // Compute FTLE exponent
        d_points[idx].ftleExponentForward = log(singularValueForward) / integrationDuration;
        d_points[idx].ftleExponentBackward = log(singularValueBackward) / integrationDuration;
    }
}

// Function to compute the flow map tensor
__global__ void computeFlowMapTensor(Point *d_points, unsigned int pointsCount, float gridSpacing, float *d_flowMapTensorMatrices)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pointsCount)
    {
        // Initialize neighbouring point positions
        glm::vec3 neighbourPositionLeftCurrentForward,
            neighbourPositionLeftCurrentBackward,
            neighbourPositionRightCurrentForward,
            neighbourPositionRightCurrentBackward,
            neighbourPositionUpCurrentForward,
            neighbourPositionUpCurrentBackward,
            neighbourPositionDownCurrentForward,
            neighbourPositionDownCurrentBackward,
            neighbourPositionFrontCurrentForward,
            neighbourPositionFrontCurrentBackward,
            neighbourPositionBackCurrentForward,
            neighbourPositionBackCurrentBackward;

        // Determine neighbouring point positions from index
        // Left neighbour positions
        neighbourPositionLeftCurrentForward = d_points[d_points[idx].neighbourIndex.left].position.currentForward;
        neighbourPositionLeftCurrentBackward = d_points[d_points[idx].neighbourIndex.left].position.currentBackward;

        // Right neighbour positions
        neighbourPositionRightCurrentForward = d_points[d_points[idx].neighbourIndex.right].position.currentForward;
        neighbourPositionRightCurrentBackward = d_points[d_points[idx].neighbourIndex.right].position.currentBackward;

        // Top neighbour positions
        neighbourPositionUpCurrentForward = d_points[d_points[idx].neighbourIndex.up].position.currentForward;
        neighbourPositionUpCurrentBackward = d_points[d_points[idx].neighbourIndex.up].position.currentBackward;

        // Bottom neighbour positions
        neighbourPositionDownCurrentForward = d_points[d_points[idx].neighbourIndex.down].position.currentForward;
        neighbourPositionDownCurrentBackward = d_points[d_points[idx].neighbourIndex.down].position.currentBackward;

        // Front neighbour positions
        neighbourPositionFrontCurrentForward = d_points[d_points[idx].neighbourIndex.front].position.currentForward;
        neighbourPositionFrontCurrentBackward = d_points[d_points[idx].neighbourIndex.front].position.currentBackward;

        // Back neighbour positions
        neighbourPositionBackCurrentForward = d_points[d_points[idx].neighbourIndex.back].position.currentForward;
        neighbourPositionBackCurrentBackward = d_points[d_points[idx].neighbourIndex.back].position.currentBackward;

        // Compute flow map jacobian matrix
        float distanceBetweenPoints = 2.0f * gridSpacing;

        glm::mat3 flowMapJacobianForward = glm::mat3(
            (neighbourPositionRightCurrentForward - neighbourPositionLeftCurrentForward) / distanceBetweenPoints,
            (neighbourPositionUpCurrentForward - neighbourPositionDownCurrentForward) / distanceBetweenPoints,
            (neighbourPositionBackCurrentForward - neighbourPositionFrontCurrentForward) / distanceBetweenPoints);

        glm::mat3 flowMapJacobianBackward = glm::mat3(
            (neighbourPositionRightCurrentBackward - neighbourPositionLeftCurrentBackward) / distanceBetweenPoints,
            (neighbourPositionUpCurrentBackward - neighbourPositionDownCurrentBackward) / distanceBetweenPoints,
            (neighbourPositionBackCurrentBackward - neighbourPositionFrontCurrentBackward) / distanceBetweenPoints);

        // Compute flow map tensor matrix
        glm::mat3 flowMapTensorForward = glm::transpose(flowMapJacobianForward) * flowMapJacobianForward;
        glm::mat3 flowMapTensorBackward = glm::transpose(flowMapJacobianBackward) * flowMapJacobianBackward;

        // Save the result
        for (unsigned int i = 0; i < 3; i++)
        {
            int idxBase = idx * 18 + i * 3;
            for (unsigned int j = 0; j < 3; j++)
            {
                d_flowMapTensorMatrices[idxBase + j] = flowMapTensorForward[i][j];
                d_flowMapTensorMatrices[idxBase + 9 + j] = flowMapTensorBackward[i][j];
            }
        }
    }
}

// Function to integrate the positions forward in time
__global__ void integratePositionForward(Point *d_points, unsigned int pointsCount, float integrationStartTime, float integrationEndTime, float stepSize)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pointsCount)
    {
        d_points[idx].position.currentForward = rk4Integration(d_points[idx].position.initial, integrationStartTime, integrationEndTime, true, stepSize);
    }
}

// Function to integrate the positions backward in time
__global__ void integratePositionBackward(Point *d_points, unsigned int pointsCount, float integrationStartTime, float integrationEndTime, float stepSize)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pointsCount)
    {
        d_points[idx].position.currentBackward = rk4Integration(d_points[idx].position.initial, integrationStartTime, integrationEndTime, false, stepSize);
    }
}

// Function to compute the new point position based on the Runge-Kutta method (RK4) integration scheme
// Ref: https://math.okstate.edu/people/yqwang/teaching/math4513_fall11/Notes/rungekutta.pdf
__device__ glm::vec3 rk4Integration(const glm::vec3 &startPosition, const float &startTime, const float &endTime, const bool &integrateForward, const float &stepSize)
{
    glm::vec3 position = startPosition;
    float time = startTime;
    glm::vec3 k1, k2, k3, k4;

    // Perform RK4 iterations
    while ((integrateForward && (time < endTime)) || (!integrateForward && (time > endTime)))
    {
        float directionFactor = integrateForward ? 1.0f : -1.0f;
        k1 = stepSize * odeAbcFlow(position, time);
        k2 = stepSize * odeAbcFlow(position + directionFactor * k1 / 2.0f, time + directionFactor * stepSize / 2.0f);
        k3 = stepSize * odeAbcFlow(position + directionFactor * k2 / 2.0f, time + directionFactor * stepSize / 2.0f);
        k4 = stepSize * odeAbcFlow(position + directionFactor * k3, time + directionFactor * stepSize);

        position += directionFactor * (k1 + 2.0f * k2 + 2.0f * k3 + k4) / 6.0f;

        float timeIncrement = directionFactor * stepSize;
        if ((integrateForward && (time + timeIncrement) > endTime) ||
            (!integrateForward && (time - timeIncrement) < endTime))
        {
            time = endTime;
        }
        else
        {
            time += timeIncrement;
        }
    }

    return position;
}

// Function to compute the flow velocity at a particular time using ABC (Arnold-Beltrami-Childress) flow
__device__ glm::vec3 odeAbcFlow(const glm::vec3 &position, const float &time)
{
    glm::vec3 velocity;
    float PI = 3.1415927f;
    float c_t = sqrt(3.0f) + (1 - exp(-0.1f * time) * sin(2.0f * PI * time));

    velocity.x = c_t * sin(position.z) + cos(position.y);
    velocity.y = sqrt(2.0f) * sin(position.x) + c_t * cos(position.z);
    velocity.z = sin(position.y) + sqrt(2.0f) * cos(position.x);

    return velocity;
}

// Function to compute the singular values
__host__ void computeSingularValues(CuSolverSvdParams *cuSolverSvdParams, cudaStream_t stream)
{
    // Run on specific stream
    cusolverDnSetStream(cuSolverSvdParams->handle, stream);

    // Compute the singular values of the matrices in batch
    cusolverDnSgesvdjBatched(
        cuSolverSvdParams->handle,
        cuSolverSvdParams->jobz,
        cuSolverSvdParams->m,
        cuSolverSvdParams->n,
        cuSolverSvdParams->A,
        cuSolverSvdParams->lda,
        cuSolverSvdParams->S,
        cuSolverSvdParams->U,
        cuSolverSvdParams->ldu,
        cuSolverSvdParams->V,
        cuSolverSvdParams->ldv,
        cuSolverSvdParams->d_work,
        cuSolverSvdParams->lwork,
        cuSolverSvdParams->d_info,
        cuSolverSvdParams->params,
        cuSolverSvdParams->batchSize);
}

// External C++ interface
namespace FTLEComputeKernel
{
    __host__ void computeFtle(Grid *grid, Settings *setttings, float integrationStartTime, float integrationDuration, cudaStream_t stream)
    {
        // Consider hidden points for integration
        unsigned int numBlocks = (grid->getPointsCount() + grid->getHiddenPointsCount() + setttings->cudaBlockSize - 1) / setttings->cudaBlockSize;
        float integrationEndTime = integrationStartTime + integrationDuration;

        integratePositionForward<<<numBlocks, setttings->cudaBlockSize, 0, stream>>>(
            (Point *)grid->getPoints_d_ptr(),
            grid->getPointsCount() + grid->getHiddenPointsCount(),
            integrationStartTime,
            integrationEndTime,
            setttings->integrationStepSize);

        integratePositionBackward<<<numBlocks, setttings->cudaBlockSize, 0, stream>>>(
            (Point *)grid->getPoints_d_ptr(),
            grid->getPointsCount() + grid->getHiddenPointsCount(),
            integrationEndTime, // Integrate backwards in time so pass integration end time as start time
            integrationStartTime,
            setttings->integrationStepSize);

        // Wait for integration to complete
        cudaStreamSynchronize(stream);

        // Compute flow map tensor excluding the hidden points
        numBlocks = (grid->getPointsCount() + setttings->cudaBlockSize - 1) / setttings->cudaBlockSize;

        computeFlowMapTensor<<<numBlocks, setttings->cudaBlockSize, 64, stream>>>(
            grid->getPoints_d_ptr(),
            grid->getPointsCount(),
            setttings->gridSpacing,
            grid->getTensorMatrices_d_ptr());
        cudaStreamSynchronize(stream);

        // Compute singular values
        computeSingularValues(grid->getCuSolverSvdParams(), stream);
        cudaStreamSynchronize(stream);

        // Compute FTLE exponent
        computeFtleExponent<<<numBlocks, setttings->cudaBlockSize, 0, stream>>>(grid->getPoints_d_ptr(), grid->getPointsCount(), grid->getSingularValues_d_ptr(), integrationDuration);
        cudaStreamSynchronize(stream);

        // Compute Normals
        computeNormalVector<<<numBlocks, setttings->cudaBlockSize, 0, stream>>>(grid->getPoints_d_ptr(), grid->getPointsCount());
        cudaStreamSynchronize(stream);
    }
}

#endif // FTLE_COMPUTE_CU
