#ifndef GRID_CPP
#define GRID_CPP

#include "Grid.h"
#include <iostream>
#include "FTLECompute.cuh"

// Constructor
Grid::Grid(Settings *settings) : settings(settings)
{
    // Initialize points in the host
    initializeGrid(settings->gridResolution, settings->gridSpacing);
    // Initialize buffers on the device
    initializeBuffers();
}

// Destructor
Grid::~Grid()
{
    // Perform cleanup
    pointsBuffer.free();
    tensorMatricesBuffer.free();
    singularValuesBuffer.free();

    // Destroy stream
    cudaStreamDestroy(stream);
}

// Function to initialize the points in a uniform 3D grid
void Grid::initializeGrid(unsigned int gridResolution, float gridSpacing)
{
    std::cout << "\nInitializing uniform 3D Grid: " << gridResolution << " x " << gridResolution << " x " << gridResolution << std::endl;
    std::cout << "Grid spacing: " << gridSpacing << std::endl;

    // Additonal boundary for computing normal vectors
    resolution = gridResolution + 2;
    spacing = gridSpacing;

    pointsCount = resolution * resolution * resolution;
    hiddenPointsCount = resolution * resolution * 6;
    points.reserve(pointsCount + hiddenPointsCount);

    // Additional hidden points for computing FTLE
    std::vector<Point> hiddenPoints;
    hiddenPoints.reserve(hiddenPointsCount);

    float halfSize = 0.5f * (resolution - 1) * spacing;
    // Initialize bounding box excluding the boundary points
    aabb = AABB(glm::vec3(-halfSize + spacing), glm::vec3(halfSize - spacing));

    unsigned int idx = 0;
    unsigned int hiddenIdx = 0;

    for (unsigned int z = 0; z < resolution; z++)
    {
        for (unsigned int y = 0; y < resolution; y++)
        {
            for (unsigned int x = 0; x < resolution; x++)
            {
                // Calculate position of the point in the grid
                float xPos = (x * spacing) - halfSize;
                float yPos = (y * spacing) - halfSize;
                float zPos = (z * spacing) - halfSize;

                // Calculate indices of the neighbouring points
                NeighbourIndex neighbour;

                // Set the left neighbour index
                if (x == 0)
                {
                    hiddenPoints.emplace_back(Point(glm::vec3(xPos - spacing, yPos, zPos)));
                    neighbour.left = pointsCount + hiddenIdx;
                    hiddenIdx++;
                }
                else
                {
                    neighbour.left = idx - 1;
                }

                // Set the right neighbour index
                if (x == resolution - 1)
                {
                    hiddenPoints.emplace_back(Point(glm::vec3(xPos + spacing, yPos, zPos)));
                    neighbour.right = pointsCount + hiddenIdx;
                    hiddenIdx++;
                }
                else
                {
                    neighbour.right = idx + 1;
                }

                // Set the up neighbour index
                if (y == resolution - 1)
                {
                    hiddenPoints.emplace_back(Point(glm::vec3(xPos, yPos + spacing, zPos)));
                    neighbour.up = pointsCount + hiddenIdx;
                    hiddenIdx++;
                }
                else
                {
                    neighbour.up = idx + resolution;
                }

                // Set the down neighbour index
                if (y == 0)
                {
                    hiddenPoints.emplace_back(Point(glm::vec3(xPos, yPos - spacing, zPos)));
                    neighbour.down = pointsCount + hiddenIdx;
                    hiddenIdx++;
                }
                else
                {
                    neighbour.down = idx - resolution;
                }

                // Set the front neighbour index
                if (z == 0)
                {
                    hiddenPoints.emplace_back(Point(glm::vec3(xPos, yPos, zPos - spacing)));
                    neighbour.front = pointsCount + hiddenIdx;
                    hiddenIdx++;
                }
                else
                {
                    neighbour.front = idx - resolution * resolution;
                }

                // Set the back neighbour index
                if (z == resolution - 1)
                {
                    hiddenPoints.emplace_back(Point(glm::vec3(xPos, yPos, zPos + spacing)));
                    neighbour.back = pointsCount + hiddenIdx;
                    hiddenIdx++;
                }
                else
                {
                    neighbour.back = idx + resolution * resolution;
                }

                idx++;
                points.emplace_back(Point(glm::vec3(xPos, yPos, zPos), neighbour));
            }
        }
    }

    std::cout << "Total no of Points: " << points.size() << std::endl;
    std::cout << "Total no of Hidden Points: " << hiddenPoints.size() << std::endl;

    points.insert(points.end(), hiddenPoints.begin(), hiddenPoints.end());
}

// Function to initialize the buffers
void Grid::initializeBuffers()
{
    // Create CUDA stream
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Copy points to device
    pointsBuffer.allocAndUpload(points);

    // Allocate temporary buffers for SVD
    tensorMatricesCount = pointsCount * 2;
    singularValuesCount = pointsCount * 2;
    tensorMatricesBuffer.alloc(sizeof(float) * SIZE_OF_MATRIX * tensorMatricesCount);
    singularValuesBuffer.alloc(sizeof(float) * SIZE_OF_VECTOR * singularValuesCount);

    // Initialize CuSolver parameters
    cuSolverSvdParams.jobz = CUSOLVER_EIG_MODE_NOVECTOR;
    CUSOLVER_CHECK(cusolverDnXgesvdjSetTolerance(cuSolverSvdParams.params, settings->svdTolerance));
    CUSOLVER_CHECK(cusolverDnXgesvdjSetMaxSweeps(cuSolverSvdParams.params, settings->svdMaxSweeps));
    CUSOLVER_CHECK(cusolverDnXgesvdjSetSortEig(cuSolverSvdParams.params, settings->svdSort));
    cuSolverSvdParams.batchSize = tensorMatricesCount;
    cuSolverSvdParams.m = 3;
    cuSolverSvdParams.n = 3;
    cuSolverSvdParams.A = static_cast<float *>(tensorMatricesBuffer.d_ptr);
    cuSolverSvdParams.lda = 3;
    cuSolverSvdParams.S = static_cast<float *>(singularValuesBuffer.d_ptr);
    cuSolverSvdParams.ldu = 3;
    cuSolverSvdParams.ldv = 3;

    // Query working space needed for gesvdjBatched
    CUSOLVER_CHECK(cusolverDnSgesvdjBatched_bufferSize(
        cuSolverSvdParams.handle,
        cuSolverSvdParams.jobz,
        cuSolverSvdParams.m,
        cuSolverSvdParams.n,
        cuSolverSvdParams.A,
        cuSolverSvdParams.lda,
        cuSolverSvdParams.S,
        cuSolverSvdParams.U,
        cuSolverSvdParams.ldu,
        cuSolverSvdParams.V,
        cuSolverSvdParams.ldv,
        &cuSolverSvdParams.lwork,
        cuSolverSvdParams.params,
        cuSolverSvdParams.batchSize));

    // Allocate space for working memory
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&cuSolverSvdParams.d_work), sizeof(float) * cuSolverSvdParams.lwork));

    // Allocate space for debug info
    cuSolverSvdParams.info = std::vector<int>(cuSolverSvdParams.batchSize, 0);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&cuSolverSvdParams.d_info), sizeof(int) * cuSolverSvdParams.info.size()));
}

void Grid::computeFTLE(float integrationStartTime, float integrationDuration)
{
    // Compute FTLE exponent
    FTLEComputeKernel::computeFtle(this, settings, integrationStartTime, integrationDuration, stream);
}

void Grid::printFTLEPoints(std::vector<Point> points)
{
    for (unsigned int i = 0; i < pointsCount; i++)
    {
        Point point = points[i];
        std::cout << "Point ID: " << i << std::endl;
        std::cout << "Initial Position (" << point.position.initial.x << ", " << point.position.initial.y << ", " << point.position.initial.z << ")" << std::endl;
        std::cout << "Current Position (Fwd) (" << point.position.currentForward.x << ", " << point.position.currentForward.y << ", " << point.position.currentForward.z << ")" << std::endl;
        std::cout << "Current Position (Bck) (" << point.position.currentBackward.x << ", " << point.position.currentBackward.y << ", " << point.position.currentBackward.z << ")" << std::endl;
        std::cout << "FTLE Exponent (Fwd): " << point.ftleExponentForward << std::endl;
        std::cout << "FTLE Exponent (Bck): " << point.ftleExponentBackward << std::endl;
        std::cout << "Normal Vector (Fwd): (" << point.normalVectorForward.x << ", " << point.normalVectorForward.y << ", " << point.normalVectorForward.z << ")" << std::endl;
        std::cout << "Normal Vector (Bck): (" << point.normalVectorBackward.x << ", " << point.normalVectorBackward.y << ", " << point.normalVectorBackward.z << ")" << std::endl;
    }
}

void Grid::downloadAndPrintFTLEPoints()
{
    // Copy points from device to host
    pointsBuffer.download(points.data(), points.size());
    std::cout << "\nFTLE Values: " << std::endl;

    printFTLEPoints(points);
}

unsigned int Grid::getResolution()
{
    return resolution;
}

float Grid::getSpacing()
{
    return spacing;
}

Point *Grid::getPoints_d_ptr()
{
    return static_cast<Point *>(pointsBuffer.d_ptr);
}

unsigned int Grid::getPointsCount()
{
    return pointsCount;
}

unsigned int Grid::getHiddenPointsCount()
{
    return hiddenPointsCount;
}

AABB *Grid::getAABB()
{
    return &aabb;
}

CuSolverSvdParams *Grid::getCuSolverSvdParams()
{
    return &cuSolverSvdParams;
}

float *Grid::getTensorMatrices_d_ptr()
{
    return static_cast<float *>(tensorMatricesBuffer.d_ptr);
}

unsigned int Grid::gettensorMatricesCount()
{
    return tensorMatricesCount;
}

float *Grid::getSingularValues_d_ptr()
{
    return static_cast<float *>(singularValuesBuffer.d_ptr);
}

unsigned int Grid::getSingularValuesCount()
{
    return singularValuesCount;
}

#endif // GRID_CPP
