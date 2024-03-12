#ifndef GRID_H
#define GRID_H

#pragma once

#include <vector>
#include "CUDABuffer.h"
#include "CuSolver.h"
#include "Point.h"
#include "AABB.h"
#include "Settings.h"

class Grid
{
public:
  Grid(Settings *settings);
  ~Grid();

  void computeFTLE(float integrationStartTime, float integrationDuration);

  void printFTLEPoints(std::vector<Point> points);
  void downloadAndPrintFTLEPoints();

  unsigned int getResolution();
  float getSpacing();

  Point *getPoints_d_ptr();
  unsigned int getPointsCount();
  unsigned int getHiddenPointsCount();
  AABB *getAABB();

  CuSolverSvdParams *getCuSolverSvdParams();
  float *getTensorMatrices_d_ptr();
  unsigned int gettensorMatricesCount();
  float *getSingularValues_d_ptr();
  unsigned int getSingularValuesCount();

  const unsigned int SIZE_OF_MATRIX = 9, SIZE_OF_VECTOR = 3;

private:
  void initializeGrid(unsigned int gridResolution, float gridSpacing);

  void initializeBuffers();

  cudaStream_t stream;

  unsigned int resolution;
  float spacing;

  std::vector<Point> points;
  CUDABuffer pointsBuffer;
  unsigned int pointsCount;
  unsigned int hiddenPointsCount;
  AABB aabb;

  CuSolverSvdParams cuSolverSvdParams;
  CUDABuffer tensorMatricesBuffer, singularValuesBuffer;
  unsigned int tensorMatricesCount, singularValuesCount;

  Settings *settings = nullptr;
};

#endif // GRID_H
