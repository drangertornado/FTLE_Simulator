#ifndef POINT_H
#define POINT_H

#pragma once

#include <vector>
#include "glm.hpp"

struct Position
{
  glm::vec3 initial, currentForward, currentBackward;

  __host__ Position(glm::vec3 &position)
      : initial(position),
        currentForward(position),
        currentBackward(position) {}

  __host__ Position()
      : initial(0.0f),
        currentForward(0.0f),
        currentBackward(0.0f) {}
};

struct NeighbourIndex
{
  unsigned int left, right, up, down, front, back;

  __host__ NeighbourIndex(unsigned int leftIndex, unsigned int rightIndex, unsigned int upIndex, unsigned int downIndex, unsigned int frontIndex, unsigned int backIndex)
      : left(leftIndex),
        right(rightIndex),
        up(upIndex),
        down(downIndex),
        front(frontIndex),
        back(backIndex) {}

  __host__ NeighbourIndex()
      : left(0),
        right(0),
        up(0),
        down(0),
        front(0),
        back(0) {}
};

struct Point
{
  Position position;
  NeighbourIndex neighbourIndex;
  float ftleExponentForward, ftleExponentBackward;
  glm::vec3 normalVectorForward, normalVectorBackward;

  __host__ Point(glm::vec3 pointPosition, NeighbourIndex neighnourIndex)
      : position(Position(pointPosition)),
        neighbourIndex(neighnourIndex),
        ftleExponentForward(0.0f),
        ftleExponentBackward(0.0f),
        normalVectorForward(0.0f),
        normalVectorBackward(0.0f) {}

  __host__ Point(glm::vec3 pointPosition)
      : position(Position(pointPosition)),
        neighbourIndex(),
        ftleExponentForward(0.0f),
        ftleExponentBackward(0.0f),
        normalVectorForward(0.0f),
        normalVectorBackward(0.0f) {}

  __host__ Point()
      : position(),
        neighbourIndex(),
        ftleExponentForward(0.0f),
        ftleExponentBackward(0.0f),
        normalVectorForward(0.0f),
        normalVectorBackward(0.0f) {}
};

#endif // POINT_H
