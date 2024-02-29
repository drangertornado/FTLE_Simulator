#ifndef CAMERA_H
#define CAMERA_H

#pragma once

#include "glm.hpp"
#include "matrix_transform.hpp"

struct Camera
{
  glm::vec3 position;
  glm::vec3 front;
  glm::vec3 up;
  glm::vec3 right;
  glm::vec3 worldUp;
  glm::vec2 viewportSize;
  float aspectRatio;
  float focalLength;
  float focusDistance;
  float apertureSize;

  glm::mat4 view;
  glm::mat4 projection;

  float pitch;
  float yaw;

  __host__ Camera(glm::vec3 cameraPosition, glm::vec3 cameraFront, glm::vec3 cameraUp, float screenWidth, float screenHeight)
      : position(cameraPosition),
        front(cameraFront),
        up(cameraUp),
        worldUp(cameraUp),
        pitch(0.0f),
        yaw(-90.0f)
  {
    right = glm::normalize(glm::cross(front, up));
    view = glm::lookAt(position, front, up);
    viewportSize = glm::vec2(screenWidth, screenHeight);
    aspectRatio = screenWidth / screenHeight;
  }

  __host__ Camera()
      : position(0.0f),
        front(0.0f),
        up(0.0f),
        right(0.0f),
        viewportSize(0.0f),
        aspectRatio(0.0f),
        focalLength(0.0f),
        focusDistance(0.0f),
        apertureSize(0.0f),
        pitch(0.0f),
        yaw(0.0f) {}

  // Camera controls
  void moveForward(float magnitude)
  {
    position += front * magnitude;
  }

  void moveBackward(float magnitude)
  {
    position -= front * magnitude;
  }

  void moveLeft(float magnitude)
  {
    position -= right * magnitude;
  }

  void moveRight(float magnitude)
  {
    position += right * magnitude;
  }

  void moveUp(float magnitude)
  {
    position += up * magnitude;
  }

  void moveDown(float magnitude)
  {
    position -= up * magnitude;
  }

  // Camera rotation
  void rotate(glm::vec2 offset)
  {
    yaw += offset.x;
    pitch += offset.y;

    // Clamp pitch
    if (pitch > 89.0f)
      pitch = 89.0f;
    if (pitch < -89.0f)
      pitch = -89.0f;

    glm::vec3 direction;
    direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    direction.y = sin(glm::radians(pitch));
    direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    front = glm::normalize(direction);
    right = glm::normalize(glm::cross(front, worldUp));
    up = glm::normalize(glm::cross(right, front));
    view = glm::lookAt(position, front, up);
  }
};

#endif // CAMERA_H
