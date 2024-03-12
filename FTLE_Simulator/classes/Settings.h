#ifndef SETTINGS_H
#define SETTINGS_H

#pragma once

#include "glm.hpp"
#include "TransferFunction.h"

class Settings
{
public:
    /* Window */
    const unsigned int screenWidth = 800;
    const unsigned int screenHeight = 600;
    const bool fullScreen = false;

    /* Grid */
    // Set the grid resolution
    const unsigned int gridResolution = 25;
    // Set the grid spacing
    const float gridSpacing = 0.5f;

    /* Camera and inputs */
    float movementSpeed = 4.0f;
    float mouseSensitivity = 0.01f;
    bool invertMouseVertically = true;

    /* Transfer functions */
    // Transfer function for forward flow
    TransferFunction tfForward{
        1.0f,                        // Minimum scalar value
        2.0f,                        // Maximum scalar value
        glm::vec3(0.0f, 0.5f, 0.0f), // Color at minimum
        glm::vec3(0.0f, 1.0f, 0.0f), // Color at maximum
        1.0f                         // Maximum opacity
    };
    // Transfer function for backward flow
    TransferFunction tfBackward{
        1.0f,                        // Minimum scalar value
        2.0f,                        // Maximum scalar value
        glm::vec3(0.5f, 0.0f, 0.5f), // Color at minimum
        glm::vec3(1.0f, 0.0f, 1.0f), // Color at maximum
        1.0f                         // Maximum opacity
    };

    /* RK4 integration */
    // Integration start time
    float integrationStartTime = 0.0f;
    // Integration duration
    float integrationDuration = 0.01f;
    // Step size for RK4
    float integrationStepSize = 0.001f;
    // Step for adjusting the start time on key press
    float integrationStartTimeStepSize = 0.01f;
    // Step for adjusting the integration duration on key press
    float integrationDurationStepSize = 0.01f;

    /* SVD Calculation */
    // Set error tolerance level
    float svdTolerance = 0.000001f;
    // Max number of iterations
    unsigned int svdMaxSweeps = 5;
    // Sort SVD in decending order
    unsigned int svdSort = 1;

    /* Ray tracing */
    unsigned int raysCount = 4;
    float rayStepSize = 0.05f;
    bool antiAliasingEnabled = true;
    float antiAliasingIntensity = 0.001f;

    /* CUDA */
    unsigned int cudaBlockSize = 256;
};

#endif // SETTINGS_H
