#ifndef TRANSFER_FUNCTION_H
#define TRANSFER_FUNCTION_H

#pragma once

#include "glm.hpp"

struct TransferFunction
{
    float minimumValue;
    float maximumValue;
    float medianValue;
    glm::vec3 minimumColor;
    glm::vec3 maximumColor;
    float maximumOpacity;

    __host__ TransferFunction() : minimumValue(0.0f),
                                  maximumValue(0.0f),
                                  medianValue(0.0f),
                                  minimumColor(0.0f),
                                  maximumColor(0.0f),
                                  maximumOpacity(0.0f) {}

    __host__ TransferFunction(float minimumValue,
                              float maximumValue,
                              glm::vec3 minimumColor,
                              glm::vec3 maximumColor,
                              float maximumOpacity) : minimumValue(minimumValue),
                                                      maximumValue(maximumValue),
                                                      minimumColor(minimumColor),
                                                      maximumColor(maximumColor),
                                                      maximumOpacity(maximumOpacity)
    {
        medianValue = (minimumValue + maximumValue) * 0.5f;
    }

    // Compute color directly form interpolated scalar values
    __device__ glm::vec4 getColor(const float &scalarValue)
    {
        glm::vec4 interpolatedColor(0.0f);

        // Check if scalarValue is within range [scalarValueMin, scalarValueMax]
        if (scalarValue >= minimumValue && scalarValue <= maximumValue)
        {
            // Fractional distance of the scalar value from scalarValueMin
            float fractionalDistance = (scalarValue - minimumValue) / (maximumValue - minimumValue);

            // Interpolate color
            interpolatedColor = glm::mix(glm::vec4(minimumColor, 0.0f), glm::vec4(maximumColor, 0.0f), fractionalDistance);

            // Interpolate opacity using bump function
            float opacity;
            if (scalarValue <= medianValue)
            {
                float fractionalDistance = (scalarValue - minimumValue) / (medianValue - minimumValue);
                opacity = glm::mix(0.0f, maximumOpacity, glm::smoothstep(0.0f, 1.0f, fractionalDistance));
            }
            else
            {
                float fractionalDistance = (scalarValue - medianValue) / (maximumValue - medianValue);
                opacity = glm::mix(maximumOpacity, 0.0f, glm::smoothstep(0.0f, 1.0f, fractionalDistance));
            }

            interpolatedColor.a = opacity;
        }

        return interpolatedColor;
    }
};

#endif // TRANSFER_FUNCTION_H
