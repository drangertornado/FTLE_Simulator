#ifndef MATERIAL_H
#define MATERIAL_H

#pragma once

#include "glm.hpp"

struct Material
{
    float ambientFactor;
    float diffuseFactor;
    float specularFactor;
    float specularExponent;

    __host__ Material() : ambientFactor(0.0f),
                          diffuseFactor(0.0f),
                          specularFactor(0.0f),
                          specularExponent(0.0f) {}

    __host__ Material(float ambientFactor,
                      float diffuseFactor,
                      float specularFactor,
                      float specularExponent) : ambientFactor(ambientFactor),
                                                diffuseFactor(diffuseFactor),
                                                specularFactor(specularFactor),
                                                specularExponent(specularExponent) {}
};

#endif // MATERIAL_H
