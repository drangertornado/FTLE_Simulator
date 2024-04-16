#ifndef PHONG_LIGHTING_H
#define PHONG_LIGHTING_H

#pragma once

#include "glm.hpp"
#include "Light.h"
#include "Camera.h"

struct PhongLighting
{
    float ambientIntensity;
    float diffuseIntensity;
    float specularIntensity;
    float specularExponent;

    __host__ PhongLighting() : ambientIntensity(0.0f),
                               specularExponent(0.0f) {}

    __host__ PhongLighting(float ambientIntensity,
                           float diffuseIntensity,
                           float specularIntensity,
                           float specularExponent) : ambientIntensity(ambientIntensity),
                                                     diffuseIntensity(diffuseIntensity),
                                                     specularIntensity(specularIntensity),
                                                     specularExponent(specularExponent) {}

    __device__ glm::vec3 computeLighting(const glm::vec3 &position, const glm::vec3 &color, const glm::vec3 &normal, const PointLight &light, const Ray &ray)
    {
        glm::vec3 lightDirection = glm::normalize(light.position - position);

        // Combine ambient component
        glm::vec3 ambient = ambientIntensity * glm::vec3(color);

        // Compute diffuse component
        float diffuseFactor = abs(glm::dot(normal, lightDirection));
        glm::vec3 diffuse = diffuseIntensity * diffuseFactor * light.color * color;

        // Compute halfway vector
        glm::vec3 halfwayVector = glm::normalize(lightDirection - ray.direction);

        // Compute specular reflection
        float specularFactor = glm::pow(abs(glm::dot(normal, halfwayVector)), specularExponent);
        glm::vec3 specular = specularIntensity * specularFactor * light.color;

        return ambient + diffuse + specular;
    }
};

#endif // PHONG_LIGHTING_H
