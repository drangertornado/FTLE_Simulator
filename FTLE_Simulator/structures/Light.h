#ifndef LIGHT_H
#define LIGHT_H

#pragma once

#include "glm.hpp"

struct PointLight
{
    glm::vec3 color;
    glm::vec3 position;
};

struct DirectionalLight
{
    glm::vec3 color;
    glm::vec3 direction;
};

struct SpotLight
{
    glm::vec3 color;
    glm::vec3 position;
    glm::vec3 direction;
    float cutOff;
    float outerCutoff;
};

#endif // LIGHT_H
