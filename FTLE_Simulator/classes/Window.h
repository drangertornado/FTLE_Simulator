#ifndef WINDOW_H
#define WINDOW_H

#pragma once

#include <vector>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "Shader.h"
#include "Grid.h"
#include "Renderer.h"
#include "Camera.h"
#include "Settings.h"

class Window
{
public:
    Window(Grid *grid, Renderer *renderer, Settings *settings);
    ~Window();

    void create(const char *windowTitle = "Window");

private:
    void processInput();
    void render();
    void draw();

    static Window *windowInstance;
    static void mouse_callback(GLFWwindow* glfwWindow, double inputPositionX, double inputPositionY);
    static void scroll_callback(GLFWwindow *glfwWindow, double offsetX, double offsetY);
    static void key_callback(GLFWwindow *glfwWindow, int key, int scancode, int action, int mods);

    GLFWwindow *glfwWindow = nullptr;
    Grid *grid = nullptr;
    Renderer *renderer = nullptr;
    Camera *camera = nullptr;
    Settings *settings = nullptr;

    float deltaTime = 0.0f;
    float lastFrame = 0.0f;
    glm::vec2 previousMousePosition;

    unsigned int VAO, VBO, EBO;
    unsigned int texture;

    const float vertices[20] = {
        // Positions and texture coords
        1.0f, 1.0f, 0.0f, 1.0f, 1.0f,   // Top right
        1.0f, -1.0f, 0.0f, 1.0f, 0.0f,  // Bottom right
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, // Bottom left
        -1.0f, 1.0f, 0.0f, 0.0f, 1.0f   // Top left
    };

    const unsigned int indices[6] = {
        0, 1, 3, // First triangle
        1, 2, 3  // Second triangle
    };

    Shader textureShader;

    const char *vertexShader = "#version 330 core\n"
                               "layout (location = 0) in vec3 aPosition;\n"
                               "layout (location = 1) in vec2 aTextureCoord;\n"
                               "out vec2 textureCoord;\n"
                               "void main()\n"
                               "{\n"
                               "   gl_Position = vec4(aPosition.x, aPosition.y, aPosition.z, 1.0);\n"
                               "   textureCoord = aTextureCoord;\n"
                               "}\0";

    const char *fragmentShader = "#version 330 core\n"
                                 "out vec4 fragColor;\n"
                                 "in vec2 textureCoord;\n"
                                 "uniform sampler2D uTextureSampler;"
                                 "void main()\n"
                                 "{\n"
                                 //  "   fragColor = vec4(color, 1.0);\n"
                                 "   fragColor = texture(uTextureSampler, textureCoord);\n"
                                 "}\n\0";
};

#endif // WINDOW_H
