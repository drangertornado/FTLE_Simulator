#ifndef WINDOW_CPP
#define WINDOW_CPP

#include "Window.h"
#include <iostream>

// Constructor
Window::Window(Grid *grid, Renderer *renderer, Settings *settings) : grid(grid),
                                                                     renderer(renderer),
                                                                     camera(renderer->getCamera()),
                                                                     settings(settings)
{
    // Set current window instance
    windowInstance = this;

    // Initialize GLFWindow
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    // Place mouse at the center of the screen
    previousMousePosition = glm::vec2(settings->screenWidth / 2.0f, settings->screenHeight / 2.0f);
}

// Destructor
Window::~Window()
{
    // Terminate GLFWindow
    glfwTerminate();
}

// Function to create the GLFW window
void Window::create(const char *title)
{
    std::cout << "\nCreating window: " << settings->screenWidth << " x " << settings->screenHeight << std::endl;

    // Create GLFW Window
    if (settings->fullScreen)
    {
        glfwWindow = glfwCreateWindow(settings->screenWidth, settings->screenHeight, title, glfwGetPrimaryMonitor(), nullptr);
    }
    else
    {
        glfwWindow = glfwCreateWindow(settings->screenWidth, settings->screenHeight, title, nullptr, nullptr);
    }

    if (glfwWindow == nullptr)
    {
        std::cout << "\nFailed to create GLFW window" << std::endl;
        glfwTerminate();
    }
    glfwMakeContextCurrent(glfwWindow);

    glfwSetInputMode(glfwWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPosCallback(glfwWindow, mouse_callback);
    glfwSetScrollCallback(glfwWindow, scroll_callback);

    // Check if glad library is initialized
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "\nFailed to initialize GLAD" << std::endl;
        glfwTerminate();
    }

    // Compile shaders
    textureShader.compile(vertexShader, fragmentShader);

    // Initialize viewport
    glViewport(0, 0, settings->screenWidth, settings->screenHeight);

    // Initialize buffers
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    // Texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Generate texture buffer
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    // Set texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    // Set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    textureShader.use();
    textureShader.setInt("uTextureSampler", 0);

    // Render loop
    while (!glfwWindowShouldClose(glfwWindow))
    {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // Process input
        processInput();

        // Render
        render();

        // Draw
        draw();

        // Swap buffers and poll IO events
        glfwSwapBuffers(glfwWindow);
        glfwPollEvents();
    }
}

void Window::processInput()
{
    if (glfwGetKey(glfwWindow, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(glfwWindow, true);
    }

    if (glfwGetKey(glfwWindow, GLFW_KEY_W) == GLFW_PRESS)
    {
        camera->moveForward(settings->movementSpeed * deltaTime);
    }

    if (glfwGetKey(glfwWindow, GLFW_KEY_S) == GLFW_PRESS)
    {
        camera->moveBackward(settings->movementSpeed * deltaTime);
    }

    if (glfwGetKey(glfwWindow, GLFW_KEY_A) == GLFW_PRESS)
    {
        camera->moveLeft(settings->movementSpeed * deltaTime);
    }

    if (glfwGetKey(glfwWindow, GLFW_KEY_D) == GLFW_PRESS)
    {
        camera->moveRight(settings->movementSpeed * deltaTime);
    }

    if (glfwGetKey(glfwWindow, GLFW_KEY_Q) == GLFW_PRESS)
    {
        camera->moveUp(settings->movementSpeed * deltaTime);
    }

    if (glfwGetKey(glfwWindow, GLFW_KEY_E) == GLFW_PRESS)
    {
        camera->moveDown(settings->movementSpeed * deltaTime);
    }

    if (glfwGetKey(glfwWindow, GLFW_KEY_RIGHT) == GLFW_PRESS)
    {
        settings->integrationDuration = std::max(
            settings->integrationDuration + settings->integrationDurationStepSize, 0.0f);
        grid->computeFTLE(settings->integrationStartTime, settings->integrationDuration);
    }

    if (glfwGetKey(glfwWindow, GLFW_KEY_LEFT) == GLFW_PRESS)
    {
        settings->integrationDuration = std::max(
            settings->integrationDuration - settings->integrationDurationStepSize, 0.0f);
        grid->computeFTLE(settings->integrationStartTime, settings->integrationDuration);
    }

    if (glfwGetKey(glfwWindow, GLFW_KEY_UP) == GLFW_PRESS)
    {
        settings->integrationStartTime = std::max(
            settings->integrationStartTime + settings->integrationStartTimeStepSize, 0.0f);
        grid->computeFTLE(settings->integrationStartTime, settings->integrationDuration);
    }

    if (glfwGetKey(glfwWindow, GLFW_KEY_DOWN) == GLFW_PRESS)
    {
        settings->integrationStartTime = std::max(
            settings->integrationStartTime - settings->integrationStartTimeStepSize, 0.0f);
        grid->computeFTLE(settings->integrationStartTime, settings->integrationDuration);
    }
}

void Window::render()
{
    renderer->render();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, settings->screenWidth, settings->screenHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, renderer->getPixels());
    glGenerateMipmap(GL_TEXTURE_2D);
}

void Window::draw()
{
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Bind textures on corresponding texture units
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);

    // Render texture
    textureShader.use();
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

void Window::mouse_callback(GLFWwindow *glfwWindow, double inputPositionX, double inputPositionY)
{
    glm::vec2 currentMousePosition(static_cast<float>(inputPositionX), static_cast<float>(inputPositionY));
    glm::vec2 offset = (currentMousePosition - windowInstance->previousMousePosition);
    if (windowInstance->settings->invertMouseVertically)
    {
        offset *= glm::vec2(1.0f, -1.0f);
    }

    windowInstance->previousMousePosition = currentMousePosition;
    windowInstance->camera->rotate(windowInstance->settings->mouseSensitivity * offset);
}

void Window::scroll_callback(GLFWwindow *glfwWindow, double offsetX, double offsetY)
{
}

#endif // WINDOW_CPP
