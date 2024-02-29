#ifndef MAIN
#define MAIN

#pragma once

#include <iostream>
#include "CUDACheck.h"
#include "Grid.h"
#include "Renderer.h"
#include "Window.h"
#include "Settings.h"

// Define the static variable
Window* Window::windowInstance = nullptr;

void initializeCUDA()
{
    CUDA_CHECK(cudaFree(0));
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
    {
        throw std::runtime_error("No CUDA capable devices found!");
    }
        
    std::cout << "Found " << numDevices << " CUDA devices" << std::endl;
    int deviceId = 0;
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, deviceId);
    // Set the primary GPU device
    CUDA_CHECK(cudaSetDevice(0));
    std::cout << "Running on device: " << deviceProps.name << std::endl;
}

int main()
{
    initializeCUDA();

    // Initialize settings
    Settings settings;

    // Initialize the 3D Grid
    Grid grid(&settings);
    grid.computeFTLE(settings.integrationStartTime, settings.integrationDuration);

    // Initialize Renderer
    Renderer renderer(&grid, &settings);

    // Create a window
    Window window(&grid, &renderer, &settings);
    window.create("FTLE Simulator");

    // Print the FTLE values for testing
    // grid.downloadAndPrintFTLEPoints();

    // std::cout << "\nPress enter to exit..."; 
    // std::cin.get();

    return 0;
}

#endif // MAIN
