cmake_minimum_required(VERSION 3.0)

if (POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif ()

add_library(glad src/glad.c)
