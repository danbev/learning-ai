cmake_minimum_required(VERSION 3.4...3.12.3)
project(pybind_example)

set(CMAKE_BUILD_TYPE Debug)

add_subdirectory(pybind11)

pybind11_add_module(pybind_example src/module.cpp)
