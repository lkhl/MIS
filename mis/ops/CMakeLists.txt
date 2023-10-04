cmake_minimum_required(VERSION 3.5...3.26)
project(_C)

set(CMAKE_CXX_FLAGS "-O3 -fopenmp -m64")
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

add_subdirectory(pybind11)
pybind11_add_module(_C src/binding.cpp)
