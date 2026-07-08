cmake_minimum_required(VERSION 3.10)
project(GaussianFilter)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

target_compile_options(test_gaussian PRIVATE -Wall -Wextra -O2 -g)