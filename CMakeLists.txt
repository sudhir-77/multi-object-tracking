cmake_minimum_required(VERSION 3.15)
project(tracker_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(tracker_cpp MODULE cpp/bindings.cpp)

if(MSVC)
  target_compile_options(tracker_cpp PRIVATE /O2 /fp:precise /EHsc)
else()
  target_compile_options(tracker_cpp PRIVATE -O3 -fno-fast-math)
endif()

install(TARGETS tracker_cpp DESTINATION .)
