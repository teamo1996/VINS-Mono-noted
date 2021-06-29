cmake_minimum_required(VERSION 3.19)
project(untitled)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV)

include_directories(/usr/local/include/eigen3
                    ${OpenCV_INCLUDE_DIRS})

add_executable(untitled main.cpp)

target_link_libraries(untitled ${OpenCV_LIBS})