cmake_minimum_required(VERSION 3.13)
project(Skeleton)

set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -pedantic")
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -Wno-unused-parameter -g")

find_package(OpenGL REQUIRED)

link_directories(${GTKMM_LIBRARY_DIRS})
include_directories(${GTKMM_INCLUDE_DIRS})

add_executable(Skeleton
        framework.h
        framework.cpp
        Skeleton.cpp)
        

target_link_libraries(Skeleton GL glut GLU GLEW)
