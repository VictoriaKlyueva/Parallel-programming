cmake_minimum_required(VERSION 3.27)
project(parallel)

set(CMAKE_CXX_STANDARD 17)

OPTION (USE_OpenMP "Use OpenMP" ON)
IF(USE_OpenMP)
    FIND_PACKAGE(OpenMP)
    IF(OPENMP_FOUND)
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    ENDIF()
ENDIF()

if (${EIGEN3_FOUND})
    include_directories(${EIGEN3_INCLUDE_DIR})
else()
    message(STATUS "Eigen3 not found")
endif()

add_executable(parallel main.cpp)
