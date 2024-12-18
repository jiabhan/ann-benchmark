cmake_minimum_required(VERSION 3.10)
project(suco CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compiler flags
add_compile_options(-O3 -fpic -march=native -mavx -mavx2 -msse3)

# Find required packages
find_package(Armadillo REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization)
find_package(OpenMP REQUIRED)
find_package(MLPACK REQUIRED)
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)

# Collect source files
file(GLOB SUCO_SOURCES "src/*.cpp")

# Create static library instead of executable
add_library(suco STATIC ${SUCO_SOURCES})

# Set include directories for the library
target_include_directories(suco
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        PRIVATE
        ${ARMADILLO_INCLUDE_DIRS}
        ${MLPACK_INCLUDE_DIRS}
)

# Link libraries to the static library
target_link_libraries(suco
        PUBLIC
        armadillo
        mlpack
        ${LAPACK_LIBRARIES}
        ${BLAS_LIBRARIES}
        Boost::serialization
        OpenMP::OpenMP_CXX
)

# Optional: If you still want to build the executable
add_executable(suco_exec ${SUCO_SOURCES})
target_link_libraries(suco_exec PRIVATE suco)