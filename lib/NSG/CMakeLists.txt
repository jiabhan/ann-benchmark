cmake_minimum_required(VERSION 3.15)

# Set policy for modern Boost finding behavior
cmake_policy(SET CMP0167 NEW)

project(efanna2e VERSION 1.0.0 LANGUAGES CXX)

# Note: We're NOT creating the library target here anymore
# Instead, we'll define project-wide settings

# Setting C++ standard for the entire project
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find dependencies at the project level
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED)

# These compiler flags will be inherited by targets in subdirectories
add_compile_options(
        -O3
        -march=native
        -Wall
        -fno-builtin-malloc
        -fno-builtin-calloc
        -fno-builtin-realloc
        -fno-builtin-free
)

# Add definitions that should apply project-wide
add_definitions(-DINFO)

# Now we add subdirectories - the actual library will be created in src
add_subdirectory(src)
add_subdirectory(tests)

