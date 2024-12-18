cmake_minimum_required(VERSION 3.15)
project(ParlayANN VERSION 1
        DESCRIPTION "ParlayANN is a library of approximate nearest neighbor search algorithms, along with a set of useful tools for designing such algorithms. It is written in C++ and uses parallel primitives from ParlayLib. Currently it includes implementations of the ANNS algorithms DiskANN, HNSW, HCNNG, and pyNNDescent."
        LANGUAGES CXX)

# Basic setup
include(CheckCXXCompilerFlag)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add parlaylib from local subdirectory
add_subdirectory(parlaylib EXCLUDE_FROM_ALL)

# Set module path
list(APPEND CMAKE_MODULE_PATH "${ParlayANN_SOURCE_DIR}/cmake")

# Create main library target
add_library(ParlayANN INTERFACE)

# Set include directories
set(ParlayANN_INCLUDE_DIR "${ParlayANN_SOURCE_DIR}/algorithms")
target_include_directories(ParlayANN INTERFACE
        $<BUILD_INTERFACE:${ParlayANN_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Link against parlay
target_link_libraries(ParlayANN INTERFACE parlay)

# Add subdirectories
add_subdirectory(algorithms)
#add_subdirectory(data_tools)  # Uncomment if needed
#add_subdirectory(range_search)  # Uncomment if needed

# Optional: Set installation rules
include(GNUInstallDirs)
install(TARGETS ParlayANN
        EXPORT ParlayANNTargets
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY ${ParlayANN_INCLUDE_DIR}/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ParlayANN
        FILES_MATCHING PATTERN "*.h")

# Optional: Export package for other projects to find
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
        "ParlayANNConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion)