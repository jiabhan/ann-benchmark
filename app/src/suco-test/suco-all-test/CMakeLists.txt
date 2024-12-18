# Add at the top after cmake_minimum_required
cmake_policy(SET CMP0167 NEW)  # Address the Boost warning

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

# Find required packages
find_package(Boost REQUIRED COMPONENTS serialization)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(CONFIGURE_INCLUDE "${CMAKE_BINARY_DIR}/configured_files/include")

# Create test executable
add_executable(suco-all-test "suco-all-test.cpp")

# Add include directories
target_include_directories(suco-all-test
        PUBLIC
        ${CONFIGURE_INCLUDE}
        ${PROJECT_BINARY_DIR}/lib
        ${PROJECT_SOURCE_DIR}/lib
        ${Boost_INCLUDE_DIRS}
)

# Link with required libraries
target_link_libraries(suco-all-test
        PRIVATE
        suco
        armadillo
        mlpack
        ${LAPACK_LIBRARIES}
        ${BLAS_LIBRARIES}
        ${Boost_LIBRARIES}
)

# Add OpenMP if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(suco-all-test PRIVATE OpenMP::OpenMP_CXX)
endif()

# Add compiler flags
target_compile_options(suco-all-test
        PRIVATE
        -O3
        -fpic
        -march=native
        -mavx
        -mavx2
        -msse3
)