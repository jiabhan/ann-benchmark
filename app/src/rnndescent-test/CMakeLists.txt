# Find OpenMP
find_package(OpenMP REQUIRED)

# Add the RNN-Descent library as a subdirectory
include_directories("${PROJECT_BINARY_DIR}/lib/RNNDescent")
include_directories("${PROJECT_SOURCE_DIR}/lib")

# Create benchmark executable
add_executable(rnndescent-all-test rnndescent-all-test.cpp)

target_link_libraries(rnndescent-all-test 
    rnndescent
    OpenMP::OpenMP_CXX
)

set(LIBRARY_INCLUDES "./" "${CMAKE_BINARY_DIR}/configured_files/include")
target_include_directories(rnndescent-all-test PUBLIC ${LIBRARY_INCLUDES})
