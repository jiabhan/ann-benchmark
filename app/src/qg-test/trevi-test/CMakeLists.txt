# First, explicitly set the BLAS vendor to ensure we're using a specific implementation
set(BLA_VENDOR OpenBLAS)

# Find required packages
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Add some diagnostic output to verify the variables are set
message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
message(STATUS "LAPACK libraries: ${LAPACK_LIBRARIES}")

add_executable(trevi-qg-test "trevi-qg-test.cpp")

# Specify the directories for NGT include files
include_directories("${PROJECT_BINARY_DIR}/lib/NGT/lib/")
include_directories("${PROJECT_SOURCE_DIR}/lib/NGT/lib/")
include_directories("${PROJECT_BINARY_DIR}/configured_files/include")

# Link the target against the libraries and frameworks
# Note the explicit addition of -lblas and -llapack
target_link_libraries(trevi-qg-test
        PRIVATE
        ngt
        ${OpenMP_CXX_LIBRARIES}
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        blas
        lapack
)