set(CONFIGURE_INCLUDE "${CMAKE_BINARY_DIR}/configured_files/include")

add_executable(sift-hnswlib-test "sift-hnswlib-test.cpp")
target_include_directories(sift-hnswlib-test PUBLIC ${CONFIGURE_INCLUDE})


# Specify the directories for NGT include files
include_directories("${PROJECT_BINARY_DIR}/lib/")
include_directories("${PROJECT_SOURCE_DIR}/lib")


# Link the target against the libraries and frameworks
target_link_libraries(sift-hnswlib-test
        hnswlib::hnswlib
)