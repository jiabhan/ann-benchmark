set(CONFIGURE_INCLUDE "${CMAKE_BINARY_DIR}/configured_files/include")

add_executable(hnswlib-all-test "hnswlib-all-test.cpp")
target_include_directories(hnswlib-all-test PUBLIC ${CONFIGURE_INCLUDE})

# Specify the directories for NGT include files
include_directories("${PROJECT_BINARY_DIR}/lib/")
include_directories("${PROJECT_SOURCE_DIR}/lib")


# Link the target against the libraries and frameworks
target_link_libraries(hnswlib-all-test
        hnswlib::hnswlib
)