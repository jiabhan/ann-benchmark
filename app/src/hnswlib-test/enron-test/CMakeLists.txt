set(CONFIGURE_INCLUDE "${CMAKE_BINARY_DIR}/configured_files/include")

add_executable(enron-hnswlib-test "enron-hnswlib-test.cpp")
target_include_directories(enron-hnswlib-test PUBLIC ${CONFIGURE_INCLUDE})


# Specify the directories for NGT include files
include_directories("${PROJECT_BINARY_DIR}/lib/")
include_directories("${PROJECT_SOURCE_DIR}/lib")


# Link the target against the libraries and frameworks
target_link_libraries(enron-hnswlib-test
        hnswlib::hnswlib
)