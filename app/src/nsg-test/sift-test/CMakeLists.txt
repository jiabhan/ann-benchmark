add_executable(sift-test "sift-test.cpp")

# Specify the directories for include files
include_directories("${PROJECT_BINARY_DIR}/lib/NSG")
include_directories("${PROJECT_BINARY_DIR}/lib/NSG/src")
include_directories("${PROJECT_BINARY_DIR}/configured_files/include")
include_directories("${PROJECT_SOURCE_DIR}/lib")

# Link the target against the libraries and frameworks
target_link_libraries(sift-test
        efanna2e
)
