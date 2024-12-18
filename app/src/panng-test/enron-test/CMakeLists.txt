add_executable(enron-panng-test "enron-panng-test.cpp")
target_link_libraries(enron-panng-test ngtstatic)

# Specify the directories for NGT include files
include_directories("${PROJECT_BINARY_DIR}/lib/NGT/lib/")
include_directories("${PROJECT_SOURCE_DIR}/lib/NGT/lib/")
include_directories("${PROJECT_BINARY_DIR}/configured_files/include")

# Link the target against the libraries and frameworks
target_link_libraries(enron-panng-test
        ngtstatic
)