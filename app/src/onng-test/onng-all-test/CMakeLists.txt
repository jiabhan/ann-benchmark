add_executable(onng-all-test "onng-all-test.cpp")

# Specify the directories for NGT include files
include_directories("${PROJECT_BINARY_DIR}/lib/NGT/lib/")
include_directories("${PROJECT_SOURCE_DIR}/lib/NGT/lib/")
include_directories("${PROJECT_BINARY_DIR}/configured_files/include")

# Link the target against the libraries and frameworks
target_link_libraries(onng-all-test
        PRIVATE
        ngt
)
