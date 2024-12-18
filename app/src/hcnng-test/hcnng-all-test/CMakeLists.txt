add_executable(hcnng-all-test "hcnng-all-test.cpp")

# Specify the directories for NGT include files
include_directories("${PROJECT_BINARY_DIR}/lib/ParlayANN")
include_directories("${PROJECT_SOURCE_DIR}/lib")

set(LIBRARY_INCLUDES "./" "${CMAKE_BINARY_DIR}/configured_files/include")
target_include_directories(hcnng-all-test PUBLIC ${LIBRARY_INCLUDES})

# Link the target against the libraries and frameworks
target_link_libraries(hcnng-all-test
        ParlayANN
        parlay
)
