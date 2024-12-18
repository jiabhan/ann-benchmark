# src/CMakeLists.txt

# First, let's gather all our source files
file(GLOB_RECURSE CPP_SOURCES *.cpp)

# Create a single library target
add_library(efanna2e STATIC ${CPP_SOURCES})

# Configure the include directories for our library
target_include_directories(efanna2e
        PUBLIC
        ${PROJECT_SOURCE_DIR}/include  # For public headers
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}    # For internal headers
        ${Boost_INCLUDE_DIRS}          # Boost headers
)

# Link against our dependencies
target_link_libraries(efanna2e
        PUBLIC
        OpenMP::OpenMP_CXX
        Boost::boost
)

# If you need specific compiler options for just this library
target_compile_options(efanna2e
        PRIVATE
        -O3
        -march=native
)

# If you want to create an alias target for use in other parts of your project
add_library(efanna2e::efanna2e ALIAS efanna2e)

# Optional: Set up installation rules
install(TARGETS efanna2e
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
)
