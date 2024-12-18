# Build Instructions

## Using CMake

1. Create a build directory:
```bash
mkdir build && cd build
```

2. Configure with CMake:
```bash
cmake ..
```

3. Build:
```bash
cmake --build .
```

## Dependencies

Before building, ensure you have the following dependencies installed:

On Ubuntu/Debian:
```bash
sudo apt-get install liblapack-dev libblas-dev libboost-dev
sudo apt-get install libarmadillo-dev
sudo apt-get install libboost-math-dev libboost-program-options-dev libboost-random-dev libboost-test-dev libxml2-dev
sudo apt-get install libmlpack-dev
```

The executable will be created as `build/suco`.

This CMake configuration:

1. Sets up the project with C++17
2. Adds all the required compiler optimizations (-O3, -mavx, etc.)
3. Finds and links all required dependencies (Armadillo, MLPack, Boost, OpenMP)
4. Collects all source files from the src directory
5. Creates the suco executable and links all libraries
6. Sets up include directories

To use this:

1. Keep the existing Makefile for compatibility
2. Add the new CMakeLists.txt file to the root directory
3. Add the BUILD.md file with instructions
4. Update .gitignore to exclude CMake build files

Users can then choose to build either with Make or CMake. CMake provides better cross-platform support and integration with IDEs, while maintaining all the optimization flags and dependencies from the original Makefile.
