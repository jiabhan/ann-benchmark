//
// Created by 付聪 on 2017/6/21.
//

#ifndef EFANNA2E_UTIL_H
#define EFANNA2E_UTIL_H
#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
#ifdef __APPLE__
#else
#include <malloc.h>
#endif
namespace efanna2e {

static void GenRandom(std::mt19937& rng, unsigned* addr, unsigned size, unsigned N)
{
    for (unsigned i = 0; i < size; ++i) {
        addr[i] = rng() % (N - size);
    }
    std::sort(addr, addr + size);
    for (unsigned i = 1; i < size; ++i) {
        if (addr[i] <= addr[i - 1]) {
            addr[i] = addr[i - 1] + 1;
        }
    }
    unsigned off = rng() % N;
    for (unsigned i = 0; i < size; ++i) {
        addr[i] = (addr[i] + off) % N;
    }
}

inline float* data_align(float* data_ori, unsigned point_num, unsigned& dim)
{
#ifdef __GNUC__
#ifdef __AVX__
#define DATA_ALIGN_FACTOR 8
#else
#ifdef __SSE2__
#define DATA_ALIGN_FACTOR 4
#else
#define DATA_ALIGN_FACTOR 1
#endif
#endif
#endif

    // Calculate new dimension with overflow check
    unsigned new_dim = (dim + DATA_ALIGN_FACTOR - 1) / DATA_ALIGN_FACTOR * DATA_ALIGN_FACTOR;

    // Check for potential overflow in total size calculation
    size_t total_size;
    if (__builtin_mul_overflow((size_t)point_num, (size_t)new_dim, &total_size)) {
        throw std::runtime_error("Data size calculation overflow");
    }

    // Log allocation size
    std::cout << "Allocating aligned memory:" << std::endl
              << "  Points: " << point_num << std::endl
              << "  Original dim: " << dim << std::endl
              << "  Aligned dim: " << new_dim << std::endl
              << "  Total size: " << (total_size * sizeof(float) / 1024.0 / 1024.0) << " MB" << std::endl;

    float* data_new = nullptr;
    try {
#ifdef __APPLE__
        data_new = new float[total_size];
#else
        // Try aligned allocation with error checking
        if ((data_new = (float*)aligned_alloc(DATA_ALIGN_FACTOR * sizeof(float),
                 total_size * sizeof(float)))
            == nullptr) {
            throw std::bad_alloc();
        }
#endif

        // Process data in chunks to avoid large contiguous memory operations
        const size_t chunk_size = 1000; // Process 1000 points at a time
        for (size_t i = 0; i < point_num; i += chunk_size) {
            size_t current_chunk = std::min(chunk_size, point_num - i);

#pragma omp parallel for if (current_chunk > 100)
            for (size_t j = 0; j < current_chunk; j++) {
                // Copy original dimensions
                std::memcpy(data_new + (i + j) * new_dim,
                    data_ori + (i + j) * dim,
                    dim * sizeof(float));

                // Zero padding
                std::memset(data_new + (i + j) * new_dim + dim,
                    0,
                    (new_dim - dim) * sizeof(float));
            }
        }

    } catch (const std::exception& e) {
        if (data_new != nullptr) {
#ifdef __APPLE__
            delete[] data_new;
#else
            free(data_new);
#endif
        }
        throw std::runtime_error(std::string("Data alignment failed: ") + e.what());
    }

    // Free original data correctly based on allocation method
#ifdef __APPLE__
    delete[] data_ori;
#else
    delete[] data_ori; // Corrected from free(data_ori);
#endif

    dim = new_dim;
    return data_new;
}

}

#endif // EFANNA2E_UTIL_H
