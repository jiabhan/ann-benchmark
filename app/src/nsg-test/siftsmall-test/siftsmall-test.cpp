#include "config.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <efanna2e/index_graph.h>
#include <efanna2e/index_nsg.h>
#include <efanna2e/index_random.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

std::vector<std::vector<float>> readFvecs(const std::string& filename, unsigned& dimension)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    std::vector<std::vector<float>> data;
    while (file.read(reinterpret_cast<char*>(&dimension), sizeof(dimension))) {
        std::vector<float> vec(dimension);
        if (!file.read(reinterpret_cast<char*>(vec.data()), dimension * sizeof(float))) {
            break;
        }
        data.push_back(vec);
    }
    return data;
}

std::vector<std::vector<int>> readIvecs(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    std::vector<std::vector<int>> data;
    int32_t dim;
    while (file.read(reinterpret_cast<char*>(&dim), sizeof(dim))) {
        std::vector<int> vec(dim);
        if (!file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(int))) {
            break;
        }
        data.push_back(vec);
    }
    return data;
}

void load_data(const std::string& filename, float*& data, unsigned& num, unsigned& dim)
{
    try {
        std::ifstream in(filename, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        // Read dimension
        if (!in.read(reinterpret_cast<char*>(&dim), sizeof(dim))) {
            throw std::runtime_error("Failed to read dimension from file");
        }
        std::cout << "Data dimension: " << dim << std::endl;

        // Get file size and calculate number of elements
        in.seekg(0, std::ios::end);
        std::ios::pos_type ss = in.tellg();
        size_t fsize = static_cast<size_t>(ss);
        num = static_cast<unsigned>(fsize / (dim + 1) / 4);

        // Calculate total size needed and check for overflow
        size_t total_floats = static_cast<size_t>(num) * static_cast<size_t>(dim);
        if (total_floats / dim != num) { // Check for multiplication overflow
            throw std::runtime_error("Data size too large, would cause overflow");
        }

        // Allocate memory
        data = nullptr;
        data = new float[total_floats];
        if (!data) {
            throw std::bad_alloc();
        }

        // Read data
        in.seekg(0, std::ios::beg);
        for (size_t i = 0; i < num; i++) {
            in.seekg(4, std::ios::cur); // Skip leading 4 bytes
            if (!in.read(reinterpret_cast<char*>(data + i * dim), dim * sizeof(float))) {
                delete[] data;
                data = nullptr;
                throw std::runtime_error("Failed to read data at element " + std::to_string(i));
            }
        }

        in.close();
    } catch (...) {
        delete[] data;
        data = nullptr;
        throw;
    }
}

void save_result(const std::string& filename,
    double total_time,
    double qps,
    float recall,
    unsigned K,
    unsigned L)
{
    // Create directory if it doesn't exist
    std::filesystem::path filepath(filename);
    std::filesystem::create_directories(filepath.parent_path());

    // Determine if we need to write header (if file doesn't exist)
    bool write_header = !std::filesystem::exists(filepath);

    std::ofstream out(filepath, std::ios::app);
    if (!out) {
        throw std::runtime_error("Cannot open output file: " + filename);
    }

    // Write CSV header if file is new
    if (write_header) {
        out << "K,L,recall,qps\n";
    }

    // Write stats in CSV format
    out << K << ","
        << L << ","
        << recall << ","
        << qps << "\n";

    out.close();
}

void createNNGraph(float* data_load, unsigned& num_of_elements, unsigned& dimension, const char* file_name)
{
    unsigned K = 200;
    unsigned L = 300;
    unsigned iter = 10;
    unsigned S = 10;
    unsigned R = 100;

    efanna2e::IndexRandom init_index(dimension, num_of_elements);
    efanna2e::IndexGraph index(dimension, num_of_elements, efanna2e::L2, (efanna2e::Index*)(&init_index));

    efanna2e::Parameters parameters;
    parameters.Set<unsigned>("K", K);
    parameters.Set<unsigned>("L", L);
    parameters.Set<unsigned>("iter", iter);
    parameters.Set<unsigned>("S", S);
    parameters.Set<unsigned>("R", R);

    auto s = std::chrono::high_resolution_clock::now();
    index.Build(num_of_elements, data_load, parameters);
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "Time cost: " << diff.count() << "\n";

    index.Save(file_name);
}

void createNSGIndex(float* data_load, unsigned& num_of_elements, unsigned& dimension, const char* graph_file_path, const char* index_file_path)
{
    unsigned L = 40;
    unsigned R = 50;
    unsigned C = 500;

    efanna2e::IndexNSG index(dimension, num_of_elements, efanna2e::L2, nullptr);

    auto s = std::chrono::high_resolution_clock::now();
    efanna2e::Parameters parameters;
    parameters.Set<unsigned>("L", L);
    parameters.Set<unsigned>("R", R);
    parameters.Set<unsigned>("C", C);
    parameters.Set<std::string>("nn_graph_path", graph_file_path);

    index.Build(num_of_elements, data_load, parameters);
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;

    std::cout << "indexing time: " << diff.count() << "\n";
    index.Save(index_file_path);
}

float calculateRecall(const std::vector<std::vector<unsigned>>& results,
    const std::vector<std::vector<int>>& ground_truth,
    size_t K)
{
    size_t num_queries = results.size();
    size_t correct = 0;
    size_t total = num_queries * K;

    for (size_t i = 0; i < num_queries; i++) {
        std::vector<unsigned> result_set(results[i].begin(), results[i].begin() + K);
        std::vector<int> gt_set(ground_truth[i].begin(), ground_truth[i].begin() + K);

        for (const auto& res : result_set) {
            for (const auto& gt : gt_set) {
                if (res == gt) {
                    correct++;
                    break;
                }
            }
        }
    }

    return float(correct) / total;
}

void searchNSGIndex(float* data_load, unsigned& num_of_elements, unsigned& dimension,
    const char* query_file_path, const char* index_file_path,
    const char* ground_truth_path, const char* result_file_path)
{
    unsigned query_dimension;
    unsigned num_of_query;
    float* query_load = nullptr;

    load_data(query_file_path, query_load, num_of_query, query_dimension);
    assert(dimension == query_dimension);

    // Load ground truth
    std::vector<std::vector<int>> ground_truth = readIvecs(ground_truth_path);

    // Define multiple L values to test
    std::vector<unsigned> L_values = { 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200 };
    unsigned K = 10;

    efanna2e::IndexNSG index(dimension, num_of_elements, efanna2e::L2, nullptr);
    index.Load(index_file_path);

    // Test each L value
    for (unsigned L : L_values) {
        if (L < K) {
            std::cout << "Skipping L=" << L << " as it's smaller than K=" << K << std::endl;
            continue;
        }

        std::cout << "\nTesting with L=" << L << std::endl;

        efanna2e::Parameters parameters;
        parameters.Set<unsigned>("L_search", L);
        parameters.Set<unsigned>("P_search", L);

        std::vector<std::vector<unsigned>> res;

        // Measure search time
        auto s = std::chrono::high_resolution_clock::now();

        for (unsigned i = 0; i < num_of_query; i++) {
            std::vector<unsigned> tmp(K);
            index.Search(query_load + i * dimension, data_load, K, parameters, tmp.data());
            res.push_back(tmp);
        }

        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> search_time = e - s;

        // Calculate metrics
        double total_time = search_time.count();
        double qps = num_of_query / total_time;
        float recall = calculateRecall(res, ground_truth, K);

        // Print results
        std::cout << "Search Statistics for L=" << L << ":" << std::endl;
        std::cout << "QPS: " << qps << " queries/second" << std::endl;
        std::cout << "Recall@" << K << ": " << recall << std::endl;

        // Save results and statistics
        save_result(std::string(result_file_path), total_time, qps, recall, K, L);
    }

    delete[] query_load;
}

// Add this function to check if file exists
bool fileExists(const std::string& filename)
{
    std::ifstream f(filename.c_str());
    return f.good();
}

int main()
{
    try {
        std::string index_data_file = std::string(data_dir) + "siftsmall/siftsmall_base.fvecs";
        std::string query_data_file = std::string(data_dir) + "siftsmall/siftsmall_query.fvecs";
        std::string ground_truth_file = std::string(data_dir) + "siftsmall/siftsmall_groundtruth.ivecs";
        std::string graph_file_path = std::string(graph_dir) + "nsg-test/siftsmall-test/sift_200nn.graph";
        std::string index_path = std::string(index_dir) + "nsg-test/siftsmall-test/";
        std::string index_file_path = index_path + "siftsmall.nsg";
        std::string result_file_path = std::string(result_dir) + "nsg-test/siftsmall-test/siftsmall_recall_qps_result.csv";

        // Create directories if they don't exist
        std::filesystem::create_directories(index_path);
        std::filesystem::create_directories(std::filesystem::path(result_file_path).parent_path());

        unsigned dimension = 0;
        unsigned num_of_elements = 0;
        float* data_load = nullptr;

        // Load and align data with RAII
        std::cout << "Loading data..." << std::endl;
        load_data(index_data_file, data_load, num_of_elements, dimension);

        if (!data_load) {
            throw std::runtime_error("Failed to load data");
        }

        // Use smart pointer for aligned data
        std::unique_ptr<float[]> aligned_data(efanna2e::data_align(data_load, num_of_elements, dimension));
        data_load = nullptr; // Original data_load is now managed by aligned_data

        if (!aligned_data) {
            throw std::runtime_error("Failed to align data");
        }

        // Create NN graph if needed
        if (!fileExists(graph_file_path)) {
            std::cout << "Creating NN graph..." << std::endl;
            createNNGraph(aligned_data.get(), num_of_elements, dimension, graph_file_path.c_str());
        } else {
            std::cout << "NN graph already exists at: " << graph_file_path << std::endl;
        }

        // Create NSG index if needed
        if (!fileExists(index_file_path)) {
            std::cout << "Creating NSG index..." << std::endl;
            createNSGIndex(aligned_data.get(), num_of_elements, dimension,
                graph_file_path.c_str(), index_file_path.c_str());
        } else {
            std::cout << "NSG index already exists at: " << index_file_path << std::endl;
        }

        // Search
        std::cout << "Performing search..." << std::endl;
        searchNSGIndex(aligned_data.get(), num_of_elements, dimension,
            query_data_file.c_str(), index_file_path.c_str(),
            ground_truth_file.c_str(), result_file_path.c_str());

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }
}