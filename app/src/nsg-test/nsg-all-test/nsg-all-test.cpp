#include "config.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <efanna2e/index_graph.h>
#include <efanna2e/index_nsg.h>
#include <efanna2e/index_random.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#ifdef __linux__
#include <sys/sysinfo.h>
#endif

const std::vector<std::string> DATASETS = {
    "audio", "cifar", "deep", "enron", "gist", "glove",
    "imagenet", "millionsong", "mnist", "notre", "nuswide",
    "sift", "sun", "trevi", "ukbench",
    "wikipedia-2024-06-bge-m3-zh"
};

struct NSGQueryParam {
    unsigned L_search; // Search parameter L
};

struct NSGParams {
    unsigned K; // kNN graph parameter
    unsigned L; // Build parameter L
    unsigned R_knn; // R parameter for kNN graph building
    unsigned R_nsg; // R parameter for NSG building
    unsigned C; // Candidate pool size
    unsigned S; // S parameter
    unsigned iter; // Number of iterations
    unsigned L_nsg; // L parameter for NSG building
    std::vector<NSGQueryParam> query_params;

    NSGParams(unsigned k, unsigned l, unsigned r_knn, unsigned r_nsg, unsigned c,
        const std::vector<unsigned>& l_search_values, unsigned s, unsigned it, unsigned l_nsg)
        : K(k)
        , L(l)
        , R_knn(r_knn)
        , R_nsg(r_nsg)
        , C(c)
        , S(s)
        , iter(it)
        , L_nsg(l_nsg)
    {
        for (auto l_search : l_search_values) {
            query_params.push_back({ l_search });
        }
    }
};

struct NSGDatasetParams {
    unsigned K, L, R_knn, R_nsg, C, S, iter, L_nsg;
};

const std::unordered_map<std::string, NSGDatasetParams> DATASET_PARAMS = {
    // Format: {K,   L,   R_knn, R_nsg, C,    S,  iter, L_nsg}
    { "sift", { 100, 120, 300, 30, 400, 25, 12, 150 } },
    { "gist", { 400, 430, 200, 20, 400, 10, 12, 500 } },
    { "glove", { 400, 420, 300, 90, 600, 20, 12, 150 } },
    { "crawl", { 400, 430, 300, 40, 600, 15, 12, 250 } },
    { "audio", { 200, 230, 100, 30, 600, 10, 5, 200 } },
    { "msong", { 300, 310, 300, 20, 500, 25, 12, 350 } },
    { "uqv", { 300, 320, 200, 30, 400, 15, 6, 350 } },
    { "enron", { 200, 200, 200, 60, 600, 25, 7, 150 } },
    { "nuswide", { 300, 400, 250, 80, 600, 25, 12, 350 } },
};

// Function to read fvecs correctly
std::vector<std::vector<float>> readFvecs(const std::string& filename, unsigned& dimension)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    std::vector<std::vector<float>> data;
    dimension = 0;
    size_t vector_count = 0;

    while (true) {
        int32_t current_dim;
        // Read the dimension header for each vector
        if (!file.read(reinterpret_cast<char*>(&current_dim), sizeof(current_dim))) {
            break; // EOF reached
        }

        // // Debug: Print the read dimension
        // std::cout << "Reading Vector " << vector_count << " dimension: " << current_dim << std::endl;

        if (current_dim <= 0) {
            throw std::runtime_error("Invalid vector dimension encountered: " + std::to_string(current_dim));
        }

        if (dimension == 0) {
            dimension = static_cast<unsigned>(current_dim);
            std::cout << "Setting initial dimension to: " << dimension << std::endl;
        } else if (current_dim != static_cast<int32_t>(dimension)) {
            throw std::runtime_error("Inconsistent vector dimensions at vector " + std::to_string(vector_count) + ". Expected: " + std::to_string(dimension) + ", Got: " + std::to_string(current_dim));
        }

        std::vector<float> vec(dimension);
        // Read the vector data
        if (!file.read(reinterpret_cast<char*>(vec.data()), sizeof(float) * dimension)) {
            throw std::runtime_error("Unexpected end of file while reading vector data.");
        }

        // Optional: Validate vector data (e.g., check for NaNs or infinities)
        for (unsigned i = 0; i < dimension; ++i) {
            if (!std::isfinite(vec[i])) {
                throw std::runtime_error("Non-finite value encountered in vector " + std::to_string(vector_count) + ", position " + std::to_string(i));
            }
        }

        data.emplace_back(std::move(vec));
        vector_count++;
    }

    std::cout << "Total vectors read: " << vector_count << std::endl;
    return data;
}

std::vector<std::vector<int32_t>> readIvecs(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    std::vector<std::vector<int32_t>> data;
    int32_t dimension = 0;

    while (true) {
        int32_t current_dim;
        if (!file.read(reinterpret_cast<char*>(&current_dim), sizeof(current_dim))) {
            break; // EOF reached
        }

        if (current_dim <= 0) {
            throw std::runtime_error("Invalid vector dimension encountered: " + std::to_string(current_dim));
        }

        if (dimension == 0) {
            dimension = current_dim;
        } else if (current_dim != dimension) {
            throw std::runtime_error("Inconsistent vector dimensions: expected " + std::to_string(dimension) + ", got " + std::to_string(current_dim));
        }

        std::vector<int32_t> vec(dimension);
        if (!file.read(reinterpret_cast<char*>(vec.data()), sizeof(int32_t) * dimension)) {
            throw std::runtime_error("Unexpected end of file while reading ivec data.");
        }

        data.emplace_back(std::move(vec));
    }

    return data;
}

// Function to validate fvecs file
bool validateDataFile(const std::string& filename, unsigned expected_dim = 0)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Error: Cannot open file: " << filename << std::endl;
        return false;
    }

    int32_t dim;
    size_t vector_count = 0;

    while (true) {
        // Read the dimension header for each vector
        if (!in.read(reinterpret_cast<char*>(&dim), sizeof(dim))) {
            break; // EOF reached
        }

        if (dim <= 0) {
            std::cerr << "Error: Invalid vector dimension encountered: " << dim << " at vector " << vector_count << std::endl;
            return false;
        }

        if (expected_dim != 0 && static_cast<unsigned>(dim) != expected_dim) {
            std::cerr << "Error: Dimension mismatch at vector " << vector_count
                      << ". Expected: " << expected_dim
                      << ", Got: " << dim << std::endl;
            return false;
        }

        // Skip the vector data
        in.seekg(sizeof(float) * dim, std::ios::cur);
        vector_count++;
    }

    std::cout << "Validation successful. Total vectors: " << vector_count << std::endl;
    return true;
}

void inspectFirstVectors(const std::string& filename, size_t num_vectors_to_inspect = 3)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open file for inspection: " << filename << std::endl;
        return;
    }

    for (size_t i = 0; i < num_vectors_to_inspect; ++i) {
        int32_t vector_dim;
        if (!in.read(reinterpret_cast<char*>(&vector_dim), sizeof(vector_dim))) {
            std::cerr << "Failed to read dimension of vector " << i << std::endl;
            break;
        }
        std::cout << "Vector " << i << " dimension: " << vector_dim << std::endl;

        // Read the first 5 floats of the vector for inspection
        size_t floats_to_inspect = std::min(static_cast<size_t>(5), static_cast<size_t>(vector_dim));
        std::vector<float> vec(floats_to_inspect);
        if (!in.read(reinterpret_cast<char*>(vec.data()), sizeof(float) * floats_to_inspect)) {
            std::cerr << "Failed to read data of vector " << i << std::endl;
            break;
        }

        std::cout << "First " << floats_to_inspect << " floats of vector " << i << ": ";
        for (size_t j = 0; j < floats_to_inspect; ++j) {
            std::cout << vec[j] << " ";
        }
        std::cout << std::endl;

        // Skip the remaining floats of the vector
        size_t remaining_floats = vector_dim > floats_to_inspect ? vector_dim - floats_to_inspect : 0;
        in.seekg(sizeof(float) * remaining_floats, std::ios::cur);
    }

    std::cout << "Inspection completed for first " << num_vectors_to_inspect << " vectors." << std::endl;
}

bool checkMemoryAvailability(size_t required_bytes)
{
#ifdef __linux__
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        unsigned long available_mem = si.freeram * si.mem_unit;
        std::cout << "Memory check:" << std::endl
                  << "  Required: " << (required_bytes / 1024.0 / 1024.0) << " MB" << std::endl
                  << "  Available: " << (available_mem / 1024.0 / 1024.0) << " MB" << std::endl;

        // return available_mem >= required_bytes * 1.5; // Require 50% more than needed
        return true;
    }
#endif
    return true; // Default to true on unsupported platforms
}

// Update load_data to use unique_ptr
void load_data(const std::string& filename, std::unique_ptr<float[]>& data, unsigned& num, unsigned& dim)
{
    std::cout << "Loading data from: " << filename << std::endl;

    // Validate the file before loading
    if (!validateDataFile(filename)) {
        throw std::runtime_error("Data file validation failed: " + filename);
    }

    // Read all vectors
    std::vector<std::vector<float>> vectors = readFvecs(filename, dim);
    num = static_cast<unsigned>(vectors.size());

    // Check memory requirements
    size_t required_memory = static_cast<size_t>(num) * static_cast<size_t>(dim) * sizeof(float);
    if (!checkMemoryAvailability(required_memory)) {
        throw std::runtime_error("Insufficient memory available for loading data.");
    }

    // Allocate memory using unique_ptr
    data.reset(new float[num * dim]);

    // Flatten the data into a contiguous array
    for (unsigned i = 0; i < num; ++i) {
        std::copy(vectors[i].begin(), vectors[i].end(), data.get() + i * dim);
    }

    std::cout << "Successfully loaded " << num << " vectors of dimension " << dim << std::endl;
}

void save_result(const std::string& filename, double total_time, double qps,
    float recall, unsigned K, unsigned L)
{
    static std::unordered_map<std::string, bool> file_initialized;

    std::filesystem::path filepath(filename);
    std::filesystem::create_directories(filepath.parent_path());

    // If this file hasn't been initialized in this run, truncate it
    // Otherwise append to it
    bool& initialized = file_initialized[filename];
    std::ios_base::openmode mode = initialized ? std::ios::app : std::ios::trunc;

    std::ofstream out(filepath, mode);
    if (!out) {
        throw std::runtime_error("Cannot open output file: " + filename);
    }

    // Write header only if we're creating a new file
    if (!initialized) {
        out << "recall,qps\n";
        initialized = true;
    }

    out << recall << "," << qps << "\n";
    out.close();
}

std::vector<NSGParams> getNSGParamSets(const std::string& dataset)
{
    std::vector<NSGParams> params;
    std::vector<unsigned> l_search_values = { 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200 };

    // Default parameters if dataset not found
    NSGDatasetParams p = { 200, 220, 150, 40, 500, 15, 12, 200 };

    // Override with dataset-specific parameters if available
    auto it = DATASET_PARAMS.find(dataset);
    if (it != DATASET_PARAMS.end()) {
        p = it->second;
    }

    params.emplace_back(p.K, p.L, p.R_knn, p.R_nsg, p.C, l_search_values, p.S, p.iter, p.L_nsg);
    return params;
}

void createNNGraph(float* data_load, unsigned num_of_elements,
    unsigned dimension, const char* file_name,
    const NSGParams& params)
{
    efanna2e::IndexRandom init_index(dimension, num_of_elements);
    efanna2e::IndexGraph index(dimension, num_of_elements, efanna2e::L2, (efanna2e::Index*)(&init_index));

    efanna2e::Parameters parameters;
    parameters.Set<unsigned>("K", params.K);
    parameters.Set<unsigned>("L", params.L);
    parameters.Set<unsigned>("iter", 12);
    parameters.Set<unsigned>("S", 25);
    parameters.Set<unsigned>("R", params.R_knn); // Using R_knn for graph building

    auto s = std::chrono::high_resolution_clock::now();
    index.Build(num_of_elements, data_load, parameters);
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "NN Graph build time: " << diff.count() << "s\n";

    index.Save(file_name);
}

void createNSGIndex(float* data_load, unsigned num_of_elements,
    unsigned dimension, const char* graph_file_path,
    const char* index_file_path, const NSGParams& params)
{
    std::cout << "Creating NSG index with:" << std::endl;
    std::cout << "num_of_elements: " << num_of_elements << std::endl;
    std::cout << "dimension: " << dimension << std::endl;
    std::cout << "data_load address: " << static_cast<void*>(data_load) << std::endl;

    try {
        // First verify the graph file exists and is readable
        std::ifstream graph_check(graph_file_path, std::ios::binary);
        if (!graph_check.good()) {
            throw std::runtime_error("Cannot read graph file");
        }
        graph_check.close();

        // Create and verify index object
        std::unique_ptr<efanna2e::IndexNSG> index;
        try {
            index = std::make_unique<efanna2e::IndexNSG>(dimension, num_of_elements, efanna2e::L2, nullptr);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to create IndexNSG object: ") + e.what());
        }

        // Setup parameters
        efanna2e::Parameters parameters;
        parameters.Set<unsigned>("L", params.L);
        parameters.Set<unsigned>("R", params.R_nsg);
        parameters.Set<unsigned>("C", params.C);
        parameters.Set<std::string>("nn_graph_path", graph_file_path);

        std::cout << "Starting index build..." << std::endl;
        std::cout << "Parameters:" << std::endl;
        std::cout << "L: " << params.L << std::endl;
        std::cout << "R: " << params.R_nsg << std::endl;
        std::cout << "C: " << params.C << std::endl;
        std::cout << "Graph path: " << graph_file_path << std::endl;

        // Start build with error checking
        try {
            index->Build(num_of_elements, data_load, parameters);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed during index build: ") + e.what());
        }

        std::cout << "Index build completed, saving..." << std::endl;

        try {
            index->Save(index_file_path);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to save index: ") + e.what());
        }

    } catch (const std::exception& e) {
        std::cerr << "Error in createNSGIndex: " << e.what() << std::endl;
        throw;
    }
}

float calculateRecall(const std::vector<std::vector<unsigned>>& results,
    const std::vector<std::vector<int32_t>>& ground_truth,
    size_t K)
{
    size_t num_queries = results.size();
    size_t correct = 0;
    size_t total = num_queries * K;

    for (size_t i = 0; i < num_queries; i++) {
        const std::vector<unsigned>& result_set = results[i];
        const std::vector<int32_t>& gt_set = ground_truth[i];

        for (const auto& res : result_set) {
            for (const auto& gt : gt_set) {
                if (static_cast<int32_t>(res) == gt) { // Cast res to int32_t for comparison
                    correct++;
                    break;
                }
            }
        }
    }

    return static_cast<float>(correct) / total;
}

void searchNSGIndex(float* aligned_data, unsigned num_of_elements,
    unsigned original_dim, unsigned aligned_dim,
    const char* query_file_path, const char* index_file_path,
    const char* ground_truth_path, const char* result_file_path,
    const NSGParams& params)
{
    std::cout << "\nStarting search with following parameters:" << std::endl;
    std::cout << "Original dimension: " << original_dim << std::endl;
    std::cout << "Aligned dimension: " << aligned_dim << std::endl;
    std::cout << "Number of base elements: " << num_of_elements << std::endl;

    unsigned query_dimension;
    unsigned num_of_query;
    std::unique_ptr<float[]> query_load;

    try {
        // Validate query file against original dimension
        if (!validateDataFile(query_file_path, original_dim)) {
            throw std::runtime_error("Query file validation failed");
        }

        // Load query data
        load_data(query_file_path, query_load, num_of_query, query_dimension);

        if (query_dimension != original_dim) {
            throw std::runtime_error("Dimension mismatch: base=" + std::to_string(original_dim) + ", query=" + std::to_string(query_dimension));
        }

        // Align query data
        unsigned aligned_query_dim = query_dimension;
        std::unique_ptr<float[]> aligned_query(efanna2e::data_align(query_load.release(), num_of_query, aligned_query_dim));

        if (!aligned_query) {
            throw std::runtime_error("Query data alignment failed");
        }

        if (aligned_query_dim != aligned_dim) {
            throw std::runtime_error("Aligned dimension mismatch: base=" + std::to_string(aligned_dim) + ", query=" + std::to_string(aligned_query_dim));
        }

        std::cout << "Successfully loaded and aligned " << num_of_query << " query vectors" << std::endl;

        // Load ground truth data
        std::vector<std::vector<int32_t>> ground_truth = readIvecs(ground_truth_path);
        if (ground_truth.empty()) {
            throw std::runtime_error("Failed to load ground truth data");
        }

        if (ground_truth.size() != num_of_query) {
            throw std::runtime_error("Ground truth size mismatch: expected=" + std::to_string(num_of_query) + ", got=" + std::to_string(ground_truth.size()));
        }

        // Initialize NSG index with aligned dimension
        unsigned K = 10; // Fixed K for search
        efanna2e::IndexNSG index(aligned_dim, num_of_elements, efanna2e::L2, nullptr);

        try {
            index.Load(index_file_path);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to load index: ") + e.what());
        }

        // Perform search for each query parameter
        for (const auto& qparam : params.query_params) {
            unsigned L = qparam.L_search;
            if (L < K) {
                std::cout << "Skipping L=" << L << " as it's smaller than K=" << K << std::endl;
                continue;
            }

            std::cout << "\nTesting with L=" << L << std::endl;

            efanna2e::Parameters parameters;
            parameters.Set<unsigned>("L_search", L);
            parameters.Set<unsigned>("P_search", L);

            std::vector<std::vector<unsigned>> res;
            res.reserve(num_of_query);

            // Start timing
            auto s = std::chrono::high_resolution_clock::now();

            // Search for each query
            for (unsigned i = 0; i < num_of_query; i++) {
                std::vector<unsigned> tmp(K);
                try {
                    // Updated to use aligned_data.get()
                    index.Search(aligned_query.get() + i * aligned_dim, aligned_data, K,
                        parameters, tmp.data());
                } catch (const std::exception& e) {
                    throw std::runtime_error(std::string("Search failed for query ") + std::to_string(i) + ": " + e.what());
                }
                res.emplace_back(std::move(tmp));
            }

            // End timing
            auto e = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> search_time = e - s;

            // Calculate metrics
            double total_time = search_time.count();
            double qps = num_of_query / total_time;
            float recall = calculateRecall(res, ground_truth, K);

            // Report results
            std::cout << "Search time: " << total_time << " seconds" << std::endl;
            std::cout << "QPS: " << qps << " queries/second" << std::endl;
            std::cout << "Recall@" << K << ": " << recall << std::endl;

            // Save results
            try {
                save_result(std::string(result_file_path), total_time, qps, recall, K, L);
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("Failed to save results: ") + e.what());
            }
        }

    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Search failed: ") + e.what());
    }
}

bool fileExists(const std::string& filename)
{
    std::ifstream f(filename.c_str());
    return f.good();
}
void processDataset(const std::string& dataset_name)
{
    std::cout << "\nProcessing dataset: " << dataset_name << std::endl;

    // Setup paths
    std::string base_path = std::string(data_dir) + dataset_name + "/";
    std::string base_file = base_path + dataset_name + "_base.fvecs";
    std::string query_file = base_path + dataset_name + "_query.fvecs";
    std::string groundtruth_file = base_path + dataset_name + "_groundtruth.ivecs";

    std::string graph_path = std::string(graph_dir) + "nsg-test/" + dataset_name;
    std::string index_path = std::string(index_dir) + "nsg-test/" + dataset_name;
    std::string result_path = std::string(result_dir) + "nsg-test/" + dataset_name;

    std::unique_ptr<float[]> data_load;
    std::unique_ptr<float[]> aligned_data;
    unsigned original_dim = 0, aligned_dim = 0, num_points = 0;

    try {
        // Create directories if they don't exist
        std::filesystem::create_directories(graph_path);
        std::filesystem::create_directories(index_path);
        std::filesystem::create_directories(result_path);

        std::string graph_file = graph_path + "/" + dataset_name + "_nn.graph";
        std::string index_file = index_path + "/" + dataset_name + ".nsg";
        std::string result_file = result_path + "/" + dataset_name + "_recall_qps_result.csv";

        // Verify input files exist
        if (!std::filesystem::exists(base_file) || !std::filesystem::exists(query_file) || !std::filesystem::exists(groundtruth_file)) {
            throw std::runtime_error("Missing required input files for dataset " + dataset_name);
        }

        // Inspect first few vectors in query file
        std::cout << "\nInspecting first vectors in query file for debugging..." << std::endl;
        inspectFirstVectors(query_file, 3); // Inspect first 3 vectors

        // Load and align base data
        try {
            std::cout << "Loading base data..." << std::endl;
            load_data(base_file, data_load, num_points, original_dim);

            if (!data_load) {
                throw std::runtime_error("Failed to load base data");
            }

            // Calculate memory requirements for aligned data
            size_t aligned_dim_temp = (original_dim + DATA_ALIGN_FACTOR - 1) / DATA_ALIGN_FACTOR * DATA_ALIGN_FACTOR;
            size_t required_memory = static_cast<size_t>(num_points) * aligned_dim_temp * sizeof(float);

            if (!checkMemoryAvailability(required_memory)) {
                throw std::runtime_error("Insufficient memory for data alignment");
            }

            // Align data
            std::cout << "Aligning data..." << std::endl;
            aligned_dim = original_dim;
            aligned_data.reset(efanna2e::data_align(data_load.release(), num_points, aligned_dim));

            if (!aligned_data) {
                throw std::runtime_error("Data alignment returned null");
            }

            std::cout << "Data alignment completed:" << std::endl
                      << "Original dimension: " << original_dim << std::endl
                      << "Aligned dimension: " << aligned_dim << std::endl;

            // Get parameters for this dataset
            auto nsg_params = getNSGParamSets(dataset_name);

            for (const auto& params : nsg_params) {
                std::cout << "\nProcessing with parameters:" << std::endl
                          << "K: " << params.K << std::endl
                          << "L: " << params.L << std::endl
                          << "R_knn: " << params.R_knn << std::endl
                          << "R_nsg: " << params.R_nsg << std::endl
                          << "C: " << params.C << std::endl;

                // Create NN graph if needed
                if (!std::filesystem::exists(graph_file)) {
                    std::cout << "\nCreating NN graph..." << std::endl;
                    try {
                        createNNGraph(aligned_data.get(), num_points, aligned_dim,
                            graph_file.c_str(), params);
                    } catch (const std::exception& e) {
                        throw std::runtime_error(std::string("NN graph creation failed: ") + e.what());
                    }
                }

                // Verify graph file
                if (!std::filesystem::exists(graph_file)) {
                    throw std::runtime_error("Graph file was not created successfully");
                }
                std::cout << "Graph file size: " << std::filesystem::file_size(graph_file)
                          << " bytes" << std::endl;

                // Create NSG index if needed
                if (!std::filesystem::exists(index_file)) {
                    std::cout << "\nCreating NSG index..." << std::endl;
                    try {
                        createNSGIndex(aligned_data.get(), num_points, aligned_dim,
                            graph_file.c_str(), index_file.c_str(), params);
                    } catch (const std::exception& e) {
                        throw std::runtime_error(std::string("NSG index creation failed: ") + e.what());
                    }
                }

                // Verify index file
                if (!std::filesystem::exists(index_file)) {
                    throw std::runtime_error("Index file was not created successfully");
                }
                std::cout << "Index file size: " << std::filesystem::file_size(index_file)
                          << " bytes" << std::endl;

                // Perform search
                std::cout << "\nPerforming search..." << std::endl;
                try {
                    searchNSGIndex(aligned_data.get(), num_points, original_dim, aligned_dim,
                        query_file.c_str(), index_file.c_str(),
                        groundtruth_file.c_str(), result_file.c_str(),
                        params);
                } catch (const std::exception& e) {
                    throw std::runtime_error(std::string("Search failed: ") + e.what());
                }
            }

        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Dataset processing failed: ") + e.what());
        }

    } catch (const std::exception& e) {
        std::cerr << "Error processing dataset " << dataset_name << ": " << e.what() << std::endl;
    }

    std::cout << "\nFinished processing " << dataset_name << std::endl;
    // Allow system time to cleanup
    std::this_thread::sleep_for(std::chrono::seconds(2));
}

int main(int argc, char** argv)
{
    std::vector<std::string> datasets_to_process;

    if (argc > 1) {
        // Process specific datasets provided as command-line arguments
        for (int i = 1; i < argc; i++) {
            datasets_to_process.push_back(argv[i]);
        }
    } else {
        // Process all datasets
        datasets_to_process = DATASETS;
    }

    for (const auto& dataset : datasets_to_process) {
        try {
            std::cout << "\nStarting processing of dataset: " << dataset << std::endl;
            processDataset(dataset);
            std::cout << "Finished processing dataset: " << dataset << "\n"
                      << std::endl;

            // Give system time to free memory between datasets
            std::this_thread::sleep_for(std::chrono::seconds(2));
        } catch (const std::exception& e) {
            std::cerr << "Error processing dataset " << dataset << ": " << e.what() << std::endl;
            continue; // Continue with next dataset even if one fails
        }
    }

    std::cout << "\nAll datasets processed." << std::endl;
    return 0;
}