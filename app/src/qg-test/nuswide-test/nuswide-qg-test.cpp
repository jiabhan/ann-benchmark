#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "config.h"

#include <NGT/Index.h>
#include <NGT/NGTQ/Capi.h>

#include <NGT/GraphOptimizer.h>

#include <cfloat>

const size_t K = 10; // Number of nearest neighbors to retrieve

struct ONNGQueryParam {
    float epsilon;
    float expansion;
};

struct ONNGParams {
    int edge;
    int indegree;
    int outdegree;
    bool refine;
    bool tree;
    std::vector<ONNGQueryParam> query_params;
};

std::vector<ONNGParams> getONNGParamSets()
{
    std::vector<ONNGParams> params;

    // Base parameters common to all parameter sets
    ONNGParams base;
    base.edge = 100;
    base.indegree = 120;
    base.outdegree = 10;

    // Standard parameter sets (without refinement)
    {
        ONNGParams p = base;
        p.refine = false;
        p.tree = true;
        params.push_back(p);
    }

    // Refinement parameter sets
    {
        ONNGParams p = base;
        p.refine = true;
        p.tree = false;
        params.push_back(p);
    }

    return params;
}

struct PruneConfig {
    bool shortcutReduction = true;
    bool searchOptimization = true;
    bool prefetchOptimization = true;
    bool accuracyTable = true;
    bool lessMemory = false;
    size_t numThreads = 0;
};

std::vector<std::vector<float>> readFvecs(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    std::vector<std::vector<float>> data;
    int32_t dim;
    while (file.read(reinterpret_cast<char*>(&dim), sizeof(dim))) {
        std::vector<float> vec(dim);
        if (!file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float))) {
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

void buildIndex(const std::vector<std::vector<float>>& data, const char* index_name)
{
    // Create error and property objects
    NGTError err = ngt_create_error_object();
    NGTProperty prop = ngt_create_property(err);
    if (prop == NULL) {
        std::cerr << "Error creating property: " << ngt_get_error_string(err) << std::endl;
        ngt_destroy_error_object(err);
        throw std::runtime_error("Error creating property");
    }

    // Set property dimensions
    if (!ngt_set_property_dimension(prop, data[0].size(), err)) {
        std::cerr << "Error setting property dimension: " << ngt_get_error_string(err) << std::endl;
        ngt_destroy_property(prop);
        ngt_destroy_error_object(err);
        throw std::runtime_error("Error setting property dimension");
    }

    // Set edge size for creation
    if (!ngt_set_property_edge_size_for_creation(prop, 100, err)) {
        std::cerr << "Error setting property edge size for creation: " << ngt_get_error_string(err) << std::endl;
        ngt_destroy_property(prop);
        ngt_destroy_error_object(err);
        throw std::runtime_error("Error setting property edge size for creation");
    }

    // Create the index
    NGTIndex index = ngt_create_graph_and_tree(index_name, prop, err);
    if (index == NULL) {
        std::cerr << "Error creating index: " << ngt_get_error_string(err) << std::endl;
        ngt_destroy_property(prop);
        ngt_destroy_error_object(err);
        throw std::runtime_error("Error creating index");
    }

    // Insert data into the index
    std::cerr << "Inserting " << data.size() << " objects..." << std::endl;
    try {
        for (const auto& obj : data) {
            if (!ngt_insert_index_as_float(index, const_cast<float*>(obj.data()), obj.size(), err)) {
                std::cerr << "Error inserting object: " << ngt_get_error_string(err) << std::endl;
                ngt_close_index(index);
                ngt_destroy_property(prop);
                ngt_destroy_error_object(err);
                throw std::runtime_error("Error inserting object");
            }
        }
    } catch (NGT::Exception& e) {
        std::cerr << "NGT Exception: " << e.what() << std::endl;
        ngt_close_index(index);
        ngt_destroy_property(prop);
        ngt_destroy_error_object(err);
        throw;
    } catch (...) {
        std::cerr << "Unknown error during insertion." << std::endl;
        ngt_close_index(index);
        ngt_destroy_property(prop);
        ngt_destroy_error_object(err);
        throw std::runtime_error("Unknown error during insertion.");
    }

    // Build the index
    std::cout << "Building the index..." << std::endl;
    if (!ngt_create_index(index, 16, err)) {
        std::cerr << "Error creating index: " << ngt_get_error_string(err) << std::endl;
        ngt_close_index(index);
        ngt_destroy_property(prop);
        ngt_destroy_error_object(err);
        throw std::runtime_error("Error creating index");
    }

    // Save the index
    std::cout << "Saving the index..." << std::endl;
    if (!ngt_save_index(index, index_name, err)) {
        std::cerr << "Error saving index: " << ngt_get_error_string(err) << std::endl;
        ngt_close_index(index);
        ngt_destroy_property(prop);
        ngt_destroy_error_object(err);
        throw std::runtime_error("Error saving index");
    }

    // Close the original index
    std::cerr << "Closing the index..." << std::endl;
    ngt_close_index(index);

    // Handle paths using std::filesystem
    std::filesystem::path index_path(index_name);

    // Remove trailing slash if present
    if (index_path.has_filename() && index_path.filename() == "") {
        index_path = index_path.parent_path();
    }

    // Define the optimized index path as a sibling directory, not a subdirectory
    std::filesystem::path optimized_path = index_path.parent_path() / (index_path.stem().string() + "_optimized");

    // After defining optimized_path
    std::cout << "Original Index Path: " << index_path.string() << std::endl;
    std::cout << "Optimized Index Path: " << optimized_path.string() << std::endl;

    // Before executing the optimizer
    std::cout << "Executing optimizer from " << index_path.string() << " to " << optimized_path.string() << std::endl;

    try {
        // Remove any existing optimized index directory
        if (std::filesystem::exists(optimized_path)) {
            std::filesystem::remove_all(optimized_path);
        }

        // Configure pruning options
        PruneConfig config = PruneConfig();

        NGT::GraphOptimizer optimizer;
        optimizer.shortcutReduction = config.shortcutReduction;
        optimizer.searchParameterOptimization = config.searchOptimization;
        optimizer.prefetchParameterOptimization = config.prefetchOptimization;
        optimizer.accuracyTableGeneration = config.accuracyTable;
        optimizer.shortcutReductionWithLessMemory = config.lessMemory;
        optimizer.numOfThreads = config.numThreads;

        // Execute optimization: source is index_path, destination is optimized_path
        optimizer.execute(index_path.string().c_str(), optimized_path.string().c_str());

        // Verify the optimized index exists
        if (!std::filesystem::exists(optimized_path)) {
            throw std::runtime_error("Optimized index was not created successfully");
        }

        // Remove the original index directory
        std::filesystem::remove_all(index_path);

        // Rename the optimized index to the original index name
        std::filesystem::rename(optimized_path, index_path);

    } catch (NGT::Exception& e) {
        // Clean up optimized index if an error occurs
        if (std::filesystem::exists(optimized_path)) {
            std::filesystem::remove_all(optimized_path);
        }
        throw std::runtime_error(std::string("Error pruning index: ") + e.what());
    } catch (const std::exception& e) {
        // Handle other exceptions
        if (std::filesystem::exists(optimized_path)) {
            std::filesystem::remove_all(optimized_path);
        }
        throw std::runtime_error(std::string("Error pruning index: ") + e.what());
    }

    // Quantize the optimized index (now at index_path)
    std::cout << "Quantizing the index..." << std::endl;
    NGTQGQuantizationParameters qparams;
    ngtqg_initialize_quantization_parameters(&qparams);
    qparams.max_number_of_edges = 96;

    // Perform quantization on the optimized index
    if (!ngtqg_quantize(index_path.string().c_str(), qparams, err)) {
        std::cerr << "Error quantizing index: " << ngt_get_error_string(err) << std::endl;
        ngt_destroy_property(prop);
        ngt_destroy_error_object(err);
        throw std::runtime_error("Error quantizing index");
    }

    std::cout << "Quantization complete." << std::endl;

    // Clean up
    ngt_destroy_property(prop);
    ngt_destroy_error_object(err);
}

NGTQGIndex loadIndex(const char* index_name)
{
    std::cerr << "Opening the quantized index..." << std::endl;
    NGTError err = ngt_create_error_object();

    NGTQGIndex index = ngtqg_open_index(index_name, err);
    if (index == NULL) {
        throw std::runtime_error("Error loading index");
    }

    ngt_destroy_error_object(err);

    return index;
}

bool indexExists(const std::string& index_path)
{
    return std::filesystem::exists(index_path + "/grp") && std::filesystem::exists(index_path + "/tre") && std::filesystem::exists(index_path + "/obj");
}

std::pair<NGTIndex, bool> getOrCreateIndex(
    const std::string& index_path,
    const std::vector<std::vector<float>>& data)
{
    bool created = false;

    // If index doesn't exist, create it
    if (!indexExists(index_path)) {
        buildIndex(data, index_path.c_str());
        created = true;
    }

    // Load the index
    NGTIndex index = loadIndex(index_path.c_str());
    return { index, created };
}

std::pair<double, std::vector<std::vector<int>>>
benchmarkSearch(
    NGTQGIndex index,
    const std::vector<std::vector<float>>& queries,
    float epsilon,
    float result_expansion)
{
    std::cout << "Starting benchmark search with " << queries.size() << " queries." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<int>> all_results;

    NGTError err = ngt_create_error_object();
    if (err == NULL) {
        throw std::runtime_error("Error creating error object.");
    }

    for (size_t i = 0; i < queries.size(); ++i) {
        const auto& query = queries[i];

        NGTQGQuery qg_query;
        ngtqg_initialize_query(&qg_query);

        qg_query.query = const_cast<float*>(query.data());
        qg_query.size = 10;
        qg_query.epsilon = epsilon - 1;
        qg_query.result_expansion = result_expansion;
        qg_query.radius = FLT_MAX;

        NGTObjectDistances results = ngt_create_empty_results(err);
        if (results == NULL) {
            ngt_destroy_error_object(err);
            throw std::runtime_error("Error creating empty results object for query");
        }

        bool search_success = ngtqg_search_index(index, qg_query, results, err);
        if (!search_success) {
            std::string error_msg = ngt_get_error_string(err);
            std::cerr << "Search failed: " << error_msg << std::endl;
            ngt_destroy_results(results);
            ngt_destroy_error_object(err);
            throw std::runtime_error(error_msg);
        }

        std::vector<int> query_results;
        size_t result_size = ngt_get_result_size(results, err);

        for (size_t i = 0; i < result_size; ++i) {
            NGTObjectDistance result = ngt_get_result(results, i, err);
            query_results.push_back(result.id - 1);
        }
        all_results.push_back(query_results);

        ngt_destroy_results(results);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double qps = static_cast<double>(queries.size()) / (duration.count() / 1000.0);

    std::cout << "Benchmark search completed in " << duration.count() << " milliseconds." << std::endl;
    std::cout << "Queries per second (QPS): " << qps << std::endl;

    ngt_destroy_error_object(err);
    return { qps, all_results };
}

double calculateRecall(const std::vector<std::vector<int>>& ground_truth, const std::vector<std::vector<int>>& results)
{
    if (ground_truth.size() != results.size()) {
        std::cerr << "Error: ground truth size (" << ground_truth.size()
                  << ") doesn't match results size (" << results.size() << ")" << std::endl;
        return 0.0;
    }

    double total_recall = 0.0;

    for (size_t i = 0; i < ground_truth.size(); ++i) {
        size_t correct_count = 0;
        for (size_t j = 0; j < std::min(K, results[i].size()); ++j) {
            if (results[i][j] == ground_truth[i][0]) {
                ++correct_count;
            }
        }
        total_recall += static_cast<double>(correct_count);
    }
    return total_recall / ground_truth.size();
}

std::vector<std::tuple<double, double>> evaluateParameters(
    const std::string& index_path,
    const std::vector<std::vector<float>>& data,
    const std::vector<std::vector<float>>& queries,
    const std::vector<std::vector<int>>& ground_truth,
    const std::vector<std::pair<double, double>>& query_args)
{
    std::vector<std::tuple<double, double>> results;

    try {
        auto [index, was_created] = getOrCreateIndex(index_path, data);
        ngt_close_index(index);

        // Load and search
        index = loadIndex(index_path.c_str());
        for (const auto& [result_expansion, epsilon] : query_args) {
            auto [qps, search_results] = benchmarkSearch(index, queries, epsilon, result_expansion);
            double recall = calculateRecall(ground_truth, search_results);
            results.push_back({ recall, qps });

            std::cout << "QG Params: "
                      << "result_expansion=" << result_expansion
                      << ", epsilon=" << epsilon
                      << " | Recall: " << recall << ", Avg QPS: " << qps << std::endl;
        }

        ngt_close_index(index);

    } catch (const std::exception& e) {
        std::cerr << "Error processing index: " << e.what() << std::endl;
    }

    return results;
}

int main()
{

    std::string index_data_file = std::string(data_dir) + "nuswide/nuswide_base.fvecs";
    std::string query_data_file = std::string(data_dir) + "nuswide/nuswide_query.fvecs";
    std::string ground_truth_file = std::string(data_dir) + "nuswide/nuswide_groundtruth.ivecs";
    std::string index_path = std::string(index_dir) + "qg-test/nuswide-test";
    std::string result_file_path = std::string(result_dir) + "qg-test/nuswide/nuswide_recall_qps_result.csv";

    std::cout << "Reading index data file: " << index_data_file << std::endl;
    std::vector<std::vector<float>> index_data = readFvecs(index_data_file);
    if (index_data.empty()) {
        std::cerr << "Error: Failed to read index data from " << index_data_file << std::endl;
        return 1;
    }
    std::cout << "Read " << index_data.size() << " vectors from " << index_data_file << std::endl;

    std::cout << "Reading query data file: " << query_data_file << std::endl;
    std::vector<std::vector<float>> all_query_data = readFvecs(query_data_file);
    if (all_query_data.empty()) {
        std::cerr << "Error: Failed to read query data from " << query_data_file << std::endl;
        return 1;
    }
    std::cout << "Read " << all_query_data.size() << " vectors from " << query_data_file << std::endl;

    std::cout << "Reading ground truth data file: " << ground_truth_file << std::endl;
    std::vector<std::vector<int>> ground_truth = readIvecs(ground_truth_file);
    if (ground_truth.empty()) {
        std::cerr << "Error: Failed to read ground truth data from " << ground_truth_file << std::endl;
        return 1;
    }
    std::cout << "Read " << ground_truth.size() << " vectors from " << ground_truth_file << std::endl;

    if (index_data.empty() || all_query_data.empty() || ground_truth.empty()) {
        std::cerr << "Error reading data files." << std::endl;
        return 1;
    }

    std::vector<std::pair<double, double>> query_args = {
        { 0.0, 0.9 }, { 0.0, 0.95 }, { 0.0, 0.98 }, { 0.0, 1.0 },
        { 1.2, 0.9 }, { 1.5, 0.9 }, { 2.0, 0.9 }, { 3.0, 0.9 },
        { 1.2, 0.95 }, { 1.5, 0.95 }, { 2.0, 0.95 }, { 3.0, 0.95 },
        { 1.2, 0.98 }, { 1.5, 0.98 }, { 2.0, 0.98 }, { 3.0, 0.98 },
        { 1.2, 1.0 }, { 1.5, 1.0 }, { 2.0, 1.0 }, { 3.0, 1.0 },
        { 5.0, 1.0 }, { 10.0, 1.0 }, { 20.0, 1.0 },
        { 1.2, 1.02 }, { 1.5, 1.02 }, { 2.0, 1.02 }, { 3.0, 1.02 },
        { 2.0, 1.04 }, { 3.0, 1.04 }, { 5.0, 1.04 }, { 8.0, 1.04 }
    };

    auto qg_results = evaluateParameters(index_path, index_data, all_query_data, ground_truth, query_args);

    std::ofstream qg_file(result_file_path);
    qg_file << "Recall,QPS\n";
    for (const auto& [recall, qps] : qg_results) {
        qg_file << recall << "," << qps << "\n";
    }
    qg_file.close();

    std::cout << "Complete. Results written to " << result_file_path << std::endl;

    return 0;
}