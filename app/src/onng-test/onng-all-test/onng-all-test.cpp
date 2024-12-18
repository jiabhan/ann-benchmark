#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

#include "config.h"
#include <NGT/Capi.h>
#include <NGT/Index.h>
#include <cfloat>

#include <NGT/GraphOptimizer.h>

const size_t K = 10; // Number of nearest neighbors to retrieve

const std::vector<std::string> DATASETS = {
    "audio", "cifar", "deep", "enron", "gist", "glove",
    "imagenet", "millionsong", "mnist", "notre", "nuswide",
    "sift", "siftsmall", "sun", "trevi", "ukbench",
    "wikipedia-2024-06-bge-m3-zh"
};

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
        p.query_params = {
            { 0.02, -2 },
            { 0.03, -2 },
            { 0.04, -2 },
            { 0.06, -2 },
            { 0.08, -2 },
            { 0.1, -2 }
        };
        params.push_back(p);
    }

    // Refinement parameter sets
    {
        ONNGParams p = base;
        p.refine = true;
        p.tree = false;
        p.query_params = {
            { 0.08, 40 }, { 0.1, 40 }
        };
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

void buildIndex(const std::vector<std::vector<float>>& data, const char* index_name, const ONNGParams& params)
{
    NGTError err = ngt_create_error_object();
    NGTProperty prop = ngt_create_property(err);

    ngt_set_property_dimension(prop, data[0].size(), err);
    ngt_set_property_distance_type_l2(prop, err);
    ngt_set_property_edge_size_for_creation(prop, params.edge, err);
    ngt_set_property_edge_size_for_search(prop, params.indegree, err);

    NGTIndex index = ngt_create_graph_and_tree(index_name, prop, err);
    if (index == NULL) {
        throw std::runtime_error("Error creating index");
    }

    std::cout << "Inserting " << data.size() << " objects..." << std::endl;
    for (const auto& obj : data) {
        if (!ngt_insert_index_as_float(index, const_cast<float*>(obj.data()), obj.size(), err)) {
            throw std::runtime_error("Error inserting object");
        }
    }

    std::cout << "Building the index..." << std::endl;
    if (!ngt_create_index(index, params.outdegree, err)) {
        throw std::runtime_error("Error creating index");
    }

    std::cout << "Saving the index..." << std::endl;
    if (!ngt_save_index(index, index_name, err)) {
        throw std::runtime_error("Error saving index");
    }

    ngt_close_index(index);
    ngt_destroy_property(prop);
    ngt_destroy_error_object(err);
}

void pruneIndex(const char* index_path, int pathadj_size, const PruneConfig& config = PruneConfig())
{
    std::cout << "Pruning index with pathadj_size: " << pathadj_size << "..." << std::endl;

    try {
        std::string out_path = std::string(index_path) + "_optimized";

        // Clean up any existing optimized index first
        if (std::filesystem::exists(out_path)) {
            std::filesystem::remove_all(out_path);
        }

        NGT::GraphOptimizer optimizer;
        optimizer.shortcutReduction = config.shortcutReduction;
        optimizer.searchParameterOptimization = config.searchOptimization;
        optimizer.prefetchParameterOptimization = config.prefetchOptimization;
        optimizer.accuracyTableGeneration = config.accuracyTable;
        optimizer.shortcutReductionWithLessMemory = config.lessMemory;
        optimizer.numOfThreads = config.numThreads;

        optimizer.execute(index_path, out_path);

        // Verify the optimized index exists before removing original
        if (!std::filesystem::exists(out_path)) {
            throw std::runtime_error("Optimized index was not created successfully");
        }

        std::filesystem::remove_all(index_path);
        std::filesystem::rename(out_path, index_path);

    } catch (NGT::Exception& err) {
        // Clean up optimized index if it exists
        std::string out_path = std::string(index_path) + "_optimized";
        if (std::filesystem::exists(out_path)) {
            std::filesystem::remove_all(out_path);
        }
        throw std::runtime_error("Error pruning index: " + std::string(err.what()));
    }
}

NGTIndex loadIndex(const char* index_name)
{
    NGTError err = ngt_create_error_object();
    NGTIndex index = ngt_open_index(index_name, err);
    if (index == NULL) {
        throw std::runtime_error("Error loading index");
    }
    ngt_destroy_error_object(err);
    return index;
}

std::pair<double, std::vector<std::vector<int>>>
benchmarkSearch(NGTIndex index, const std::vector<std::vector<float>>& queries, const ONNGQueryParam& query_param)
{
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<int>> all_results;
    NGTError err = ngt_create_error_object();

    for (const auto& query : queries) {
        NGTQueryParameters query_params;
        query_params.size = K;
        query_params.epsilon = query_param.epsilon;
        query_params.radius = FLT_MAX;
        query_params.edge_size = 60; // Default edge size for search

        NGTQueryFloat query_float;
        query_float.query = const_cast<float*>(query.data());
        query_float.params = query_params;

        NGTObjectDistances results = ngt_create_empty_results(err);

        if (!ngt_search_index_with_query_float(index, query_float, results, err)) {
            throw std::runtime_error("Search failed");
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

    ngt_destroy_error_object(err);
    return { qps, all_results };
}

double calculateRecall(const std::vector<std::vector<int>>& ground_truth, const std::vector<std::vector<int>>& results)
{
    double total_recall = 0.0;
    for (size_t i = 0; i < ground_truth.size(); ++i) {
        for (size_t j = 0; j < std::min(K, results[i].size()); ++j) {
            if (results[i][j] == ground_truth[i][0]) {
                total_recall += 1.0;
                break;
            }
        }
    }
    return total_recall / ground_truth.size();
}

bool indexExists(const std::string& index_path)
{
    return std::filesystem::exists(index_path + "/grp") && std::filesystem::exists(index_path + "/tre") && std::filesystem::exists(index_path + "/obj");
}

std::pair<NGTIndex, bool> getOrCreateIndex(
    const std::string& index_path,
    const std::vector<std::vector<float>>& data,
    const ONNGParams& params)
{
    bool created = false;

    // If index doesn't exist, create it
    if (!indexExists(index_path)) {
        buildIndex(data, index_path.c_str(), params);
        created = true;
    }

    // Load the index
    NGTIndex index = loadIndex(index_path.c_str());
    return { index, created };
}

std::vector<std::tuple<double, double>> evaluateParameters(
    const std::string& base_index_path,
    const std::vector<std::vector<float>>& data,
    const std::vector<std::vector<float>>& queries,
    const std::vector<std::vector<int>>& ground_truth,
    const std::vector<ONNGParams>& param_sets)
{
    std::vector<std::tuple<double, double>> results;

    for (const auto& params : param_sets) {
        std::string current_index = base_index_path;
        if (params.refine) {
            current_index += "_refined";
        }

        try {
            // Get or create the base index
            auto [index, was_created] = getOrCreateIndex(base_index_path, data, params);
            ngt_close_index(index);

            if (params.refine) {
                // If refined index doesn't exist
                if (!indexExists(current_index)) {
                    // Copy base index to refined location
                    std::filesystem::copy(base_index_path, current_index,
                        std::filesystem::copy_options::recursive);
                    // Optimize the copy
                    pruneIndex(current_index.c_str(), 40);
                }
            }

            // Load and search
            index = loadIndex(current_index.c_str());
            for (const auto& qparam : params.query_params) {
                auto [qps, search_results] = benchmarkSearch(index, queries, qparam);
                double recall = calculateRecall(ground_truth, search_results);
                results.push_back({ recall, qps });

                std::cout << "ONNG Results: "
                          << "epsilon=" << qparam.epsilon
                          << " edge=" << params.edge
                          << " indegree=" << params.indegree
                          << " outdegree=" << params.outdegree
                          << " refine=" << params.refine
                          << " | Recall=" << recall
                          << " QPS=" << qps << std::endl;
            }

            ngt_close_index(index);

        } catch (const std::exception& e) {
            std::cerr << "Error processing index " << current_index << ": " << e.what() << std::endl;
        }
    }

    return results;
}

void processDataset(const std::string& dataset_name)
{
    // Construct paths
    std::string base_path = std::string(data_dir) + dataset_name + "/";
    std::string index_path = std::string(index_dir) + "onng-test/" + dataset_name;
    std::string result_path = std::string(result_dir) + "onng-test/" + dataset_name;
    std::string result_file = result_path + "/" + dataset_name + "_recall_qps_result.csv";

    // Create directories
    std::filesystem::create_directories(index_path);
    std::filesystem::create_directories(result_path);

    // Construct file paths
    std::string base_file = base_path + dataset_name + "_base.fvecs";
    std::string query_file = base_path + dataset_name + "_query.fvecs";
    std::string groundtruth_file = base_path + dataset_name + "_groundtruth.ivecs";

    try {
        std::cout << "\nProcessing dataset: " << dataset_name << std::endl;
        std::cout << "Reading data files..." << std::endl;

        if (!std::filesystem::exists(base_file) || !std::filesystem::exists(query_file) || !std::filesystem::exists(groundtruth_file)) {
            std::cerr << "Missing required files for dataset " << dataset_name << std::endl;
            return;
        }

        // Scope block to ensure large vectors are freed at end
        {
            auto index_data = readFvecs(base_file);
            auto query_data = readFvecs(query_file);
            auto ground_truth = readIvecs(groundtruth_file);
            auto onng_params = getONNGParamSets();

            std::cout << "Evaluating ONNG parameters for " << dataset_name << "..." << std::endl;
            auto results = evaluateParameters(index_path, index_data, query_data, ground_truth, onng_params);

            // Write results
            std::ofstream result_file_stream(result_file);
            result_file_stream << "Recall,QPS\n";
            for (const auto& [recall, qps] : results) {
                result_file_stream << recall << "," << qps << "\n";
            }

            std::cout << "Results written to " << result_file << std::endl;

            // Explicitly clear the data to free memory
            index_data.clear();
            index_data.shrink_to_fit();
            query_data.clear();
            query_data.shrink_to_fit();
            ground_truth.clear();
            ground_truth.shrink_to_fit();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error processing dataset " << dataset_name << ": " << e.what() << std::endl;
    }

    // Additional pause if desired
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

    // Process each dataset
    for (const auto& dataset : datasets_to_process) {
        processDataset(dataset);

        // Optional: Force garbage collection and give system time to free memory
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    return 0;
}