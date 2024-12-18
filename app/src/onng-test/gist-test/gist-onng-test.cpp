#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "config.h"
#include <NGT/Capi.h>
#include <NGT/GraphOptimizer.h>
#include <NGT/Index.h>
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

void pruneIndex(const char* index_path, int pathadj_size)
{
    std::cout << "Pruning index with pathadj_size: " << pathadj_size << "..." << std::endl;

    try {
        // Create temporary output path for optimized index
        std::string out_path = std::string(index_path) + "_optimized";

        // Create graph optimizer
        NGT::GraphOptimizer optimizer;

        // Configure optimization parameters
        optimizer.shortcutReduction = true;
        optimizer.searchParameterOptimization = true;
        optimizer.prefetchParameterOptimization = true;
        optimizer.accuracyTableGeneration = true;
        optimizer.shortcutReductionWithLessMemory = false;
        optimizer.numOfThreads = 0; // Use default

        // Execute optimization
        optimizer.execute(index_path, out_path);

        // Replace original index with optimized version
        std::filesystem::remove_all(index_path);
        std::filesystem::rename(out_path, index_path);

    } catch (NGT::Exception &err) {
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
            if (!indexExists(current_index)) {
                buildIndex(data, current_index.c_str(), params);
                if (params.refine) {
                    pruneIndex(current_index.c_str(), 40);
                }
            }

            NGTIndex index = loadIndex(current_index.c_str());

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

int main()
{
    std::string index_data_file = std::string(data_dir) + "gist/gist_base.fvecs";
    std::string query_data_file = std::string(data_dir) + "gist/gist_query.fvecs";
    std::string ground_truth_file = std::string(data_dir) + "gist/gist_groundtruth.ivecs";
    std::string index_path = std::string(index_dir) + "onng-test/gist";
    std::string result_file = std::string(result_dir) + "onng-test/gist-test/gist_recall_qps_result.csv";

    // Create directories if they don't exist
    std::filesystem::create_directories(std::string(index_dir) + "onng-test/");
    std::filesystem::create_directories(std::string(result_dir) + "onng-test/");

    std::cout << "Reading data files..." << std::endl;
    auto index_data = readFvecs(index_data_file);
    auto query_data = readFvecs(query_data_file);
    auto ground_truth = readIvecs(ground_truth_file);

    auto onng_params = getONNGParamSets();

    std::cout << "Evaluating ONNG parameters..." << std::endl;
    auto results = evaluateParameters(index_path, index_data, query_data, ground_truth, onng_params);

    std::ofstream result_file_stream(result_file);
    result_file_stream << "Recall,QPS\n";
    for (const auto& [recall, qps] : results) {
        result_file_stream << recall << "," << qps << "\n";
    }

    std::cout << "Results written to " << result_file << std::endl;
    return 0;
}
