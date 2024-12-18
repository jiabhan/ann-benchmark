#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "config.h"

#include <NGT/Capi.h>
#include <NGT/Index.h>
#include <cfloat>

const size_t K = 10; // Number of nearest neighbors to retrieve
const size_t NUM_FOLDS = 10; // Number of folds for cross-validation

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
    NGTError err = ngt_create_error_object();
    NGTProperty prop = ngt_create_property(err);
    if (prop == NULL) {
        std::cerr << "Error creating property: " << ngt_get_error_string(err) << std::endl;
        ngt_destroy_error_object(err);
        throw std::runtime_error("Error creating property");
    }

    if (!ngt_set_property_distance_type_l2(prop, err)) {
        std::cerr << "Error setting property distance type L2: " << ngt_get_error_string(err) << std::endl;
        ngt_destroy_property(prop);
        ngt_destroy_error_object(err);
        throw std::runtime_error("Error setting property distance type L2");
    }

    if (!ngt_set_property_dimension(prop, data[0].size(), err)) {
        std::cerr << "Error setting property dimension: " << ngt_get_error_string(err) << std::endl;
        ngt_destroy_property(prop);
        ngt_destroy_error_object(err);
        throw std::runtime_error("Error setting property dimension");
    }

    if (!ngt_set_property_edge_size_for_creation(prop, 20, err)) {
        std::cerr << "Error setting property edge size: " << ngt_get_error_string(err) << std::endl;
        ngt_destroy_property(prop);
        ngt_destroy_error_object(err);
        throw std::runtime_error("Error setting property edge size");
    }

    if (!ngt_set_property_edge_size_for_search(prop, 60, err)) {
        std::cerr << "Error setting property edge size for search: " << ngt_get_error_string(err) << std::endl;
        ngt_destroy_property(prop);
        ngt_destroy_error_object(err);
        throw std::runtime_error("Error setting property edge size for search");
    }

    // Create the index
    NGTIndex index = ngt_create_graph_and_tree(index_name, prop, err);
    if (index == NULL) {
        std::cerr << "Error creating index: " << ngt_get_error_string(err) << std::endl;
        ngt_destroy_property(prop);
        ngt_destroy_error_object(err);
        throw std::runtime_error("Error creating index");
    }

    // Insert data
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
    } catch (NGT::Exception& err) {
        std::cerr << "Error " << err.what() << std::endl;
        throw std::runtime_error("Error inserting object");
    } catch (...) {
        std::cerr << "Error" << std::endl;
        throw std::runtime_error("Error inserting object");
    }

    // Build the index
    std::cout << "Building the index..." << std::endl;
    if (!ngt_create_index(index, 20, err)) {
        std::cerr << "Error creating index: " << ngt_get_error_string(err) << std::endl;
        ngt_close_index(index);
        ngt_destroy_property(prop);
        ngt_destroy_error_object(err);
        throw std::runtime_error("Error creating index");
    }

    std::cout << "Saving the index..." << std::endl;
    if (!ngt_save_index(index, index_name, err)) {
        std::cerr << "Error saving index: " << ngt_get_error_string(err) << std::endl;
        ngt_close_index(index);
        ngt_destroy_property(prop);
        ngt_destroy_error_object(err);
        throw std::runtime_error("Error saving index");
    }

    std::cerr << "Closing the index..." << std::endl;
    ngt_close_index(index);

    ngt_destroy_property(prop);
    ngt_destroy_error_object(err);
}

void pruneIndex(const char* index_path, int pathadj_size)
{
    std::cout << "Pruning index with pathadj_size: " << pathadj_size << "..." << std::endl;

    // Construct the command
    std::string command = "ngt prune -s " + std::to_string(pathadj_size) + " " + index_path;

    // Execute the command
    int result = system(command.c_str());

    if (result != 0) {
        throw std::runtime_error("Error pruning index. Command failed: " + command);
    }

    std::cout << "Index pruning completed." << std::endl;
}

NGTIndex loadIndex(const char* index_name)
{
    std::cerr << "Opening the index..." << std::endl;
    NGTError err = ngt_create_error_object();

    NGTIndex index = ngt_open_index(index_name, err);

    if (index == NULL) {
        throw std::runtime_error("Error loading index");
    }

    ngt_destroy_error_object(err);

    return index;
}

std::pair<double, std::vector<std::vector<int>>>
benchmarkSearch(
    NGTIndex index,
    const std::vector<std::vector<float>>& queries,
    float epsilon)
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

        NGTQueryParameters panng_query_params;
        panng_query_params.size = 10;
        panng_query_params.epsilon = epsilon - 1;
        panng_query_params.radius = FLT_MAX;
        panng_query_params.edge_size = 60;

        NGTQueryFloat panng_query_float;
        panng_query_float.query = const_cast<float*>(query.data());
        panng_query_float.params = panng_query_params;

        NGTObjectDistances results = ngt_create_empty_results(err);
        if (results == NULL) {
            ngt_destroy_error_object(err);
            throw std::runtime_error("Error creating empty results object for query");
        }

        bool search_success = ngt_search_index_with_query_float(
            index,
            panng_query_float,
            results,
            err);

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

double calculateRecall(const std::vector<std::vector<int>>& ground_truth, const std::vector<std::vector<int>>& results, size_t K)
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

// Add this new function to check if an index exists
bool indexExists(const std::string& index_path)
{
    return std::filesystem::exists(index_path + "grp") && std::filesystem::exists(index_path + "tre") && std::filesystem::exists(index_path + "obj");
}

std::vector<std::pair<double, double>> kFoldParameterSweep(
    const std::string& index_path,
    const std::vector<std::vector<float>>& data,
    const std::vector<std::vector<float>>& queries,
    const std::vector<std::vector<int>>& ground_truth,
    const std::vector<double>& query_args,
    size_t num_folds)
{
    std::vector<std::pair<double, double>> overall_results;

    size_t fold_size = queries.size() / num_folds;

    NGTIndex index = NULL;
    try {
        if (indexExists(index_path)) {
            std::cout << "Attempting to load existing index from " << index_path << std::endl;
            index = loadIndex(index_path.c_str());
        } else {
            std::cout << "Creating new index at " << index_path << std::endl;
            try {
                buildIndex(data, index_path.c_str());
                try {
                    pruneIndex(index_path.c_str(), 40); // Set pathadj_size to 40
                } catch (const std::exception& e) {
                    std::cerr << "Error during index creation or pruning: " << e.what() << std::endl;
                    throw std::runtime_error("Failed to build or prune index.");
                }
                std::cout << "Attempting to load existing index from " << index_path << std::endl;
                index = loadIndex(index_path.c_str());
            } catch (...) {
                throw std::runtime_error("Failed to load or build index.");
            }
        }

        for (const auto& epsilon : query_args) {
            std::vector<size_t> indices(queries.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);

            std::vector<double> fold_recalls;
            std::vector<double> fold_qps;

            for (size_t fold = 0; fold < num_folds; ++fold) {
                std::vector<std::vector<float>> test_queries;
                std::vector<std::vector<int>> test_ground_truth;

                for (size_t i = fold * fold_size; i < (fold + 1) * fold_size && i < queries.size(); ++i) {
                    test_queries.push_back(queries[indices[i]]);
                    test_ground_truth.push_back(ground_truth[indices[i]]);
                }

                auto [qps, search_results] = benchmarkSearch(index, test_queries, epsilon);
                double recall = calculateRecall(test_ground_truth, search_results, K);

                fold_recalls.push_back(recall);
                fold_qps.push_back(qps);
            }

            double avg_recall = std::accumulate(fold_recalls.begin(), fold_recalls.end(), 0.0) / fold_recalls.size();
            double avg_qps = std::accumulate(fold_qps.begin(), fold_qps.end(), 0.0) / fold_qps.size();

            overall_results.push_back({ avg_recall, avg_qps });

            std::cout << "PANNG Params: "
                      << "epsilon=" << epsilon
                      << " | Avg Recall: " << avg_recall << ", Avg QPS: " << avg_qps << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error during parameter sweep: " << e.what() << std::endl;
        std::cerr << "Skipping to next parameter set..." << std::endl;
    }

    if (index != NULL) {
        ngt_close_index(index);
    }

    return overall_results;
}

int main()
{
    std::string index_data_file = std::string(data_dir) + "sift/sift_base.fvecs";
    std::string query_data_file = std::string(data_dir) + "sift/sift_query.fvecs";
    std::string ground_truth_file = std::string(data_dir) + "sift/sift_groundtruth.ivecs";
    std::string index_path = std::string(index_dir) + "panng-test/sift-test/";
    std::string result_file_path = std::string(result_dir) + "panng-test/sift/sift_recall_qps_result.csv";

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

    std::vector<double> query_args = {
        0.6, 0.8, 0.9, 1.0, 1.02, 1.05, 1.1, 1.2
    };

    std::cout << "Performing k-fold cross-validation with parameter sweep for PANNG..." << std::endl;
    auto panng_results = kFoldParameterSweep(index_path, index_data, all_query_data, ground_truth, query_args, NUM_FOLDS);

    std::ofstream panng_file(result_file_path);
    panng_file << "Recall,QPS\n";
    for (const auto& [recall, qps] : panng_results) {
        panng_file << recall << "," << qps << "\n";
    }
    panng_file.close();

    std::cout << "K-fold cross-validation with parameter sweep complete. Results written to " << result_file_path << std::endl;

    return 0;
}