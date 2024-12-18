// qbg_test.cpp

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <unordered_set>
#include <vector>

#include "config.h"
#include <NGT/Capi.h>
#include <NGT/Index.h>
#include <NGT/NGTQ/Capi.h>
#include <NGT/NGTQ/Quantizer.h>

const size_t K = 10; // Number of nearest neighbors to retrieve

struct QBGParams {
    // Construction params
    size_t dimension;
    size_t num_subvectors;
    size_t num_blobs;

    // Build params
    size_t num_first_objects;
    size_t num_second_objects;
    size_t num_first_clusters;
    size_t num_second_clusters;
    size_t num_third_clusters;
    size_t sample_size;
    bool enable_rotation;

    // Query params
    std::vector<std::pair<float, float>> query_params; // (epsilon, blob_epsilon) pairs

    std::string toString() const
    {
        return "QBG(subvec=" + std::to_string(num_subvectors) + ",blobs=" + std::to_string(num_blobs) + ",clusters=" + std::to_string(num_first_clusters) + "," + std::to_string(num_second_clusters) + "," + std::to_string(num_third_clusters) + ")";
    }
};

QBGParams getParamsForDataset(const std::string& dataset_name, size_t num_vectors, size_t dimension)
{
    QBGParams params;
    params.dimension = dimension;

    // Previously, we had only a few query parameters. Let's enrich them based on the insights from
    // provided configurations. We maintain the original basic sets and then add a variety of
    // (epsilon, blob_epsilon) pairs that represent different accuracy-speed trade-offs.
    // These pairs are inspired by the configurations and query_args examples provided above.
    // The first value is the epsilon (search range coefficient), and the second is a "blob epsilon"
    // or an analogous parameter controlling the search expansion within the quantized structure.

    params.query_params = {
        // Original sets
        { 0.1f, 0.02f }, // High accuracy
        { 0.2f, 0.04f }, // Balanced
        { 0.3f, 0.06f }, // High speed

        // Additional sets derived from sample configurations:
        { 0.0f, 0.90f },
        { 0.0f, 0.95f },
        { 0.0f, 0.98f },
        { 0.0f, 1.00f },

        { 1.2f, 0.90f },
        { 1.5f, 0.90f },
        { 2.0f, 0.90f },
        { 3.0f, 0.90f },

        { 1.2f, 0.95f },
        { 1.5f, 0.95f },
        { 2.0f, 0.95f },
        { 3.0f, 0.95f },

        { 1.2f, 0.98f },
        { 1.5f, 0.98f },
        { 2.0f, 0.98f },
        { 3.0f, 0.98f },

        { 1.2f, 1.00f },
        { 1.5f, 1.00f },
        { 2.0f, 1.00f },
        { 3.0f, 1.00f },
        { 5.0f, 1.00f },
        { 10.0f, 1.00f },
        { 20.0f, 1.00f },

        { 1.2f, 1.02f },
        { 1.5f, 1.02f },
        { 2.0f, 1.02f },
        { 3.0f, 1.02f },

        { 2.0f, 1.04f },
        { 3.0f, 1.04f },
        { 5.0f, 1.04f },
        { 8.0f, 1.04f }
    };

    // Construction parameters:
    if (dimension > 2000) {
        // Ultra-high dimensional datasets (e.g., trevi, enron)
        params.num_subvectors = 512;
        params.num_blobs = 2000;
        params.num_first_clusters = 100;
        params.num_second_clusters = 200;
        params.num_third_clusters = 400;
        params.sample_size = std::min(size_t(10000), num_vectors / 10);
    } else if (dimension > 1000) {
        // Very high dimension (e.g. large embeddings)
        params.num_subvectors = 256;
        params.num_blobs = 1500;
        params.num_first_clusters = 75;
        params.num_second_clusters = 150;
        params.num_third_clusters = 300;
        params.sample_size = std::min(size_t(20000), num_vectors / 20);
    } else if (num_vectors > 2000000) {
        // Extremely large datasets
        params.num_subvectors = 256;
        params.num_blobs = 2500;
        params.num_first_clusters = 150;
        params.num_second_clusters = 300;
        params.num_third_clusters = 600;
        params.sample_size = 30000;
    } else if (num_vectors > 1000000) {
        // Large datasets
        params.num_subvectors = 128;
        params.num_blobs = 2000;
        params.num_first_clusters = 100;
        params.num_second_clusters = 200;
        params.num_third_clusters = 400;
        params.sample_size = 20000;
    } else if (num_vectors > 100000) {
        // Medium datasets
        params.num_subvectors = 64;
        params.num_blobs = 1000;
        params.num_first_clusters = 50;
        params.num_second_clusters = 100;
        params.num_third_clusters = 200;
        params.sample_size = 10000;
    } else {
        // Small datasets like siftsmall
        params.num_subvectors = 32;
        params.num_blobs = 100;
        params.num_first_clusters = 32;
        params.num_second_clusters = 64;
        params.num_third_clusters = 128;
        params.sample_size = std::min(size_t(2000), num_vectors / 5);
    }

    params.num_first_objects = params.sample_size;
    params.num_second_objects = params.sample_size / 2;

    params.enable_rotation = true;

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

void buildQBGIndex(const std::vector<std::vector<float>>& data,
    const char* index_path,
    const QBGParams& params)
{
    std::cout << "Building QBG index with " << params.toString() << std::endl;

    NGTError err = ngt_create_error_object();

    // Set construction parameters
    QBGConstructionParameters cparam;
    qbg_initialize_construction_parameters(&cparam);
    cparam.dimension = params.dimension;
    cparam.number_of_subvectors = params.num_subvectors;
    cparam.number_of_blobs = params.num_blobs;
    cparam.internal_data_type = NGTQ::DataTypeFloat;
    cparam.data_type = NGTQ::DataTypeFloat;
    cparam.distance_type = NGT::ObjectSpace::DistanceTypeL2;

    // Create index
    if (!qbg_create(index_path, &cparam, err)) {
        std::string error_msg = ngt_get_error_string(err);
        ngt_destroy_error_object(err);
        throw std::runtime_error("Error creating QBG index: " + error_msg);
    }

    // Open index for insertions
    QBGIndex index = qbg_open_index(index_path, false, err);
    if (index == nullptr) {
        std::string error_msg = ngt_get_error_string(err);
        ngt_destroy_error_object(err);
        throw std::runtime_error("Error opening QBG index: " + error_msg);
    }

    // Insert data
    std::cout << "Inserting " << data.size() << " vectors..." << std::endl;
    for (size_t i = 0; i < data.size(); i++) {
        if (!qbg_append_object(index, const_cast<float*>(data[i].data()), data[i].size(), err)) {
            std::string error_msg = ngt_get_error_string(err);
            qbg_close_index(index);
            ngt_destroy_error_object(err);
            throw std::runtime_error("Error inserting object " + std::to_string(i) + ": " + error_msg);
        }

        if ((i + 1) % 100000 == 0) {
            std::cout << "Inserted " << (i + 1) << " vectors" << std::endl;
        }
    }

    // Save the index after insertion
    qbg_save_index(index, err);
    qbg_close_index(index);

    // Set build parameters
    QBGBuildParameters bparam;
    qbg_initialize_build_parameters(&bparam);

    bparam.number_of_first_objects = params.num_first_objects;
    bparam.number_of_second_objects = params.num_second_objects;
    bparam.number_of_first_clusters = params.num_first_clusters;
    bparam.number_of_second_clusters = params.num_second_clusters;
    bparam.number_of_third_clusters = params.num_third_clusters;
    bparam.number_of_objects = params.sample_size;
    bparam.rotation = params.enable_rotation;

    std::cout << "Building QBG index structure..." << std::endl;

    // Build index
    if (!qbg_build_index(index_path, &bparam, err)) {
        std::string error_msg = ngt_get_error_string(err);
        ngt_destroy_error_object(err);
        throw std::runtime_error("Error building QBG index: " + error_msg);
    }

    ngt_destroy_error_object(err);
    std::cout << "Index build complete" << std::endl;
}

std::pair<double, std::vector<std::vector<int>>>
benchmarkQBGSearch(QBGIndex index,
    const std::vector<std::vector<float>>& queries,
    float epsilon,
    float blob_epsilon)
{
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<int>> all_results;
    NGTError err = ngt_create_error_object();

    for (const auto& query : queries) {
        QBGQueryParameters qparam;
        qbg_initialize_query_parameters(&qparam);
        qparam.number_of_results = K;
        qparam.epsilon = epsilon;
        qparam.blob_epsilon = blob_epsilon;
        qparam.result_expansion = 3.0;
        qparam.number_of_explored_blobs = 256;

        QBGQueryFloat qf;
        qf.query = const_cast<float*>(query.data());
        qf.params = qparam;

        NGTObjectDistances results = ngt_create_empty_results(err);

        if (!qbg_search_index_float(index, qf, results, err)) {
            std::string error_msg = ngt_get_error_string(err);
            qbg_destroy_results(results);
            ngt_destroy_error_object(err);
            throw std::runtime_error("QBG search failed: " + error_msg);
        }

        std::vector<int> query_results;
        size_t result_size = qbg_get_result_size(results, err);
        for (size_t i = 0; i < result_size; ++i) {
            NGTObjectDistance result = qbg_get_result(results, i, err);
            query_results.push_back(result.id - 1);
        }
        all_results.push_back(query_results);
        qbg_destroy_results(results);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double qps = static_cast<double>(queries.size()) / (duration.count() / 1000.0);

    ngt_destroy_error_object(err);
    return { qps, all_results };
}

double calculateRecall(const std::vector<std::vector<int>>& ground_truth,
    const std::vector<std::vector<int>>& results)
{
    double total_recall = 0.0;
    for (size_t i = 0; i < ground_truth.size(); ++i) {
        std::unordered_set<int> gt_set(ground_truth[i].begin(),
            ground_truth[i].begin() + std::min(K, ground_truth[i].size()));
        size_t found = 0;
        for (size_t j = 0; j < std::min(K, results[i].size()); ++j) {
            if (gt_set.count(results[i][j]) > 0) {
                found++;
            }
        }
        total_recall += static_cast<double>(found) / K;
    }
    return total_recall / ground_truth.size();
}

bool qbgIndexExists(const std::string& index_path)
{
    return std::filesystem::exists(index_path + "/grp") && std::filesystem::exists(index_path + "/obj") && std::filesystem::exists(index_path + "/prf");
}

std::pair<QBGIndex, bool> getOrCreateQBGIndex(
    const std::string& index_path,
    const std::vector<std::vector<float>>& data,
    const QBGParams& params)
{
    bool created = false;
    NGTError err = ngt_create_error_object();

    // If index doesn't exist, create it
    if (!qbgIndexExists(index_path)) {
        std::cout << "Creating new QBG index..." << std::endl;
        buildQBGIndex(data, index_path.c_str(), params);
        created = true;
    } else {
        std::cout << "Using existing QBG index from " << index_path << std::endl;
    }

    // Load the index
    QBGIndex index = qbg_open_index(index_path.c_str(), true, err);
    if (index == nullptr) {
        std::string error_msg = ngt_get_error_string(err);
        ngt_destroy_error_object(err);
        throw std::runtime_error("Error opening QBG index: " + error_msg);
    }

    ngt_destroy_error_object(err);
    return { index, created };
}

void processDataset(const std::string& dataset_name)
{
    std::cout << "\nProcessing dataset: " << dataset_name << std::endl;

    // Construct paths
    std::string base_path = std::string(data_dir) + dataset_name + "/";
    std::string index_path = std::string(index_dir) + "qbg-test/" + dataset_name;
    std::string result_path = std::string(result_dir) + "qbg-test/" + dataset_name;

    // Create directories
    std::filesystem::create_directories(result_path);

    try {
        // Read dataset files
        std::cout << "Reading dataset files..." << std::endl;
        auto base_data = readFvecs(base_path + dataset_name + "_base.fvecs");
        auto query_data = readFvecs(base_path + dataset_name + "_query.fvecs");
        auto ground_truth = readIvecs(base_path + dataset_name + "_groundtruth.ivecs");

        // Get appropriate parameters for this dataset
        auto params = getParamsForDataset(dataset_name, base_data.size(), base_data[0].size());

        std::cout << "Dataset size: " << base_data.size() << " vectors" << std::endl;
        std::cout << "Vector dimension: " << base_data[0].size() << std::endl;

        // Get or create index
        auto [index, was_created] = getOrCreateQBGIndex(index_path, base_data, params);

        if (was_created) {
            std::cout << "Successfully created new index" << std::endl;
        } else {
            std::cout << "Successfully loaded existing index" << std::endl;
        }

        // Test different parameter combinations
        std::ofstream result_file(result_path + "/" + dataset_name + "_results.csv");
        result_file << "Recall,QPS\n";

        // for (const auto& [epsilon, blob_epsilon] : params.query_params) {
        for (const auto& [blob_epsilon, epsilon] : params.query_params) {

            std::cout << "\nTesting with epsilon=" << epsilon
                      << ", blob_epsilon=" << blob_epsilon << std::endl;

            auto [qps, results] = benchmarkQBGSearch(index, query_data, epsilon, blob_epsilon);
            double recall = calculateRecall(ground_truth, results);

            std::cout << "Recall: " << recall << ", QPS: " << qps << std::endl;
            result_file << recall << "," << qps << "\n";
        }

        qbg_close_index(index);
        std::cout << "Results written to " << result_path + "/" + dataset_name + "_results.csv" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error processing dataset " << dataset_name << ": " << e.what() << std::endl;
    }
}

int main(int argc, char** argv)
{
    // const std::vector<std::string> DATASETS = {
    //     "siftsmall", "sift", "deep", "audio", "notre", "sun", "ukbench" // Small/Medium dimension (128-512)
    //     "glove", "gist", "imagenet", // Medium/Large datasets
    //     "enron", "trevi", // High dimension (>1000)
    //     "nuswide", "mnist", "cifar", // Small datasets
    //     "wikipedia-2024-06-bge-m3-zh" // Very large dataset
    // };

    const std::vector<std::string> DATASETS = {
        "siftsmall"
    };

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

    // Create necessary directories
    std::filesystem::create_directories(std::string(index_dir) + "qbg-test");
    std::filesystem::create_directories(std::string(result_dir) + "qbg-test");

    // Process each dataset
    for (const auto& dataset : datasets_to_process) {
        std::cout << "\n=========================================" << std::endl;
        std::cout << "Processing dataset: " << dataset << std::endl;
        std::cout << "==========================================" << std::endl;

        processDataset(dataset);

        // Optional: Force garbage collection and give system time to free memory
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    // Summarize results
    std::cout << "\n=========================================" << std::endl;
    std::cout << "Test Summary" << std::endl;
    std::cout << "==========================================" << std::endl;

    for (const auto& dataset : datasets_to_process) {
        std::string result_file = std::string(result_dir) + "qbg-test/" + dataset + "/" + dataset + "_results.csv";
        if (std::filesystem::exists(result_file)) {
            std::cout << "\nDataset: " << dataset << std::endl;

            // Read and display best results
            std::ifstream results(result_file);
            std::string line;
            std::getline(results, line); // Skip header

            double best_recall = 0.0;
            double best_qps = 0.0;
            double best_balanced_score = 0.0;
            std::string best_params;

            while (std::getline(results, line)) {
                std::stringstream ss(line);
                std::string token;
                std::vector<double> values;

                while (std::getline(ss, token, ',')) {
                    values.push_back(std::stod(token));
                }

                if (values.size() >= 4) {
                    double epsilon = values[0];
                    double blob_epsilon = values[1];
                    double recall = values[2];
                    double qps = values[3];

                    // Track best recall and QPS
                    if (recall > best_recall) {
                        best_recall = recall;
                    }
                    if (qps > best_qps) {
                        best_qps = qps;
                    }

                    // Calculate balanced score (harmonic mean of recall and normalized QPS)
                    double balanced = 2 * recall * qps / (recall + qps);
                    if (balanced > best_balanced_score) {
                        best_balanced_score = balanced;
                        best_params = "epsilon=" + std::to_string(epsilon) + ", blob_epsilon=" + std::to_string(blob_epsilon);
                    }
                }
            }

            std::cout << "  Best Recall: " << best_recall << std::endl;
            std::cout << "  Best QPS: " << best_qps << std::endl;
            std::cout << "  Best balanced params: " << best_params << std::endl;
            std::cout << "  Best balanced score: " << best_balanced_score << std::endl;
        }
    }

    return 0;
}