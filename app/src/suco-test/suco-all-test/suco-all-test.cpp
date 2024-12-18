#include "config.h"
#include "SuCo/src/dist_calculation.h"
#include "SuCo/src/index.h"
#include "SuCo/src/preprocess.h"
#include "SuCo/src/query.h"
#include "SuCo/src/utils.h"

#include <algorithm>
#include <armadillo> // Re-introduce Armadillo since gen_indexes and transfer_data rely on it
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>
#include <unordered_map>
#include <vector>

const size_t K = 10; // Number of nearest neighbors to retrieve

const std::vector<std::string> DATASETS = {
    "audio", "cifar", "deep", "enron", "gist", "glove",
    "imagenet", "millionsong", "mnist", "notre", "nuswide",
    "sift", "siftsmall", "sun", "trevi", "ukbench",
    "wikipedia-2024-06-bge-m3-zh"
};

struct SuCoParams {
    float collision_ratio;
    float candidate_ratio;
};

// Function to generate parameter sets based on dataset size and dimensionality
std::vector<SuCoParams> getSuCoParams(size_t dataset_size, int dimension)
{
    std::vector<SuCoParams> params;

    if (dataset_size < 100000) {
        // Small datasets (<100K)
        params = {
            { 0.05f, 0.005f },
            { 0.04f, 0.004f },
            { 0.03f, 0.003f },
            { 0.025f, 0.0025f },
            { 0.02f, 0.002f },
            { 0.015f, 0.0015f },
            { 0.01f, 0.001f }
        };
    } else if (dataset_size < 1000000) {
        // Medium datasets (100K-1M)
        params = {
            { 0.03f, 0.003f },
            { 0.025f, 0.0025f },
            { 0.02f, 0.002f },
            { 0.015f, 0.0015f },
            { 0.01f, 0.001f },
            { 0.008f, 0.0008f },
            { 0.005f, 0.0005f },
            { 0.003f, 0.0003f }
        };
    } else {
        // Large datasets (>1M)
        params = {
            { 0.02f, 0.002f },
            { 0.015f, 0.0015f },
            { 0.01f, 0.001f },
            { 0.008f, 0.0008f },
            { 0.005f, 0.0005f },
            { 0.003f, 0.0003f },
            { 0.002f, 0.0002f },
            { 0.001f, 0.0001f }
        };
    }

    // Adjust for high-dimensional data
    if (dimension > 500) {
        std::vector<SuCoParams> high_dim_params = params;
        for (auto& p : high_dim_params) {
            p.collision_ratio *= 0.8f;
            p.candidate_ratio *= 0.8f;
        }
        params.insert(params.end(), high_dim_params.begin(), high_dim_params.end());
    }

    return params;
}

// Function to read fvecs files
bool readFvecsFile(const std::string& filename, std::vector<std::vector<float>>& data)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    try {
        while (true) {
            int32_t dim;
            if (!file.read(reinterpret_cast<char*>(&dim), sizeof(dim))) {
                if (file.eof())
                    break;
                throw std::runtime_error("Error reading dimension");
            }

            std::vector<float> vec(dim);
            if (!file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float))) {
                if (file.eof())
                    break;
                throw std::runtime_error("Error reading vector data");
            }

            data.push_back(std::move(vec));
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error reading file '" << filename << "': " << e.what() << std::endl;
        return false;
    }
}

// Function to read ivecs files
bool readIvecsFile(const std::string& filename, std::vector<std::vector<int>>& data)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    try {
        while (true) {
            int32_t dim;
            if (!file.read(reinterpret_cast<char*>(&dim), sizeof(dim))) {
                if (file.eof())
                    break;
                throw std::runtime_error("Error reading dimension");
            }

            std::vector<int> vec(dim);
            if (!file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(int))) {
                if (file.eof())
                    break;
                throw std::runtime_error("Error reading vector data");
            }

            data.push_back(std::move(vec));
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error reading file '" << filename << "': " << e.what() << std::endl;
        return false;
    }
}

// Function to calculate recall
double calculateRecall(const std::vector<std::vector<int>>& ground_truth,
    const std::vector<std::vector<int>>& results)
{
    if (ground_truth.empty() || results.empty())
        return 0.0;

    double total_recall = 0.0;
    size_t count = 0;

    for (size_t i = 0; i < ground_truth.size() && i < results.size(); ++i) {
        if (!ground_truth[i].empty() && !results[i].empty()) {
            for (size_t j = 0; j < K && j < results[i].size(); ++j) {
                if (std::find(ground_truth[i].begin(), ground_truth[i].end(),
                        results[i][j])
                    != ground_truth[i].end()) {
                    total_recall += 1.0;
                    break;
                }
            }
            count++;
        }
    }

    return count > 0 ? total_recall / count : 0.0;
}

// Function to process each dataset
void processDataset(const std::string& dataset_name)
{
    try {
        // Setup paths
        std::string base_path = std::string(data_dir) + dataset_name + "/";
        std::string index_base_path = std::string(index_dir) + "suco-test/" + dataset_name + "/";
        std::string result_base_path = std::string(result_dir) + "suco-test/" + dataset_name + "/";

        // Create necessary directories
        std::filesystem::create_directories(index_base_path);
        std::filesystem::create_directories(result_base_path);

        // Define file paths
        std::string base_file = base_path + dataset_name + "_base.fvecs";
        std::string query_file = base_path + dataset_name + "_query.fvecs";
        std::string groundtruth_file = base_path + dataset_name + "_groundtruth.ivecs";
        std::string result_file = result_base_path + "/" + dataset_name + "_recall_qps_results.csv";

        std::cout << "\nProcessing dataset: " << dataset_name << std::endl;

        // Load data
        std::vector<std::vector<float>> dataset, query_data;
        std::vector<std::vector<int>> ground_truth;

        if (!readFvecsFile(base_file, dataset) || !readFvecsFile(query_file, query_data) || !readIvecsFile(groundtruth_file, ground_truth)) {
            throw std::runtime_error("Failed to read input files");
        }

        // Validate dataset
        size_t dataset_size = dataset.size();
        if (dataset_size == 0) {
            throw std::runtime_error("Dataset is empty");
        }
        int dimension = (int)dataset[0].size();

        // Generate query parameter sets
        auto query_param_sets = getSuCoParams(dataset_size, dimension);
        if (query_param_sets.empty()) {
            throw std::runtime_error("No parameter sets generated");
        }

        // Define fixed indexing parameters
        int fixed_subspace_num = 8;
        int fixed_kmeans_centroid = 50;
        if (dimension > 500) {
            fixed_subspace_num = std::max(8, dimension / 64);
        }

        std::cout << "Using fixed indexing parameters: "
                  << "SubspaceNum=" << fixed_subspace_num
                  << ", KMeansCentroid=" << fixed_kmeans_centroid
                  << std::endl;

        // Allocate memory for index data
        float** index_data = new float*[dataset_size];
        for (size_t i = 0; i < dataset_size; i++) {
            index_data[i] = new float[dimension];
            std::copy(dataset[i].begin(), dataset[i].end(), index_data[i]);
        }

        // Prepare data for indexing
        // transfer_data requires a vector<arma::mat>
        std::vector<arma::mat> data_list;
        transfer_data(index_data, data_list, dataset_size, fixed_subspace_num, dimension / fixed_subspace_num);

        std::vector<std::unordered_map<std::pair<int, int>, std::vector<int>, hash_pair>> indexes;
        long int index_time = 0;
        long int query_time = 0;

        float* centroids_list = new float[fixed_kmeans_centroid * (dimension / fixed_subspace_num / 2) * fixed_subspace_num * 2];
        int* assignments_list = new int[dataset_size * fixed_subspace_num * 2];

        // Build the index
        gen_indexes(data_list, indexes, dataset_size, centroids_list, assignments_list,
            dimension / fixed_subspace_num / 2, // kmeans_dim
            fixed_subspace_num,
            fixed_kmeans_centroid,
            2,
            index_time);

        // Serialize the index to disk for future use
        std::string current_index_path = index_base_path + "index.bin";
        std::ofstream index_file(current_index_path, std::ios::binary);
        if (index_file.is_open()) {
            // Serialize centroids and assignments
            index_file.write(reinterpret_cast<char*>(centroids_list),
                sizeof(float) * fixed_kmeans_centroid * (dimension / fixed_subspace_num / 2) * fixed_subspace_num * 2);
            index_file.write(reinterpret_cast<char*>(assignments_list),
                sizeof(int) * dataset_size * fixed_subspace_num * 2);
            index_file.close();
        } else {
            std::cerr << "Warning: Could not save index to " << current_index_path << std::endl;
        }

        // Prepare for writing results
        std::ofstream result_file_stream(result_file);
        if (!result_file_stream.is_open()) {
            throw std::runtime_error("Cannot open result file for writing");
        }

        // Write CSV header
        result_file_stream << "Recall,QPS\n";

        // Iterate over each parameter set
        for (const auto& params : query_param_sets) {
            std::cout << "Testing parameters: "
                      << "CollisionRatio=" << params.collision_ratio
                      << ", CandidateRatio=" << params.candidate_ratio
                      << std::endl;

            size_t query_size = query_data.size();
            float** querypoints = new float*[query_size];
            int** queryknn_results = new int*[query_size];
            long int** gt = new long int*[query_size];

            for (size_t i = 0; i < query_size; i++) {
                querypoints[i] = new float[dimension];
                std::copy(query_data[i].begin(), query_data[i].end(), querypoints[i]);

                queryknn_results[i] = new int[K];
                gt[i] = new long int[K];
                for (size_t j = 0; j < K; j++) {
                    if ((size_t)j < ground_truth[i].size()) {
                        gt[i][j] = ground_truth[i][j];
                    } else {
                        gt[i][j] = -1;
                    }
                }
            }

            int collision_num = static_cast<int>(params.collision_ratio * dataset_size);
            int candidate_num = static_cast<int>(params.candidate_ratio * dataset_size);
            int number_of_threads = (int)std::thread::hardware_concurrency() / 2;

            auto start = std::chrono::high_resolution_clock::now();

            ann_query(index_data, queryknn_results, dataset_size, dimension,
                query_size, K, querypoints, indexes, centroids_list,
                fixed_subspace_num, dimension / fixed_subspace_num,
                fixed_kmeans_centroid, dimension / fixed_subspace_num / 2,
                collision_num, candidate_num, number_of_threads, query_time);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            double qps = query_size / (duration.count() / 1000.0);
            std::vector<std::vector<int>> results(query_size, std::vector<int>(K));
            for (size_t i = 0; i < query_size; i++) {
                for (size_t j = 0; j < K; j++) {
                    results[i][j] = queryknn_results[i][j];
                }
            }
            double recall = calculateRecall(ground_truth, results);
            double avg_query_time_ms = (duration.count() * 1.0) / query_size;

            // Display results in the terminal
            std::cout << "Parameters: CollisionRatio=" << std::fixed << std::setprecision(4) << params.collision_ratio
                      << ", CandidateRatio=" << std::fixed << std::setprecision(4) << params.candidate_ratio
                      << " | Recall: " << std::fixed << std::setprecision(4) << recall
                      << ", QPS: " << std::fixed << std::setprecision(2) << qps
                      << std::endl;

            result_file_stream << recall << ","
                               << qps << "\n";

            // Cleanup query memory
            for (size_t i = 0; i < query_size; i++) {
                delete[] querypoints[i];
                delete[] queryknn_results[i];
                delete[] gt[i];
            }
            delete[] querypoints;
            delete[] queryknn_results;
            delete[] gt;
        }

        result_file_stream.close();

        // Cleanup index_data memory
        for (size_t i = 0; i < dataset_size; i++) {
            delete[] index_data[i];
        }
        delete[] index_data;

        // Cleanup centroids_list and assignments_list
        delete[] centroids_list;
        delete[] assignments_list;

        std::cout << "Waiting for memory cleanup..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));
    } catch (const std::exception& e) {
        std::cerr << "Error processing dataset " << dataset_name
                  << ": " << e.what() << std::endl;
    }
}

// Main function
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
