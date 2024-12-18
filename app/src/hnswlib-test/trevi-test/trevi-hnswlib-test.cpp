#include "config.h"
#include "hnswlib/hnswlib.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
#include <unordered_set>
#include <vector>


const size_t TREVI_DIM = 4096;
const size_t K = 10; // Number of nearest neighbors to retrieve
const size_t NUM_FOLDS = 10; // Number of folds for cross-validation

// Define the parameters from ann-benchmarks
const std::vector<int> M_VALUES = { 4, 8, 12, 16, 24, 36, 48, 64, 96 };
const std::vector<int> EF_CONSTRUCTION_VALUES = { 500 };
const std::vector<int> EF_SEARCH_VALUES = { 10, 20, 40, 80, 120, 200, 400, 600, 800 };

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

std::pair<double, std::vector<std::vector<int>>>
benchmarkSearch(hnswlib::HierarchicalNSW<float>* index, const std::vector<std::vector<float>>& queries, int ef_search)
{
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<int>> all_results;

    for (const auto& query : queries) {
        index->setEf(ef_search);
        auto result = index->searchKnn(query.data(), K);

        std::vector<int> query_results;
        while (!result.empty()) {
            query_results.push_back(result.top().second);
            result.pop();
        }
        std::reverse(query_results.begin(), query_results.end());
        all_results.push_back(query_results);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double qps = static_cast<double>(queries.size()) / (duration.count() / 1000.0);

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

bool indexExists(const std::string& index_path)
{
    return std::filesystem::exists(index_path);
}

std::vector<std::tuple<int, int, int, double, double>> runParameterSweepWithKFold(
    const std::string& index_base_path,
    const std::vector<std::vector<float>>& data,
    const std::vector<std::vector<float>>& queries,
    const std::vector<std::vector<int>>& ground_truth,
    size_t num_folds)
{
    std::vector<std::tuple<int, int, int, double, double>> all_results;

    size_t fold_size = queries.size() / num_folds;
    std::vector<size_t> indices(queries.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    for (int M : M_VALUES) {
        for (int ef_construction : EF_CONSTRUCTION_VALUES) {
            std::string index_path = index_base_path + "_M" + std::to_string(M) + "_efc" + std::to_string(ef_construction) + ".bin";

            try {
                hnswlib::HierarchicalNSW<float>* index;
                hnswlib::L2Space space(TREVI_DIM);

                if (indexExists(index_path)) {
                    std::cout << "Loading index for M=" << M << ", ef_construction=" << ef_construction << std::endl;
                    index = new hnswlib::HierarchicalNSW<float>(&space, index_path, false);
                } else {
                    std::cout << "Creating new index for M=" << M << ", ef_construction=" << ef_construction << std::endl;
                    index = new hnswlib::HierarchicalNSW<float>(&space, data.size(), M, ef_construction);

                    // Set the number of threads for internal parallelization
                    int num_threads = std::thread::hardware_concurrency(); // Use all available cores

                    // Multithreaded index building
                    std::atomic<size_t> counter(0);
                    std::vector<std::thread> threads;

                    auto worker = [&]() {
                        size_t i;
                        while ((i = counter.fetch_add(1)) < data.size()) {
                            index->addPoint(data[i].data(), i);
                        }
                    };

                    // Create and start threads
                    for (int i = 0; i < num_threads; ++i) {
                        threads.emplace_back(worker);
                    }

                    // Wait for all threads to finish
                    for (auto& thread : threads) {
                        thread.join();
                    }

                    index->saveIndex(index_path);
                }

                if (index == nullptr) {
                    throw std::runtime_error("Failed to load or create index");
                }

                for (int ef_search : EF_SEARCH_VALUES) {
                    std::vector<double> fold_recalls;
                    std::vector<double> fold_qps;

                    for (size_t fold = 0; fold < num_folds; ++fold) {
                        std::vector<std::vector<float>> test_queries;
                        std::vector<std::vector<int>> test_ground_truth;

                        for (size_t i = fold * fold_size; i < (fold + 1) * fold_size && i < queries.size(); ++i) {
                            test_queries.push_back(queries[indices[i]]);
                            test_ground_truth.push_back(ground_truth[indices[i]]);
                        }

                        auto [qps, search_results] = benchmarkSearch(index, test_queries, ef_search);
                        double recall = calculateRecall(test_ground_truth, search_results, K);

                        fold_recalls.push_back(recall);
                        fold_qps.push_back(qps);
                    }

                    double avg_recall = std::accumulate(fold_recalls.begin(), fold_recalls.end(), 0.0) / fold_recalls.size();
                    double avg_qps = std::accumulate(fold_qps.begin(), fold_qps.end(), 0.0) / fold_qps.size();

                    all_results.emplace_back(M, ef_construction, ef_search, avg_recall, avg_qps);

                    std::cout << "HNSW Params: M=" << M << ", ef_construction=" << ef_construction
                              << ", ef_search=" << ef_search
                              << " | Avg Recall: " << avg_recall << ", Avg QPS: " << avg_qps << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error during parameter sweep for M=" << M << ", ef_construction=" << ef_construction << ": " << e.what() << std::endl;
                std::cerr << "Skipping to next parameter set..." << std::endl;
            }
        }
    }

    return all_results;
}

int main()
{

        std::string index_data_file = std::string(data_dir) + "trevi/trevi_base.fvecs";
        std::string query_data_file = std::string(data_dir) + "trevi/trevi_query.fvecs";
        std::string ground_truth_file = std::string(data_dir) + "trevi/trevi_groundtruth.ivecs";
        std::string index_base_path = std::string(index_dir) + "hnswlib-test/trevi-test/trevi-test";
        std::string result_file_path = std::string(result_dir) + "hnswlib-test/trevi/trevi_recall_qps_result.csv";

        std::cout << "Reading data files..." << std::endl;
        std::vector<std::vector<float>> index_data = readFvecs(index_data_file);
        std::vector<std::vector<float>> query_data = readFvecs(query_data_file);
        std::vector<std::vector<int>> ground_truth = readIvecs(ground_truth_file);

        if (index_data.empty() || query_data.empty() || ground_truth.empty()) {
            std::cerr << "Error reading data files." << std::endl;
            return 1;
        }

        std::cout << "Performing parameter sweep with k-fold cross-validation for HNSW..." << std::endl;
        auto results = runParameterSweepWithKFold(index_base_path, index_data, query_data, ground_truth, NUM_FOLDS);

        std::ofstream result_file(result_file_path);
        // result_file << "M,ef_construction,ef_search,Recall,QPS\n";
        result_file << "Recall,QPS\n";
        for (const auto& [M, ef_construction, ef_search, recall, qps] : results) {
            // result_file << M << "," << ef_construction << "," << ef_search << "," << recall << "," << qps << "\n";
            result_file << recall << "," << qps << "\n";
        }
        result_file.close();

        std::cout << "Parameter sweep with k-fold cross-validation complete. Results written to " << result_file_path << std::endl;


    return 0;
}
