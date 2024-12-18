#include "config.h"
#include "hnswlib/hnswlib.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

const size_t K = 10; // Number of nearest neighbors to retrieve

// Define datasets to test
const std::vector<std::string> DATASETS = {
    "audio", "cifar", "deep", "enron", "gist", "glove",
    "imagenet", "millionsong", "mnist", "notre", "nuswide",
    "sift", "siftsmall", "sun", "trevi", "ukbench",
    "wikipedia-2024-06-bge-m3-zh"
};

// Define the parameters from ann-benchmarks
// const std::vector<int> M_VALUES = { 4, 8, 12, 16, 24, 36, 48, 64, 96 };
const std::vector<int> M_VALUES = { 8, 12, 16, 24 };
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
    return std::filesystem::exists(index_path);
}

std::vector<std::tuple<double, double>> evaluateParameters(
    const std::string& index_base_path,
    const std::vector<std::vector<float>>& data,
    const std::vector<std::vector<float>>& queries,
    const std::vector<std::vector<int>>& ground_truth)
{
    std::vector<std::tuple<double, double>> results;
    size_t dim = data[0].size();

    for (int M : M_VALUES) {
        for (int ef_construction : EF_CONSTRUCTION_VALUES) {
            std::string index_path = index_base_path + "_M" + std::to_string(M) + "_efc" + std::to_string(ef_construction) + ".bin";

            try {
                hnswlib::HierarchicalNSW<float>* index;
                hnswlib::L2Space space(dim);

                if (indexExists(index_path)) {
                    std::cout << "Loading index for M=" << M << ", ef_construction=" << ef_construction << std::endl;
                    index = new hnswlib::HierarchicalNSW<float>(&space, index_path, false);
                } else {
                    std::cout << "Creating new index for M=" << M << ", ef_construction=" << ef_construction << std::endl;
                    index = new hnswlib::HierarchicalNSW<float>(&space, data.size(), M, ef_construction);

                    unsigned int num_threads = std::thread::hardware_concurrency();
                    std::atomic<size_t> counter(0);
                    std::vector<std::thread> threads;

                    auto worker = [&]() {
                        size_t i;
                        while ((i = counter.fetch_add(1)) < data.size()) {
                            index->addPoint(data[i].data(), i);
                        }
                    };

                    for (unsigned int i = 0; i < num_threads; ++i) {
                        threads.emplace_back(worker);
                    }

                    for (auto& thread : threads) {
                        thread.join();
                    }

                    index->saveIndex(index_path);
                }

                for (int ef_search : EF_SEARCH_VALUES) {
                    auto [qps, search_results] = benchmarkSearch(index, queries, ef_search);
                    double recall = calculateRecall(ground_truth, search_results);
                    results.emplace_back(recall, qps);

                    std::cout << "HNSW Results: M=" << M
                              << " ef_construction=" << ef_construction
                              << " ef_search=" << ef_search
                              << " | Recall=" << recall
                              << " QPS=" << qps << std::endl;
                }

                delete index;

            } catch (const std::exception& e) {
                std::cerr << "Error during parameter sweep for M=" << M
                          << ", ef_construction=" << ef_construction
                          << ": " << e.what() << std::endl;
            }
        }
    }

    return results;
}

void processDataset(const std::string& dataset_name)
{
    std::string base_path = std::string(data_dir) + dataset_name + "/";
    std::string index_path = std::string(index_dir) + "hnswlib-test/" + dataset_name + "/" + dataset_name;
    std::string result_path = std::string(result_dir) + "hnswlib-test/" + dataset_name;
    std::string result_file = result_path + "/" + dataset_name + "_recall_qps_result.csv";

    std::filesystem::create_directories(result_path);

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

        {
            auto index_data = readFvecs(base_file);
            auto query_data = readFvecs(query_file);
            auto ground_truth = readIvecs(groundtruth_file);

            std::cout << "Evaluating HNSW parameters for " << dataset_name << "..." << std::endl;
            auto results = evaluateParameters(index_path, index_data, query_data, ground_truth);

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

    std::this_thread::sleep_for(std::chrono::seconds(2));
}

int main(int argc, char** argv)
{
    std::vector<std::string> datasets_to_process;

    if (argc > 1) {
        for (int i = 1; i < argc; i++) {
            datasets_to_process.push_back(argv[i]);
        }
    } else {
        datasets_to_process = DATASETS;
    }

    for (const auto& dataset : datasets_to_process) {
        processDataset(dataset);
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    return 0;
}