#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <thread>
#include <vector>

#include "config.h"
#include <rnn-descent/IndexRNNDescent.h>

const size_t K = 10; // Number of nearest neighbors to retrieve

const std::vector<std::string> DATASETS = {
    "audio", "cifar", "deep", "enron", "gist", "glove",
    "imagenet", "millionsong", "mnist", "notre", "nuswide",
    "sift", "siftsmall", "sun", "trevi", "ukbench",
    "wikipedia-2024-06-bge-m3-zh"
};

struct RNNDescentQueryParam {
    int search_L;
    int K0;
};

struct RNNDescentParams {
    int S; // Initial size of memory allocation
    int R; // Maximum in-degree
    int T1; // Outer iteration count
    int T2; // Inner iteration count
    std::vector<RNNDescentQueryParam> query_params;
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

// std::vector<RNNDescentParams> getRNNDescentParamSets()
// {
//     std::vector<RNNDescentParams> params;
//
//     {
//         RNNDescentParams p;
//         p.S = 20;
//         p.R = 96;
//         p.T1 = 4;
//         p.T2 = 15;
//
//         p.query_params = {
//             { 1, 32 }, { 2, 32 }, { 4, 32 }, { 8, 32 }, { 16, 32 },
//             { 32, 32 }, { 64, 32 }, { 128, 32 }, { 256, 32 },
//             { 512, 32 }, { 1024, 32 }
//         };
//         params.push_back(p);
//     }
//
//     {
//         RNNDescentParams p;
//         p.S = 20;
//         p.R = 96;
//         p.T1 = 4;
//         p.T2 = 15;
//
//         p.query_params = {
//             { 64, 16 }, { 64, 32 }, { 64, 48 }, { 64, 64 }, { 64, 96 }
//         };
//         params.push_back(p);
//     }
//
//     return params;
// }

std::vector<RNNDescentParams> getRNNDescentParamSets(const std::string& dataset)
{
    std::vector<RNNDescentParams> params;

    if (dataset == "gist" || dataset == "wikipedia-2024-06-bge-m3-zh" || dataset == "trevi") {
        // High-Dimensional & Large Datasets
        RNNDescentParams p;
        p.S = 150;
        p.R = 100;
        p.T1 = 4;
        p.T2 = 10;
        p.query_params = {
            { 5, 16 }, { 10, 24 }, { 15, 32 }, { 20, 40 },
            { 25, 50 }, { 30, 60 }, { 35, 80 }, { 40, 100 },
            { 45, 120 }, { 50, 150 }, { 55, 180 }, { 60, 200 }
        };
        params.push_back(p);
    } else if (dataset == "deep" || dataset == "imagenet" || dataset == "millionsong" || dataset == "ukbench" || dataset == "glove") {
        // Medium-Dimensional & Large Datasets
        RNNDescentParams p;
        p.S = 100;
        p.R = 80;
        p.T1 = 4;
        p.T2 = 10;
        p.query_params = {
            { 5, 12 }, { 10, 16 }, { 15, 24 }, { 20, 32 },
            { 25, 40 }, { 30, 48 }, { 35, 60 }, { 40, 72 },
            { 45, 84 }, { 50, 96 }, { 55, 108 }, { 60, 120 }
        };
        params.push_back(p);
    } else if (dataset == "enron") {
        // High-Dimensional but Smaller Dataset
        RNNDescentParams p;
        p.S = 120;
        p.R = 90;
        p.T1 = 5;
        p.T2 = 12;
        p.query_params = {
            { 10, 20 }, { 15, 30 }, { 20, 40 },
            { 25, 50 }, { 30, 60 }, { 35, 70 },
            { 40, 80 }, { 45, 90 }, { 50, 100 },
            { 55, 110 }, { 60, 120 }
        };
        params.push_back(p);
    } else if (dataset == "sift" || dataset == "mnist" || dataset == "audio" || dataset == "notre" || dataset == "cifar" || dataset == "sun") {
        // Low-Dimensional or Smaller Datasets
        RNNDescentParams p;
        if (dataset == "mnist") {
            p.S = 90;
            p.R = 70;
            p.T1 = 4;
            p.T2 = 10;
            p.query_params = {
                { 10, 20 }, { 15, 30 }, { 20, 40 },
                { 25, 50 }, { 30, 60 }, { 35, 70 },
                { 40, 80 }, { 45, 90 }, { 50, 100 },
                { 55, 110 }, { 60, 120 }
            };
        } else {
            p.S = 80;
            p.R = 64;
            p.T1 = 3;
            p.T2 = 8;
            p.query_params = {
                { 5, 8 }, { 10, 16 }, { 15, 24 }, { 20, 32 },
                { 25, 40 }, { 30, 48 }, { 35, 56 }, { 40, 64 },
                { 45, 72 }, { 50, 80 }, { 55, 88 }, { 60, 96 },
                { 65, 104 }, { 70, 112 }, { 75, 120 }, { 80, 128 }
            };
        }
        params.push_back(p);
    } else {
        // Default Parameters for Other Datasets
        RNNDescentParams p;
        p.S = 100;
        p.R = 80;
        p.T1 = 4;
        p.T2 = 10;
        p.query_params = {
            { 10, 24 }, { 15, 36 }, { 20, 48 }, { 25, 60 },
            { 30, 72 }, { 35, 84 }, { 40, 96 }, { 45, 108 },
            { 50, 120 }
        };
        params.push_back(p);
    }

    return params;
}

std::unique_ptr<rnndescent::IndexRNNDescent> buildIndex(
    const std::vector<float>& data,
    const size_t d,
    const RNNDescentParams& params)
{
    auto index = std::make_unique<rnndescent::IndexRNNDescent>(d);

    // Set parameters
    index->rnndescent.S = params.S;
    index->rnndescent.R = params.R;
    index->rnndescent.T1 = params.T1;
    index->rnndescent.T2 = params.T2;
    index->verbose = true;

    std::cout << "Building index..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    index->add(data.size() / d, data.data());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    std::cout << "Build time: " << duration.count() << " seconds" << std::endl;

    return index;
}

std::pair<double, std::vector<std::vector<faiss::idx_t>>>
benchmarkSearch(rnndescent::IndexRNNDescent& index,
    const std::vector<float>& queries,
    const size_t nq,
    const size_t d,
    const RNNDescentQueryParam& query_param)
{
    std::cout << "Running search benchmark..." << std::endl;

    std::vector<std::vector<faiss::idx_t>> all_results;
    std::vector<float> distances(K * nq);
    std::vector<faiss::idx_t> indices(K * nq);

    index.rnndescent.search_L = query_param.search_L;
    index.rnndescent.K0 = query_param.K0;

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < nq; i++) {
        const float* query = queries.data() + i * d;
        float* dist = distances.data() + i * K;
        faiss::idx_t* idx = indices.data() + i * K;

        index.search(1, query, K, dist, idx);
        all_results.push_back(std::vector<faiss::idx_t>(idx, idx + K));
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double qps = static_cast<double>(nq) / (duration.count() / 1000.0);

    return { qps, all_results };
}

double calculateRecall(
    const std::vector<std::vector<int>>& ground_truth,
    const std::vector<std::vector<faiss::idx_t>>& results)
{
    double total_recall = 0.0;
    size_t nq = results.size();

    for (size_t i = 0; i < nq; i++) {
        for (size_t j = 0; j < std::min(K, results[i].size()); j++) {
            if (results[i][j] == ground_truth[i][0]) {
                total_recall += 1.0;
                break;
            }
        }
    }

    return total_recall / nq;
}

template<typename T>
std::vector<T> flatten(const std::vector<std::vector<T>>& vec2d)
{
    std::vector<T> vec1d;
    for (const auto& inner : vec2d) {
        vec1d.insert(vec1d.end(), inner.begin(), inner.end());
    }
    return vec1d;
}

std::vector<std::pair<double, double>> evaluateParameters(
    const std::string& dataset_name,
    const std::vector<RNNDescentParams>& param_sets)
{
    std::vector<std::pair<double, double>> results;

    std::string base_path = std::string(data_dir) + dataset_name + "/";
    std::string index_path = std::string(index_dir) + "rnndescent-test/" + dataset_name;
    std::string result_path = std::string(result_dir) + "rnndescent-test/" + dataset_name;
    std::string result_file = result_path + "/" + dataset_name + "_recall_qps_result.csv";

    std::filesystem::create_directories(index_path);
    std::filesystem::create_directories(result_path);

    std::string base_file = base_path + dataset_name + "_base.fvecs";
    std::string query_file = base_path + dataset_name + "_query.fvecs";
    std::string groundtruth_file = base_path + dataset_name + "_groundtruth.ivecs";

    auto base_data_2d = readFvecs(base_file);
    auto query_data_2d = readFvecs(query_file);
    auto ground_truth = readIvecs(groundtruth_file);

    auto base_data = flatten(base_data_2d);
    auto query_data = flatten(query_data_2d);

    size_t d = base_data_2d[0].size();
    size_t nb = base_data_2d.size();
    size_t nq = query_data_2d.size();

    std::cout << "Dataset: " << dataset_name << std::endl;
    std::cout << "Dimensions: " << d << std::endl;
    std::cout << "Base vectors: " << nb << std::endl;
    std::cout << "Query vectors: " << nq << std::endl;

    for (const auto& params : param_sets) {
        try {
            auto index = buildIndex(base_data, d, params);

            for (const auto& qparam : params.query_params) {
                auto [qps, search_results] = benchmarkSearch(*index, query_data, nq, d, qparam);
                double recall = calculateRecall(ground_truth, search_results);

                results.push_back({ recall, qps });

                std::cout << "Results: "
                          << "search_L=" << qparam.search_L
                          << " K0=" << qparam.K0
                          << " S=" << params.S
                          << " R=" << params.R
                          << " T1=" << params.T1
                          << " T2=" << params.T2
                          << " | Recall=" << recall
                          << " QPS=" << qps << std::endl;
            }

        } catch (const std::exception& e) {
            std::cerr << "Error processing parameters: " << e.what() << std::endl;
        }
    }

    return results;
}

void processDataset(const std::string& dataset_name)
{
    try {
        std::cout << "\nProcessing dataset: " << dataset_name << std::endl;

        auto rnn_descent_params = getRNNDescentParamSets(dataset_name);
        std::string result_path = std::string(result_dir) + "rnndescent-test/" + dataset_name;
        std::filesystem::create_directories(result_path);

        std::cout << "Evaluating RNN-Descent parameters..." << std::endl;
        auto results = evaluateParameters(dataset_name, rnn_descent_params);

        std::string result_file = result_path + "/" + dataset_name + "_recall_qps_result.csv";
        std::ofstream result_file_stream(result_file);
        result_file_stream << "Recall,QPS\n";
        for (const auto& [recall, qps] : results) {
            result_file_stream << recall << "," << qps << "\n";
        }

        std::cout << "Results written to " << result_file << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error processing dataset " << dataset_name << ": " << e.what() << std::endl;
    }
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
    }

    return 0;
}