#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#include "config.h"
#include "HCNNG/hcnng_index.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "utils/beamSearch.h"
#include "utils/euclidian_point.h"
#include "utils/parse_results.h"
#include "utils/point_range.h"
#include "utils/types.h"

const size_t K = 10; // Number of nearest neighbors to retrieve

const std::vector<std::string> DATASETS = {
    "audio", "cifar", "deep", "enron", "gist", "glove",
    "imagenet", "millionsong", "mnist", "notre", "nuswide",
    "sift", "siftsmall", "sun", "trevi", "ukbench",
    "wikipedia-2024-06-bge-m3-zh"
};

struct HCNNGQueryParam {
    int beamSize;
    float cut;
    long limit;
    long degree_limit;
};

struct HCNNGParams {
    long cluster_size;
    long mst_deg;
    long num_clusters;
    std::vector<HCNNGQueryParam> query_params;
};

HCNNGParams getHCNNGParams(size_t dataset_size, int dimension)
{
    HCNNGParams p;

    // Base parameter selection based on dataset size
    if (dataset_size < 100000) {
        // Small datasets (<100K)
        p.cluster_size = std::min(1000L, static_cast<long>(dataset_size / 10));
        p.mst_deg = 3;
        p.num_clusters = std::min(30L, static_cast<long>(dataset_size / p.cluster_size));
        p.query_params = {
            { 8, 1.35, 5, 12 },
            { 10, 1.35, 6, 16 },
            { 15, 1.35, 10, 20 },
            { 20, 1.35, 15, 25 },
            { 25, 1.35, 20, 28 },
            { 30, 1.35, 25, 30 },
            { 40, 1.35, 30, 32 },
            { 50, 1.35, 35, 32 }
        };
    } else if (dataset_size < 1000000) {
        // Medium datasets (100K-1M)
        p.cluster_size = 1000;
        p.mst_deg = 3;
        p.num_clusters = 30;
        p.query_params = {
            { 10, 1.35, 8, 16 },
            { 12, 1.35, 10, 18 },
            { 15, 1.35, 15, 25 },
            { 20, 1.35, 20, 28 },
            { 30, 1.35, 30, 32 },
            { 40, 1.35, 35, 32 },
            { 50, 1.35, 40, 32 },
            { 70, 1.35, 45, 32 },
            { 100, 1.35, 50, 32 },
            { 150, 1.35, 60, 32 }
        };
    } else {
        // Large datasets (>1M)
        p.cluster_size = 1000;
        p.mst_deg = 3;
        p.num_clusters = 30;
        p.query_params = {
            { 15, 1.35, 10, 19 },
            { 20, 1.35, 15, 22 },
            { 25, 1.35, 20, 25 },
            { 30, 1.35, 25, 28 },
            { 40, 1.35, 30, 32 },
            { 50, 1.35, 35, 32 },
            { 70, 1.35, 40, 32 },
            { 100, 1.35, 45, 32 },
            { 150, 1.35, 50, 32 },
            { 200, 1.35, 60, 32 },
            { 300, 1.35, 70, 32 },
            { 500, 1.35, 80, 32 }
        };
    }

    // Adjust for high dimensional data
    if (dimension > 500) {
        p.cluster_size = std::min(static_cast<long>(p.cluster_size * 1.5),
            static_cast<long>(dataset_size / 5));
        for (auto& qp : p.query_params) {
            qp.beamSize = static_cast<int>(qp.beamSize * 1.2);
            qp.limit = static_cast<long>(qp.limit * 1.2);
            qp.degree_limit = std::min(32L, static_cast<long>(qp.degree_limit * 1.1));
        }
    }

    return p;
}

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

            if (dim <= 0 || dim > 100000) {
                throw std::runtime_error("Invalid dimension: " + std::to_string(dim));
            }

            std::vector<float> vec(dim);
            if (!file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float))) {
                if (file.eof())
                    break;
                throw std::runtime_error("Error reading vector data");
            }

            data.push_back(std::move(vec));
        }

        std::cout << "Successfully read " << data.size() << " vectors of dimension "
                  << (data.empty() ? 0 : data[0].size()) << " from " << filename << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error reading file " << filename << ": " << e.what() << std::endl;
        data.clear();
        return false;
    }
}

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

            if (dim <= 0 || dim > 100000) {
                throw std::runtime_error("Invalid dimension: " + std::to_string(dim));
            }

            std::vector<int> vec(dim);
            if (!file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(int))) {
                if (file.eof())
                    break;
                throw std::runtime_error("Error reading vector data");
            }

            data.push_back(std::move(vec));
        }

        std::cout << "Successfully read " << data.size() << " vectors of dimension "
                  << (data.empty() ? 0 : data[0].size()) << " from " << filename << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error reading file " << filename << ": " << e.what() << std::endl;
        data.clear();
        return false;
    }
}

bool buildIndex(const std::vector<std::vector<float>>& data,
    const std::string& index_path,
    const HCNNGParams& params)
{
    if (data.empty()) {
        std::cerr << "Error: Empty dataset provided" << std::endl;
        return false;
    }

    try {
        using Point = parlayANN::Euclidian_Point<float>;
        using Graph = parlayANN::Graph<unsigned int>;
        using indexType = unsigned int;

        std::cout << "Building index with parameters:" << std::endl
                  << "- Cluster size: " << params.cluster_size << std::endl
                  << "- MST degree: " << params.mst_deg << std::endl
                  << "- Number of clusters: " << params.num_clusters << std::endl;

        auto point_range = parlayANN::PointRange<Point>(data, Point::parameters(data[0].size()));
        long max_degree = params.mst_deg * params.num_clusters;
        Graph G(max_degree, data.size());

        parlayANN::hcnng_index<Point, decltype(point_range), indexType> index;
        index.build_index(G, point_range, params.num_clusters,
            params.cluster_size, params.mst_deg);

        G.save(const_cast<char*>(index_path.c_str()));

        std::cout << "Successfully built and saved index to " << index_path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error building index: " << e.what() << std::endl;
        return false;
    }
}

std::pair<double, std::vector<std::vector<int>>>
benchmarkSearch(const std::string& index_path,
    const std::vector<std::vector<float>>& queries,
    const std::vector<std::vector<float>>& base_data,
    const HCNNGQueryParam& query_param)
{
    try {
        using Point = parlayANN::Euclidian_Point<float>;
        using indexType = unsigned int;

        // Load the graph
        parlayANN::Graph<indexType> G(const_cast<char*>(index_path.c_str()));
        if (G.size() == 0) {
            throw std::runtime_error("Empty graph loaded");
        }

        // Create point ranges
        auto base_points = parlayANN::PointRange<Point>(base_data, Point::parameters(base_data[0].size()));
        auto query_points = parlayANN::PointRange<Point>(queries, Point::parameters(queries[0].size()));

        // Setup query parameters
        parlayANN::QueryParams QP(K, query_param.beamSize, query_param.cut,
            query_param.limit, query_param.degree_limit);

        // Create stats tracker
        parlayANN::stats<indexType> QueryStats(queries.size());

        // Start timing
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<int>> results(queries.size());

        // Perform searches
        for (size_t i = 0; i < queries.size(); i++) {
            std::mt19937 rng(i);
            std::uniform_int_distribution<long> dis(0, G.size() - 1);
            indexType start_point = dis(rng);

            parlay::sequence<indexType> starting_points = { start_point };
            auto query_results = parlayANN::beam_search(query_points[i], G, base_points,
                starting_points, QP);

            auto [beam_results, visited] = query_results.first;
            results[i].resize(K);
            for (size_t j = 0; j < K && j < beam_results.size(); j++) {
                results[i][j] = beam_results[j].first;
            }

            QueryStats.increment_visited(i, visited.size());
            QueryStats.increment_dist(i, query_results.second);
        }

        // Calculate QPS
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        double qps = static_cast<double>(queries.size()) / (duration.count() / 1000.0);

        return { qps, results };
    } catch (const std::exception& e) {
        std::cerr << "Error in search: " << e.what() << std::endl;
        return { 0.0, std::vector<std::vector<int>>() };
    }
}

double calculateRecall(const std::vector<std::vector<int>>& ground_truth,
    const std::vector<std::vector<int>>& results)
{
    if (ground_truth.empty() || results.empty()) {
        return 0.0;
    }

    double total_recall = 0.0;
    size_t count = 0;

    for (size_t i = 0; i < ground_truth.size() && i < results.size(); ++i) {
        if (!ground_truth[i].empty() && !results[i].empty()) {
            for (size_t j = 0; j < std::min(K, results[i].size()); ++j) {
                if (std::find(ground_truth[i].begin(),
                        ground_truth[i].end(),
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

void processDataset(const std::string& dataset_name)
{
    try {
        // Setup paths
        std::string base_path = std::string(data_dir) + dataset_name + "/";
        std::string index_path = std::string(index_dir) + "hcnng-test/" + dataset_name;
        std::string result_path = std::string(result_dir) + "hcnng-test/" + dataset_name;

        std::filesystem::create_directories(index_path);
        std::filesystem::create_directories(result_path);

        std::string base_file = base_path + dataset_name + "_base.fvecs";
        std::string query_file = base_path + dataset_name + "_query.fvecs";
        std::string groundtruth_file = base_path + dataset_name + "_groundtruth.ivecs";
        std::string result_file = result_path + "/" + dataset_name + "_recall_qps_result.csv";

        {
            std::cout << "\nProcessing dataset: " << dataset_name << std::endl;

            // Load data
            std::vector<std::vector<float>> index_data, query_data;
            std::vector<std::vector<int>> ground_truth;

            if (!readFvecsFile(base_file, index_data) || !readFvecsFile(query_file, query_data) || !readIvecsFile(groundtruth_file, ground_truth)) {
                throw std::runtime_error("Failed to read input files");
            }

            size_t dataset_size = index_data.size();
            int dimension = index_data[0].size();

            // Get single parameter set
            auto params = getHCNNGParams(dataset_size, dimension);

            // Build index path
            std::string current_index_path = index_path + "/index_" + std::to_string(params.cluster_size) + "_" + std::to_string(params.mst_deg) + "_" + std::to_string(params.num_clusters);

            // Build index if needed
            if (!std::filesystem::exists(current_index_path)) {
                if (!buildIndex(index_data, current_index_path, params)) {
                    throw std::runtime_error("Failed to build index");
                }
            }

            // Open results file
            std::ofstream result_file_stream(result_file);
            result_file_stream << "Recall,QPS\n";

            // Test each query parameter
            for (const auto& qparam : params.query_params) {
                auto [qps, search_results] = benchmarkSearch(current_index_path,
                    query_data, index_data, qparam);

                if (!search_results.empty()) {
                    double recall = calculateRecall(ground_truth, search_results);

                    // Write results
                    result_file_stream << recall << ","
                                       << qps << "\n";

                    std::cout << "BeamSize=" << qparam.beamSize
                              << ": Recall=" << recall
                              << ", QPS=" << qps << std::endl;
                }
            }

            result_file_stream.close();

            index_data.clear();
            index_data.shrink_to_fit();
            query_data.clear();
            query_data.shrink_to_fit();
            ground_truth.clear();
            ground_truth.shrink_to_fit();
        }

    } catch (const std::exception& e) {
        std::cerr << "Error processing dataset " << dataset_name
                  << ": " << e.what() << std::endl;
    }
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