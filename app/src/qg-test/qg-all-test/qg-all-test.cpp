#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <thread>
#include <vector>

#include "config.h" // Ensure config.h defines data_dir, index_dir, result_dir appropriately
#include "NGT/Index.h"
#include "NGT/NGTQ/Quantizer.h"

#include <NGT/NGTQ/QuantizedGraph.h>

const std::vector<std::string> DATASETS = {
    "audio",
    "cifar",
    "deep",
    "gist",
    "glove",
    "imagenet",
    "mnist",
    "notre",
    "sift",
    "siftsmall",
    "sun",
    "ukbench",
};


const size_t K = 10; // Number of nearest neighbors to retrieve

// Configuration params for QG from config.yml
struct QGQueryParam {
    float result_expansion;
    float epsilon;
};

struct QGParams {
    int edge { 100 };
    int outdegree { 64 };
    int indegree { 120 };
    int max_edge { 96 };
    float epsilon { 0.02 };
    int sample { 2000 };
    std::vector<QGQueryParam> query_params;
};

// Get default query parameters from config.yml
std::vector<QGQueryParam> getDefaultQueryParams()
{
    return {
        { 0.0, 0.9 }, { 0.0, 0.95 }, { 0.0, 0.98 }, { 0.0, 1.0 },
        { 1.2, 0.9 }, { 1.5, 0.9 }, { 2.0, 0.9 }, { 3.0, 0.9 },
        { 1.2, 0.95 }, { 1.5, 0.95 }, { 2.0, 0.95 }, { 3.0, 0.95 },
        { 1.2, 0.98 }, { 1.5, 0.98 }, { 2.0, 0.98 }, { 3.0, 0.98 },
        { 1.2, 1.0 }, { 1.5, 1.0 }, { 2.0, 1.0 }, { 3.0, 1.0 },
        { 5.0, 1.0 }, { 10.0, 1.0 }, { 20.0, 1.0 },
        { 1.2, 1.02 }, { 1.5, 1.02 }, { 2.0, 1.02 }, { 3.0, 1.02 },
        { 2.0, 1.04 }, { 3.0, 1.04 }, { 5.0, 1.04 }, { 8.0, 1.04 }
    };
}

// Get parameter sets from config.yml
std::vector<QGParams> getQGParamSets()
{
    std::vector<QGParams> params;

    // s2000-e0.02
    QGParams p1;
    p1.sample = 2000;
    p1.epsilon = 0.02;
    p1.query_params = getDefaultQueryParams();
    params.push_back(p1);

    // s20000-e0.1
    QGParams p2;
    p2.sample = 20000;
    p2.epsilon = 0.1;
    p2.query_params = getDefaultQueryParams();
    params.push_back(p2);

    // s4000-e0.02
    QGParams p3;
    p3.sample = 4000;
    p3.epsilon = 0.02;
    p3.query_params = getDefaultQueryParams();
    params.push_back(p3);

    // s4000-e0.04
    QGParams p4 = p3;
    p4.epsilon = 0.04;
    params.push_back(p4);

    // s4000-e0.06
    QGParams p5 = p3;
    p5.epsilon = 0.06;
    params.push_back(p5);

    // s4000-e0.08
    QGParams p6 = p3;
    p6.epsilon = 0.08;
    params.push_back(p6);

    return params;
}

class QG {
public:
    QG(std::string metric, std::string object_type, float epsilon, const QGParams& params, const std::string dataset_name)
    {
        edge_size_ = params.edge;
        outdegree_ = params.outdegree;
        indegree_ = params.indegree;
        max_edge_ = params.max_edge;
        object_type_ = object_type;
        epsilon_ = params.epsilon;
        sample_ = params.sample;
        query_params_ = params.query_params;
        dataset_name_ = dataset_name;

        std::cout << "Dataset name: " << dataset_name_ << std::endl;
        std::cout << "QG: edge_size=" << edge_size_ << std::endl;
        std::cout << "QG: outdegree=" << outdegree_ << std::endl;
        std::cout << "QG: indegree=" << indegree_ << std::endl;
        std::cout << "QG: max_edge=" << max_edge_ << std::endl;
        std::cout << "QG: epsilon=" << epsilon_ << std::endl;
        std::cout << "QG: metric=" << metric << std::endl;
        std::cout << "QG: object_type=" << object_type_ << std::endl;

        if (metric == "euclidean") {
            metric_ = NGT::Property::DistanceType::DistanceTypeL2;
        } else if (metric == "angular") {
            metric_ = NGT::Property::DistanceType::DistanceTypeAngle;
        } else {
            throw std::runtime_error("Invalid metric type");
        }
    }

    bool indexExists(const std::string& index_path, bool check_qg = false)
    {
        if (check_qg) {
            // Check for QG-specific directories/files
            return std::filesystem::exists(index_path + "/qg/grp") && std::filesystem::exists(index_path + "/qg/tre") && std::filesystem::exists(index_path + "/qg/obj");
        } else {
            // Check for base or optimized index directories/files
            return std::filesystem::exists(index_path + "/grp") && std::filesystem::exists(index_path + "/tre") && std::filesystem::exists(index_path + "/obj");
        }
    }

    void fit(const std::vector<std::vector<float>>& X)
    {
        std::cout << "QG: start indexing..." << std::endl;
        if (X.empty()) {
            throw std::runtime_error("Input data is empty");
        }
        size_t dim = X[0].size();
        std::cout << "QG: # of data=" << X.size() << std::endl;
        std::cout << "QG: dimensionality=" << dim << std::endl;

        // Construct base index path
        std::filesystem::path base_index_path = std::filesystem::path(index_dir) / "qg-test" / dataset_name_;
        std::filesystem::create_directories(base_index_path);

        // Construct unique index path using parameters
        std::stringstream ss;
        ss << "ONNG-s" << sample_ << "-e" << epsilon_ << "-"
           << edge_size_ << "-" << outdegree_ << "-" << indegree_;
        std::string index_subdir = ss.str();
        std::filesystem::path index_path_full = base_index_path / index_subdir;
        index_path_ = index_path_full.string();

        // Construct ANNG path
        std::filesystem::path anng_path = base_index_path / ("ANNG-" + std::to_string(edge_size_));

        std::cout << "QG: Base Index Path: " << base_index_path << std::endl;
        std::cout << "QG: Index Path: " << index_path_full << std::endl;
        std::cout << "QG: ANNG Path: " << anng_path << std::endl;

        // Create Base Index if it doesn't exist
        if (!indexExists(index_path_full.string()) && !indexExists(anng_path.string())) {
            std::cout << "QG: Creating base index..." << std::endl;
            createBaseIndex(anng_path.string(), dim, X);
        }

        // Create Optimized Index if it doesn't exist
        if (!indexExists(index_path_full.string())) {
            std::cout << "QG: Creating optimized index..." << std::endl;
            createOptimizedIndex(anng_path.string(), index_path_full.string());
        }

        // Construct QG path
        std::filesystem::path qg_path = index_path_full / "qg";

        // Check if QG exists
        if (indexExists(index_path_full.string(), true)) {
            std::cout << "QG: Quantized Graph already exists. Loading existing QG..." << std::endl;
            // No need to create QG, it will be loaded in openIndex()
        } else {
            // Create Quantized Index if it doesn't exist
            std::cout << "QG: Creating quantized index..." << std::endl;
            createQuantizedIndex(index_path_full.string());
        }

        // Open the index
        openIndex();
    }

    void set_query_arguments(const QGQueryParam& param)
    {
        std::cout << "QG: Setting query arguments - "
                  << "result_expansion=" << param.result_expansion
                  << ", epsilon=" << param.epsilon << std::endl;

        result_expansion_ = param.result_expansion;
        search_epsilon_ = param.epsilon - 1.0f;
    }

    std::vector<std::pair<int, float>> query(const std::vector<float>& v, size_t n)
    {
        NGTQG::SearchQuery sq(v);
        sq.setResults(&results_);
        sq.setSize(n);
        sq.setEpsilon(search_epsilon_);
        sq.setResultExpansion(result_expansion_);

        index_->search(sq);

        std::vector<std::pair<int, float>> query_results;
        query_results.reserve(results_.size());
        for (const auto& result : results_) {
            query_results.emplace_back(result.id - 1, result.distance);
        }

        return query_results;
    }

    const std::vector<QGQueryParam>& getQueryParams() const
    {
        return query_params_;
    }

private:
    void createBaseIndex(const std::string& path, size_t dim, const std::vector<std::vector<float>>& data)
    {
        std::cout << "QG: Creating ANNG at " << path << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        // Construct the NGT create command
        std::stringstream cmd;
        cmd << "ngt create -it -p8 -b500 -ga -of -D2"
            << " -d " << dim
            << " -E " << edge_size_
            << " -S40"
            << " -e " << epsilon_
            << " -P0 -B30"
            << " " << path;

        std::cout << "QG: Executing command: " << cmd.str() << std::endl;
        int ret = system(cmd.str().c_str());
        if (ret != 0) {
            throw std::runtime_error("QG: Failed to create base index with command: " + cmd.str());
        }

        // Insert data using NGT API
        try {
            NGT::Index index(path);
            std::cout << "QG: Inserting " << data.size() << " objects..." << std::endl;
            for (const auto& v : data) {
                index.append(v); // Directly append the vector
            }

            std::cout << "QG: Creating index..." << std::endl;
            index.createIndex(31); // Using 31 as per original code
            index.save();

            // Verify that index has been created with data
            size_t num_objects = index.getNumberOfObjects();
            if (num_objects == 0) {
                throw std::runtime_error("QG: Index created but contains no objects");
            }

            std::cout << "QG: Base index created with " << num_objects << " objects." << std::endl;

        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("QG: Error during base index creation: ") + e.what());
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "QG: ANNG construction time(sec)="
                  << std::chrono::duration<float>(end - start).count() << std::endl;
    }

    void createOptimizedIndex(const std::string& src_path, const std::string& dst_path)
    {
        std::cout << "QG: Degree adjustment for index." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        // Construct the NGT reconstruct-graph command
        std::stringstream cmd;
        cmd << "ngt reconstruct-graph -mS"
            << " -E " << outdegree_
            << " -o " << outdegree_
            << " -i " << indegree_
            << " " << src_path
            << " " << dst_path;

        std::cout << "QG: Executing command: " << cmd.str() << std::endl;
        int ret = system(cmd.str().c_str());
        if (ret != 0) {
            throw std::runtime_error("QG: Failed to create optimized index with command: " + cmd.str());
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "QG: Degree adjustment time(sec)="
                  << std::chrono::duration<float>(end - start).count() << std::endl;
    }

    void createQuantizedIndex(const std::string& path)
    {
        std::cout << "QG: Creating and appending quantizer..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        // Define the QG path
        std::filesystem::path qg_path = std::filesystem::path(path) / "qg";

        // Ensure QG does not already exist
        if (std::filesystem::exists(qg_path)) {
            std::cout << "QG: Quantized Graph already exists at " << qg_path << ". Skipping creation." << std::endl;
            return; // Skip creation
        }

        // Create QG
        std::string cmd_create_qg = "qbg create-qg " + path;
        std::cout << "QG: Executing command: " << cmd_create_qg << std::endl;
        int ret = system(cmd_create_qg.c_str());
        if (ret != 0) {
            throw std::runtime_error("QG: Failed to create QG index with command: " + cmd_create_qg);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "QG: Create QG time(sec)="
                  << std::chrono::duration<float>(end - start).count() << std::endl;

        // Build QG
        std::cout << "QG: Building quantizer..." << std::endl;
        start = std::chrono::high_resolution_clock::now();

        std::stringstream build_cmd;
        build_cmd << "qbg build-qg"
                  << " -o " << sample_
                  << " -M6"
                  << " -ib"
                  << " -I400"
                  << " -Gz"
                  << " -Pn"
                  << " -E " << max_edge_
                  << " " << path;

        std::cout << "QG: Executing command: " << build_cmd.str() << std::endl;
        ret = system(build_cmd.str().c_str());
        if (ret != 0) {
            throw std::runtime_error("QG: Failed to build QG index with command: " + build_cmd.str());
        }

        end = std::chrono::high_resolution_clock::now();
        std::cout << "QG: Build QG time(sec)="
                  << std::chrono::duration<float>(end - start).count() << std::endl;
    }

    void openIndex()
    {
        std::filesystem::path qg_grp_path = std::filesystem::path(index_path_) / "qg" / "grp";
        if (std::filesystem::exists(qg_grp_path)) {
            auto start = std::chrono::high_resolution_clock::now();
            try {
                index_.reset(new NGTQG::Index(index_path_));
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("QG: Failed to open quantized graph index: ") + e.what());
            }
            auto end = std::chrono::high_resolution_clock::now();
            std::cout << "QG: Quantized Graph index successfully opened at " << index_path_ << std::endl;
            std::cout << "QG: Open time(sec)="
                      << std::chrono::duration<float>(end - start).count() << std::endl;
        } else {
            throw std::runtime_error("QG: Failed to find QG index at " + index_path_);
        }
    }


    std::unique_ptr<NGTQG::Index> index_;
    NGT::ObjectDistances results_;
    std::string index_path_;

    int edge_size_;
    int outdegree_;
    int indegree_;
    int max_edge_;
    NGT::Property::DistanceType metric_;
    std::string object_type_;
    float epsilon_;
    int sample_;
    std::vector<QGQueryParam> query_params_;
    std::string dataset_name_;

    float result_expansion_;
    float search_epsilon_;
};

// Function to read float vectors from a .fvecs file
std::vector<std::vector<float>> readFvecs(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    std::vector<std::vector<float>> data;
    int32_t dim;
    size_t vector_count = 0;
    while (file.read(reinterpret_cast<char*>(&dim), sizeof(dim))) {
        if (dim <= 0) {
            throw std::runtime_error("Invalid dimension (" + std::to_string(dim) + ") in file: " + filename);
        }
        std::vector<float> vec(dim);
        if (!file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float))) {
            throw std::runtime_error("Error reading vector data from file: " + filename);
        }
        data.push_back(vec);
        vector_count++;
    }

    std::cout << "readFvecs: Read " << vector_count << " vectors from " << filename << std::endl;
    return data;
}

// Function to read integer vectors from a .ivecs file
std::vector<std::vector<int>> readIvecs(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    std::vector<std::vector<int>> data;
    int32_t dim;
    size_t vector_count = 0;
    while (file.read(reinterpret_cast<char*>(&dim), sizeof(dim))) {
        if (dim <= 0) {
            throw std::runtime_error("Invalid dimension (" + std::to_string(dim) + ") in file: " + filename);
        }
        std::vector<int> vec(dim);
        if (!file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(int))) {
            throw std::runtime_error("Error reading int vector data from file: " + filename);
        }
        data.push_back(vec);
        vector_count++;
    }

    std::cout << "readIvecs: Read " << vector_count << " vectors from " << filename << std::endl;
    return data;
}

// Function to calculate recall
double calculateRecall(const std::vector<std::vector<int>>& ground_truth,
    const std::vector<std::vector<int>>& results)
{
    if (ground_truth.size() != results.size()) {
        std::cerr << "Warning: ground truth size (" << ground_truth.size()
                  << ") doesn't match results size (" << results.size() << ")" << std::endl;
        return 0.0;
    }

    double total_recall = 0.0;
    for (size_t i = 0; i < ground_truth.size(); ++i) {
        if (!ground_truth[i].empty() && !results[i].empty()) {
            size_t found = 0;
            for (size_t j = 0; j < std::min(K, results[i].size()); ++j) {
                if (std::find(ground_truth[i].begin(),
                        ground_truth[i].end(),
                        results[i][j])
                    != ground_truth[i].end()) {
                    found++;
                }
            }
            total_recall += static_cast<double>(found) / std::min(K, ground_truth[i].size());
        }
    }
    return total_recall / ground_truth.size();
}

// Function to process a single dataset
void processDataset(const std::string& dataset_name)
{
    try {
        // Construct paths using std::filesystem::path
        std::filesystem::path base_path = std::filesystem::path(data_dir) / dataset_name;
        std::filesystem::path index_path = std::filesystem::path(index_dir) / "qg-test" / dataset_name;
        std::filesystem::path result_path = std::filesystem::path(result_dir) / "qg-test" / dataset_name;
        std::filesystem::path result_file = result_path / (dataset_name + "_recall_qps_result.csv");

        // Create directories
        std::filesystem::create_directories(index_path);
        std::filesystem::create_directories(result_path);

        // Open result file
        std::ofstream result_stream(result_file);
        if (!result_stream) {
            throw std::runtime_error("Failed to open result file: " + result_file.string());
        }
        result_stream << "Recall,QPS" << std::endl;

        std::cout << "\nProcessing dataset: " << dataset_name << std::endl;

        // Load dataset files
        std::cout << "Loading dataset files..." << std::endl;
        std::filesystem::path base_file = base_path / (dataset_name + "_base.fvecs");
        std::filesystem::path query_file = base_path / (dataset_name + "_query.fvecs");
        std::filesystem::path groundtruth_file = base_path / (dataset_name + "_groundtruth.ivecs");

        // Check if all required files exist
        if (!std::filesystem::exists(base_file) || !std::filesystem::exists(query_file) || !std::filesystem::exists(groundtruth_file)) {
            std::cerr << "Missing required files for dataset " << dataset_name << std::endl;
            return;
        }

        // Read data files
        auto base_data = readFvecs(base_file.string());
        auto query_data = readFvecs(query_file.string());
        auto ground_truth = readIvecs(groundtruth_file.string());

        std::cout << "Dataset loaded:"
                  << "\nBase objects: " << base_data.size()
                  << "\nQueries: " << query_data.size()
                  << "\nGround truth: " << ground_truth.size()
                  << std::endl;

        // Get parameter sets to test
        auto param_sets = getQGParamSets();

        // Process each parameter set
        for (const auto& params : param_sets) {
            std::cout << "\nTesting parameter set:"
                      << "\nedge=" << params.edge
                      << "\noutdegree=" << params.outdegree
                      << "\nindegree=" << params.indegree
                      << "\nmax_edge=" << params.max_edge
                      << "\nepsilon=" << params.epsilon
                      << "\nsample=" << params.sample
                      << std::endl;

            try {
                // Create QG instance
                QG qg("euclidean", "Float", params.epsilon, params, dataset_name);

                // Build or load index
                qg.fit(base_data);

                // Test each query parameter combination
                for (const auto& qparam : params.query_params) {
                    qg.set_query_arguments(qparam);

                    // Run queries and measure time
                    auto start = std::chrono::high_resolution_clock::now();
                    std::vector<std::vector<int>> all_results;
                    all_results.reserve(query_data.size());

                    for (const auto& query : query_data) {
                        auto results = qg.query(query, K);
                        std::vector<int> result_ids;
                        result_ids.reserve(results.size());
                        for (const auto& r : results) {
                            result_ids.push_back(r.first);
                        }
                        all_results.push_back(result_ids);
                    }

                    auto end = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration<double>(end - start);
                    double qps = query_data.size() / duration.count();

                    // Calculate recall
                    double recall = calculateRecall(ground_truth, all_results);

                    // Log results
                    std::cout << "Results for expansion=" << qparam.result_expansion
                              << " epsilon=" << qparam.epsilon
                              << ":\nRecall=" << recall
                              << "\nQPS=" << qps << std::endl;

                    // Write to CSV
                    result_stream << recall << ","
                                  << qps << std::endl;
                }

            } catch (const std::exception& e) {
                std::cerr << "Error processing parameter set: " << e.what() << std::endl;
                continue; // Proceed to the next parameter set
            }
        }

        result_stream.close();
        std::cout << "Results written to " << result_file << std::endl;

        // Clear data to free memory
        base_data.clear();
        base_data.shrink_to_fit();
        query_data.clear();
        query_data.shrink_to_fit();
        ground_truth.clear();
        ground_truth.shrink_to_fit();

    } catch (const std::exception& e) {
        std::cerr << "Error processing dataset " << dataset_name << ": " << e.what() << std::endl;
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
        // Process default datasets
        datasets_to_process = DATASETS;
    }

    std::cout << "QG Benchmark Started" << std::endl;
    std::cout << "===================" << std::endl;

    for (const auto& dataset : datasets_to_process) {
        try {
            std::cout << "\nProcessing dataset: " << dataset << std::endl;
            std::cout << "----------------------------------------" << std::endl;

            processDataset(dataset);

            // Give system time to free memory and handle cleanup
            std::this_thread::sleep_for(std::chrono::seconds(2));

        } catch (const std::exception& e) {
            std::cerr << "Fatal error processing dataset " << dataset << ": "
                      << e.what() << std::endl;
            std::cerr << "Continuing with next dataset..." << std::endl;
        }
    }

    std::cout << "\nQG Benchmark Completed" << std::endl;
    std::cout << "======================" << std::endl;

    return 0;
}
