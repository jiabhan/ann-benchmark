#include "config.h"
#include "SuCo/src/dist_calculation.h"
#include "SuCo/src/index.h"
#include "SuCo/src/preprocess.h"
#include "SuCo/src/query.h"
#include "SuCo/src/utils.h"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

// Base constants
const int K = 10; // Number of nearest neighbors
const int QUERY_SIZE = 100;
const int KMEANS_ITERS = 2;

// Parameter sets structure
struct ParameterSet {
    int subspace_num;
    int kmeans_centroid;
    float collision_ratio;
    float candidate_ratio;
    std::string name;
};

// Results structure
struct Result {
    std::string param_set;
    double recall;
    double qps;
    double index_time_ms;
    double avg_query_time_ms;
    size_t memory_usage_mb;
    bool success;
    std::string error_message;
};

struct FileStructure {
    std::string index_data_file;
    std::string query_data_file;
    std::string ground_truth_file;
    std::string index_path;
    std::string result_file_path;
};

struct DatasetInfo {
    int size;
    int dimension;
    int query_size;
    int k;
};

struct MemoryStructure {
    float** dataset = nullptr;
    float** querypoints = nullptr;
    long int** groundtruth = nullptr;
    int** query_results = nullptr;
    int* assignments_list = nullptr;
    float* centroids_list = nullptr;
    std::vector<std::unordered_map<std::pair<int, int>, std::vector<int>, hash_pair>> indexes;
    std::vector<arma::mat> data_list;
};

// Helper function to find nearest valid divisor
int findNearestDivisor(int n, int target)
{
    if (n <= 0 || target <= 0) {
        throw std::invalid_argument("Invalid input for findNearestDivisor");
    }

    int lower = target;
    int upper = target;

    while (lower > 0 || upper <= n) {
        if (lower > 0) {
            if (n % lower == 0)
                return lower;
            lower--;
        }
        if (upper <= n) {
            if (n % upper == 0)
                return upper;
            upper++;
        }
    }
    return 1;
}

std::vector<ParameterSet> getParameterSets(int dimension)
{
    if (dimension <= 0) {
        throw std::invalid_argument("Invalid dimension for parameter sets");
    }

    std::vector<ParameterSet> sets;

    // Adjust subspace numbers to be valid divisors of dimension
    int small_subspace = findNearestDivisor(dimension, 8);
    int medium_subspace = findNearestDivisor(dimension, 10);
    int large_subspace = findNearestDivisor(dimension, 12);

    // Small datasets (1M scale)
    sets.push_back({ small_subspace, 50, 0.05f, 0.005f, "small_default" });
    sets.push_back({ small_subspace, 50, 0.03f, 0.003f, "small_conservative" });
    sets.push_back({ medium_subspace, 50, 0.05f, 0.005f, "small_more_subspaces" });

    return sets; // Return just the small dataset parameters for testing
}

class MemoryManager {
    MemoryStructure& mem;
    const DatasetInfo& info;

public:
    MemoryManager(MemoryStructure& m, const DatasetInfo& i)
        : mem(m)
        , info(i)
    {
    }

    ~MemoryManager()
    {
        cleanupMemory();
    }

private:
    void cleanupMemory()
    {
        try {
            if (mem.dataset) {
                for (int i = 0; i < info.size; i++) {
                    delete[] mem.dataset[i];
                }
                delete[] mem.dataset;
                mem.dataset = nullptr;
            }
            if (mem.querypoints) {
                for (int i = 0; i < info.query_size; i++) {
                    delete[] mem.querypoints[i];
                }
                delete[] mem.querypoints;
                mem.querypoints = nullptr;
            }
            if (mem.groundtruth) {
                for (int i = 0; i < info.query_size; i++) {
                    delete[] mem.groundtruth[i];
                }
                delete[] mem.groundtruth;
                mem.groundtruth = nullptr;
            }
            if (mem.query_results) {
                for (int i = 0; i < info.query_size; i++) {
                    delete[] mem.query_results[i];
                }
                delete[] mem.query_results;
                mem.query_results = nullptr;
            }
            delete[] mem.assignments_list;
            delete[] mem.centroids_list;
            mem.assignments_list = nullptr;
            mem.centroids_list = nullptr;

            mem.indexes.clear();
            mem.data_list.clear();
        } catch (...) {
            // Log error but don't throw from destructor
            std::cerr << "Error during memory cleanup" << std::endl;
        }
    }
};

FileStructure initializePaths()
{
    FileStructure files;

    // Create directories if they don't exist
    std::filesystem::create_directories(std::string(index_dir) + "suco-test/siftsmall-test/");
    std::filesystem::create_directories(std::string(result_dir) + "suco-test/siftsmall-test/");

    files.index_data_file = std::string(data_dir) + "siftsmall/siftsmall_base.fvecs";
    files.query_data_file = std::string(data_dir) + "siftsmall/siftsmall_query.fvecs";
    files.ground_truth_file = std::string(data_dir) + "siftsmall/siftsmall_groundtruth.ivecs";
    files.index_path = std::string(index_dir) + "suco-test/siftsmall-test/siftsmall_index.bin";
    files.result_file_path = std::string(result_dir) + "suco-test/siftsmall-test/parameter_test_results.csv";

    // Validate paths exist
    if (!std::filesystem::exists(files.index_data_file)) {
        throw std::runtime_error("Index data file not found: " + files.index_data_file);
    }
    if (!std::filesystem::exists(files.query_data_file)) {
        throw std::runtime_error("Query data file not found: " + files.query_data_file);
    }
    if (!std::filesystem::exists(files.ground_truth_file)) {
        throw std::runtime_error("Ground truth file not found: " + files.ground_truth_file);
    }

    return files;
}

std::vector<std::vector<float>> readFvecs(const std::string& filename)
{
    std::vector<std::vector<float>> data;
    FILE* fp = fopen(filename.c_str(), "rb");
    if (!fp) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    try {
        while (true) {
            int d;
            if (fread(&d, sizeof(int), 1, fp) != 1) {
                break;
            }

            std::vector<float> vec(d);
            if (fread(vec.data(), sizeof(float), d, fp) != d) {
                break;
            }
            data.push_back(vec);
        }
    } catch (...) {
        fclose(fp);
        throw;
    }

    fclose(fp);
    return data;
}

std::vector<std::vector<long int>> readIvecs(const std::string& filename)
{
    std::vector<std::vector<long int>> data;
    FILE* fp = fopen(filename.c_str(), "rb");
    if (!fp) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    try {
        while (true) {
            int d;
            if (fread(&d, sizeof(int), 1, fp) != 1) {
                break;
            }

            std::vector<long int> vec(d);
            for (int i = 0; i < d; i++) {
                int val;
                if (fread(&val, sizeof(int), 1, fp) != 1) {
                    break;
                }
                vec[i] = val;
            }
            data.push_back(vec);
        }
    } catch (...) {
        fclose(fp);
        throw;
    }

    fclose(fp);
    return data;
}

DatasetInfo validateDatasets(const std::vector<std::vector<float>>& dataset,
    const std::vector<std::vector<float>>& queries,
    const std::vector<std::vector<long int>>& groundtruth)
{
    if (dataset.empty() || queries.empty() || groundtruth.empty()) {
        throw std::runtime_error("One or more datasets are empty");
    }

    DatasetInfo info;
    info.size = dataset.size();
    info.dimension = dataset[0].size();
    info.query_size = QUERY_SIZE;
    info.k = K;

    // Validate dimensions match
    for (const auto& vec : dataset) {
        if (vec.size() != info.dimension) {
            throw std::runtime_error("Inconsistent dimensions in dataset");
        }
    }

    for (const auto& vec : queries) {
        if (vec.size() != info.dimension) {
            throw std::runtime_error("Query dimensions don't match dataset");
        }
    }

    if (queries.size() < info.query_size) {
        throw std::runtime_error("Not enough query points available");
    }

    if (groundtruth.size() < info.query_size) {
        throw std::runtime_error("Not enough groundtruth data available");
    }

    return info;
}

void initializeMemory(MemoryStructure& mem,
    const DatasetInfo& info,
    const std::vector<std::vector<float>>& raw_data,
    const std::vector<std::vector<float>>& raw_queries,
    const std::vector<std::vector<long int>>& raw_groundtruth,
    const ParameterSet& params)
{
    try {
        // Calculate memory sizes
        size_t centroids_size = params.kmeans_centroid * (info.dimension / params.subspace_num / 2) * params.subspace_num * 2;
        size_t assignments_size = info.size * params.subspace_num * 2;

        if (centroids_size == 0 || assignments_size == 0) {
            throw std::runtime_error("Invalid memory allocation sizes calculated");
        }

        mem.dataset = new float*[info.size];
        mem.querypoints = new float*[info.query_size];
        mem.groundtruth = new long int*[info.query_size];
        mem.query_results = new int*[info.query_size];

        for (int i = 0; i < info.size; i++) {
            mem.dataset[i] = new float[info.dimension];
            memcpy(mem.dataset[i], raw_data[i].data(), info.dimension * sizeof(float));
        }

        for (int i = 0; i < info.query_size; i++) {
            mem.querypoints[i] = new float[info.dimension];
            mem.groundtruth[i] = new long int[info.k];
            mem.query_results[i] = new int[info.k];

            memcpy(mem.querypoints[i], raw_queries[i].data(), info.dimension * sizeof(float));
            memcpy(mem.groundtruth[i], raw_groundtruth[i].data(), info.k * sizeof(long int));
        }

        mem.assignments_list = new int[assignments_size];
        mem.centroids_list = new float[centroids_size];

    } catch (std::bad_alloc&) {
        throw std::runtime_error("Failed to allocate memory");
    } catch (...) {
        throw std::runtime_error("Unknown error during memory initialization");
    }
}

double evaluateResults(float** dataset, float** queries, int dim,
    int** results, long int** groundtruth,
    int query_size, int k)
{
    if (!dataset || !queries || !results || !groundtruth) {
        throw std::runtime_error("Null pointer in evaluateResults");
    }

    int hits = 0;
#pragma omp parallel for reduction(+ : hits)
    for (int i = 0; i < query_size; i++) {
        for (int j = 0; j < k; j++) {
            for (int g = 0; g < k; g++) {
                if (results[i][j] == groundtruth[i][g]) {
                    hits++;
                    break;
                }
            }
        }
    }
    return (double)hits / (query_size * k);
}

void writeResultsToCSV(const std::string& filepath, const std::vector<Result>& results)
{
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open results file for writing: " + filepath);
    }

    file << "Recall,QPS\n";

    for (const auto& result : results) {
        if (result.success) {
            file << std::fixed << std::setprecision(4)
                 << result.recall << ","
                 << result.qps << "\n";
        }
    }
}

Result runParameterTest(const ParameterSet& params,
    const DatasetInfo& info,
    const std::vector<std::vector<float>>& raw_data,
    const std::vector<std::vector<float>>& raw_queries,
    const std::vector<std::vector<long int>>& raw_groundtruth,
    const std::string& index_path)
{

    Result result;
    result.param_set = params.name;
    result.success = false;

    try {
        // Validate parameters
        if (params.subspace_num <= 0 || info.dimension % params.subspace_num != 0) {
            throw std::runtime_error("Invalid subspace_num - must be positive and evenly divide dimension");
        }

        if (info.dimension % (params.subspace_num * 2) != 0) {
            throw std::runtime_error("Dimension must be evenly divisible by (subspace_num * 2)");
        }

        std::cout << "\nTesting parameter set: " << params.name << std::endl;
        std::cout << "Dataset information:" << std::endl
                  << "Size: " << info.size << std::endl
                  << "Dimension: " << info.dimension << std::endl
                  << "Query size: " << info.query_size << std::endl
                  << "K: " << info.k << std::endl;

        MemoryStructure mem {};
        MemoryManager manager(mem, info);

        initializeMemory(mem, info, raw_data, raw_queries, raw_groundtruth, params);

        long int index_time = 0;
        size_t memory_before = getCurrentRSS();

        // Check if index file exists
        bool load_index = false;
        {
            std::ifstream index_check(index_path, std::ios::binary);
            load_index = index_check.good();
        }

        if (load_index) {
            std::cout << "Loading existing index from: " << index_path << std::endl;
            load_indexes(const_cast<char*>(index_path.c_str()),
                mem.indexes,
                mem.centroids_list,
                mem.assignments_list,
                info.size,
                info.dimension / params.subspace_num / 2,
                params.subspace_num,
                params.kmeans_centroid);
        } else {
            std::cout << "Creating new index..." << std::endl;

            // Transfer data and validate
            transfer_data(mem.dataset, mem.data_list, info.size, params.subspace_num,
                info.dimension / params.subspace_num);

            if (mem.data_list.empty()) {
                throw std::runtime_error("Failed to transfer data to subspaces");
            }

            // Generate indexes with error handling
            bool index_success = false;
            try {
                gen_indexes(mem.data_list, mem.indexes, info.size, mem.centroids_list,
                    mem.assignments_list, info.dimension / params.subspace_num / 2,
                    params.subspace_num, params.kmeans_centroid, KMEANS_ITERS, index_time);
                index_success = true;
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("Failed to generate indexes: ") + e.what());
            }

            if (index_success) {
                // Save the newly created index with error handling
                try {
                    save_indexes(const_cast<char*>(index_path.c_str()),
                        mem.centroids_list,
                        mem.assignments_list,
                        info.size,
                        info.dimension / params.subspace_num / 2,
                        params.subspace_num,
                        params.kmeans_centroid);
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Failed to save index: " << e.what() << std::endl;
                }
            }
        }

        size_t memory_after = getCurrentRSS();
        result.memory_usage_mb = (memory_after > memory_before) ? (memory_after - memory_before) / (1024 * 1024) : 0;

        // Query phase
        long int query_time = 0;
        auto start = std::chrono::high_resolution_clock::now();

        int collision_num = static_cast<int>(params.collision_ratio * info.size);
        int candidate_num = static_cast<int>(params.candidate_ratio * info.size);
        int num_threads = get_nprocs_conf() / 2;

        // Validate query parameters
        if (collision_num <= 0 || candidate_num <= 0) {
            throw std::runtime_error("Invalid collision or candidate numbers");
        }

        // Perform query with error handling
        try {
            ann_query(mem.dataset, mem.query_results, info.size, info.dimension,
                info.query_size, info.k, mem.querypoints, mem.indexes,
                mem.centroids_list, params.subspace_num, info.dimension / params.subspace_num,
                params.kmeans_centroid, info.dimension / params.subspace_num / 2,
                collision_num, candidate_num, num_threads, query_time);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Query failed: ") + e.what());
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Calculate metrics
        if (duration.count() > 0) {
            result.qps = info.query_size / (duration.count() / 1000.0);
        } else {
            result.qps = 0;
        }

        result.recall = evaluateResults(mem.dataset, mem.querypoints, info.dimension,
            mem.query_results, mem.groundtruth,
            info.query_size, info.k);

        result.index_time_ms = static_cast<double>(index_time) / 1000.0;
        result.avg_query_time_ms = static_cast<double>(query_time) / info.query_size / 1000.0;
        result.success = true;

        // Print results
        std::cout << "\nResults for " << params.name << ":" << std::endl
                  << "Recall@" << info.k << ": " << result.recall << std::endl
                  << "QPS: " << result.qps << std::endl
                  << "Index time: " << result.index_time_ms << "ms" << std::endl
                  << "Avg query time: " << result.avg_query_time_ms << "ms" << std::endl
                  << "Memory usage: " << result.memory_usage_mb << "MB" << std::endl;

    } catch (const std::exception& e) {
        result.error_message = e.what();
        std::cerr << "Error testing " << params.name << ": " << e.what() << std::endl;
    } catch (...) {
        result.error_message = "Unknown error occurred";
        std::cerr << "Unknown error testing " << params.name << std::endl;
    }

    return result;
}

int main()
{
    try {
        FileStructure files = initializePaths();

        std::cout << "Loading datasets..." << std::endl;
        auto raw_data = readFvecs(files.index_data_file);
        auto raw_queries = readFvecs(files.query_data_file);
        auto raw_groundtruth = readIvecs(files.ground_truth_file);

        DatasetInfo info = validateDatasets(raw_data, raw_queries, raw_groundtruth);
        auto parameter_sets = getParameterSets(info.dimension);
        std::vector<Result> results;

        for (const auto& params : parameter_sets) {
            std::string param_index_path = files.index_path + "." + params.name;
            Result result = runParameterTest(params, info, raw_data, raw_queries, raw_groundtruth, param_index_path);
            results.push_back(result);
        }

        writeResultsToCSV(files.result_file_path, results);
        std::cout << "\nResults have been written to: " << files.result_file_path << std::endl;

        int successful_tests = std::count_if(results.begin(), results.end(),
            [](const Result& r) { return r.success; });

        std::cout << "\nTest Summary:" << std::endl
                  << "Total parameter sets: " << results.size() << std::endl
                  << "Successful tests: " << successful_tests << std::endl
                  << "Failed tests: " << results.size() - successful_tests << std::endl;

        if (successful_tests == 0) {
            std::cerr << "Warning: No parameter sets completed successfully" << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown fatal error occurred" << std::endl;
        return 1;
    }

    return 0;
}