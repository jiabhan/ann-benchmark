/**
 * This implementation is heavily based on faiss::IndexNNDescent.cpp
 * (https://github.com/facebookresearch/faiss/blob/main/faiss/IndexNNDescent.cpp)
 */

// -*- c++ -*-

#include <omp.h>
#include <rnn-descent/IndexRNNDescent.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <unordered_set>

#ifdef __SSE__
#endif

#include <faiss/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>

extern "C" {

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

int sgemm_(const char* transa, const char* transb, FINTEGER* m, FINTEGER* n,
    FINTEGER* k, const float* alpha, const float* a, FINTEGER* lda,
    const float* b, FINTEGER* ldb, float* beta, float* c, FINTEGER* ldc);
}

namespace rnndescent {

using namespace faiss;

using storage_idx_t = NNDescent::storage_idx_t;

/**************************************************************
 * add / search blocks of descriptors
 **************************************************************/

namespace {

/* Wrap the distance computer into one that negates the
   distances. This makes supporting INNER_PRODUCE search easier */

struct NegativeDistanceComputer : DistanceComputer {
    /// owned by this
    DistanceComputer* basedis;

    explicit NegativeDistanceComputer(DistanceComputer* basedis)
        : basedis(basedis)
    {
    }

    void set_query(const float* x) override { basedis->set_query(x); }

    /// compute distance of vector i to current query
    float operator()(idx_t i) override { return -(*basedis)(i); }

    /// compute distance between two stored vectors
    float symmetric_dis(idx_t i, idx_t j) override
    {
        return -basedis->symmetric_dis(i, j);
    }

    ~NegativeDistanceComputer() override { delete basedis; }
};

DistanceComputer* storage_distance_computer(const Index* storage)
{
    if (is_similarity_metric(storage->metric_type)) {
        return new NegativeDistanceComputer(storage->get_distance_computer());
    } else {
        return storage->get_distance_computer();
    }
}

} // namespace

/**************************************************************
 * IndexRNNDescent implementation
 **************************************************************/

IndexRNNDescent::IndexRNNDescent(int d, int K, MetricType metric)
    : Index(d, metric)
    , rnndescent(d)
    , own_fields(false)
    , storage(nullptr)
{
    // the default storage is IndexFlat
    storage = new IndexFlat(d, metric);
    own_fields = true;
}

IndexRNNDescent::IndexRNNDescent(Index* storage, int K)
    : Index(storage->d, storage->metric_type)
    , rnndescent(storage->d)
    , own_fields(false)
    , storage(storage)
{
}

IndexRNNDescent::~IndexRNNDescent()
{
    if (own_fields) {
        delete storage;
    }
}

void IndexRNNDescent::train(idx_t n, const float* x)
{
    FAISS_THROW_IF_NOT_MSG(storage,
        "Please use IndexNNDescentFlat (or variants) "
        "instead of IndexNNDescent directly");
    // nndescent structure does not require training
    storage->train(n, x);
    is_trained = true;
}

void IndexRNNDescent::search(idx_t n, const float* x, idx_t k, float* distances,
    idx_t* labels,
    const SearchParameters* params) const
{
    FAISS_THROW_IF_NOT_MSG(!params, "search params not supported for this index");
    FAISS_THROW_IF_NOT(storage);
    FAISS_THROW_IF_NOT_MSG(rnndescent.has_built, "Index is not built");
    FAISS_THROW_IF_NOT_MSG(k > 0, "k must be positive");
    FAISS_THROW_IF_NOT_MSG(n > 0, "n must be positive");
    FAISS_THROW_IF_NOT_MSG(x != nullptr, "query vector is null");
    FAISS_THROW_IF_NOT_MSG(distances != nullptr, "distances array is null");
    FAISS_THROW_IF_NOT_MSG(labels != nullptr, "labels array is null");

    // Validate graph structure
    FAISS_THROW_IF_NOT_MSG(!rnndescent.final_graph.empty(), "Graph is empty");
    FAISS_THROW_IF_NOT_MSG(!rnndescent.offsets.empty(), "Offsets array is empty");
    FAISS_THROW_IF_NOT_MSG(rnndescent.offsets.size() == ntotal + 1,
        "Invalid offsets array size");

    idx_t check_period = InterruptCallback::get_period_hint(d * rnndescent.search_L);

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

#pragma omp parallel
        {
            VisitedTable vt(ntotal);

            DistanceComputer* dis = storage_distance_computer(storage);
            ScopeDeleter1<DistanceComputer> del(dis);

#pragma omp for
            for (idx_t i = i0; i < i1; i++) {
                idx_t* idxi = labels + i * k;
                float* simi = distances + i * k;
                dis->set_query(x + i * d);

                rnndescent.search(*dis, k, idxi, simi, vt);
            }
        }
        InterruptCallback::check();
    }

    if (metric_type == METRIC_INNER_PRODUCT) {
        // we need to revert the negated distances
        for (size_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }
}

void IndexRNNDescent::add(idx_t n, const float* x)
{
    FAISS_THROW_IF_NOT_MSG(storage,
        "Please use IndexNNDescentFlat (or variants) "
        "instead of IndexNNDescent directly");
    FAISS_THROW_IF_NOT(is_trained);

    if (ntotal != 0) {
        fprintf(stderr,
            "WARNING NNDescent doest not support dynamic insertions,"
            "multiple insertions would lead to re-building the index");
    }

    storage->add(n, x);
    ntotal = storage->ntotal;

    DistanceComputer* dis = storage_distance_computer(storage);
    ScopeDeleter1<DistanceComputer> del(dis);
    rnndescent.build(*dis, ntotal, verbose);
}

void IndexRNNDescent::reset()
{
    rnndescent.reset();
    storage->reset();
    ntotal = 0;
}

void IndexRNNDescent::reconstruct(idx_t key, float* recons) const
{
    storage->reconstruct(key, recons);
}

// void IndexRNNDescent::write(faiss::IOWriter* writer) const
// {
//     // Write index header and version
//     uint32_t h = fourcc("rnd1");
//     uint32_t version = 1;
//     writer->operator()(&h, sizeof(h), 1);
//     writer->operator()(&version, sizeof(version), 1);
//
//     // Write index metadata
//     writer->operator()(&d, sizeof(d), 1);
//     writer->operator()(&ntotal, sizeof(ntotal), 1);
//     writer->operator()(&metric_type, sizeof(metric_type), 1);
//     writer->operator()(&is_trained, sizeof(is_trained), 1);
//     writer->operator()(&verbose, sizeof(verbose), 1);
//
//     // Write RNNDescent parameters
//     writer->operator()(&rnndescent.S, sizeof(rnndescent.S), 1);
//     writer->operator()(&rnndescent.R, sizeof(rnndescent.R), 1);
//     writer->operator()(&rnndescent.T1, sizeof(rnndescent.T1), 1);
//     writer->operator()(&rnndescent.T2, sizeof(rnndescent.T2), 1);
//     writer->operator()(&rnndescent.K0, sizeof(rnndescent.K0), 1);
//     writer->operator()(&rnndescent.search_L, sizeof(rnndescent.search_L), 1);
//     writer->operator()(&rnndescent.has_built, sizeof(rnndescent.has_built), 1);
//
//     // Write graph structure
//     size_t nneighbors = rnndescent.final_graph.size();
//     writer->operator()(&nneighbors, sizeof(nneighbors), 1);
//     writer->operator()(rnndescent.final_graph.data(), sizeof(int), nneighbors);
//
//     size_t noffsets = rnndescent.offsets.size();
//     writer->operator()(&noffsets, sizeof(noffsets), 1);
//     writer->operator()(rnndescent.offsets.data(), sizeof(int), noffsets);
//
//     // Write storage data
//     if (storage) {
//         // Write storage type indicator
//         int32_t storage_type = 0; // 0 for IndexFlat
//         writer->operator()(&storage_type, sizeof(storage_type), 1);
//
//         // Write raw vector data for IndexFlat
//         const float* xb = static_cast<const faiss::IndexFlat*>(storage)->get_xb();
//         writer->operator()(xb, sizeof(float), d * ntotal);
//     } else {
//         int32_t storage_type = -1;
//         writer->operator()(&storage_type, sizeof(storage_type), 1);
//     }
// }
//
// void IndexRNNDescent::read(faiss::IOReader* reader)
// {
//     try {
//         // Read and verify header
//         uint32_t h;
//         uint32_t version;
//         reader->operator()(&h, sizeof(h), 1);
//         reader->operator()(&version, sizeof(version), 1);
//
//         if (h != fourcc("rnd1")) {
//             throw std::runtime_error("Invalid index header");
//         }
//         if (version != 1) {
//             throw std::runtime_error("Unsupported index version");
//         }
//
//         // Read index metadata
//         reader->operator()(&d, sizeof(d), 1);
//         reader->operator()(&ntotal, sizeof(ntotal), 1);
//         reader->operator()(&metric_type, sizeof(metric_type), 1);
//         reader->operator()(&is_trained, sizeof(is_trained), 1);
//         reader->operator()(&verbose, sizeof(verbose), 1);
//
//         // Validate basic parameters
//         if (d <= 0 || ntotal < 0) {
//             throw std::runtime_error("Invalid dimension or ntotal");
//         }
//
//         // Read RNNDescent parameters
//         reader->operator()(&rnndescent.S, sizeof(rnndescent.S), 1);
//         reader->operator()(&rnndescent.R, sizeof(rnndescent.R), 1);
//         reader->operator()(&rnndescent.T1, sizeof(rnndescent.T1), 1);
//         reader->operator()(&rnndescent.T2, sizeof(rnndescent.T2), 1);
//         reader->operator()(&rnndescent.K0, sizeof(rnndescent.K0), 1);
//         reader->operator()(&rnndescent.search_L, sizeof(rnndescent.search_L), 1);
//         reader->operator()(&rnndescent.has_built, sizeof(rnndescent.has_built), 1);
//
//         // Validate RNNDescent parameters
//         if (rnndescent.S <= 0 || rnndescent.R <= 0 || rnndescent.T1 <= 0 || rnndescent.T2 <= 0) {
//             throw std::runtime_error("Invalid RNNDescent parameters");
//         }
//
//         // Additional validation after loading
//         if (rnndescent.K0 <= 0) {
//             rnndescent.K0 = 32;  // Set default K0 if invalid
//         }
//
//         // Verify the graph structure is valid for search
//         size_t max_edges = 0;
//         for (size_t i = 0; i < rnndescent.offsets.size() - 1; i++) {
//             size_t degree = rnndescent.offsets[i + 1] - rnndescent.offsets[i];
//             if (degree == 0) {
//                 throw std::runtime_error("Invalid graph: vertex with no edges found");
//             }
//             max_edges = std::max(max_edges, degree);
//         }
//
//         if (max_edges == 0) {
//             throw std::runtime_error("Invalid graph: no edges found");
//         }
//
//         // Read graph structure
//         size_t nneighbors;
//         reader->operator()(&nneighbors, sizeof(nneighbors), 1);
//         if (nneighbors > 1000000000) { // Sanity check
//             throw std::runtime_error("Invalid number of neighbors");
//         }
//
//         rnndescent.final_graph.resize(nneighbors);
//         reader->operator()(rnndescent.final_graph.data(), sizeof(int), nneighbors);
//
//         size_t noffsets;
//         reader->operator()(&noffsets, sizeof(noffsets), 1);
//         if (noffsets != ntotal + 1) {
//             throw std::runtime_error("Invalid number of offsets");
//         }
//
//         rnndescent.offsets.resize(noffsets);
//         reader->operator()(rnndescent.offsets.data(), sizeof(int), noffsets);
//
//         // Validate graph structure
//         for (const auto& neighbor : rnndescent.final_graph) {
//             if (neighbor < -1 || neighbor >= ntotal) {
//                 throw std::runtime_error("Invalid neighbor index in graph");
//             }
//         }
//
//         for (size_t i = 0; i < noffsets - 1; i++) {
//             if (rnndescent.offsets[i] > rnndescent.offsets[i + 1] || rnndescent.offsets[i] < 0 || rnndescent.offsets[i] >= nneighbors) {
//                 throw std::runtime_error("Invalid offset values");
//             }
//         }
//
//         // Read storage
//         int32_t storage_type;
//         reader->operator()(&storage_type, sizeof(storage_type), 1);
//
//         if (storage_type == 0) { // IndexFlat
//             // Create new storage with proper RAII
//             std::unique_ptr<faiss::IndexFlat> new_storage(
//                 new faiss::IndexFlat(d, metric_type));
//
//             // Read vector data
//             std::vector<float> xb(d * ntotal);
//             reader->operator()(xb.data(), sizeof(float), d * ntotal);
//
//             // Add vectors to storage
//             new_storage->add(ntotal, xb.data());
//
//             // Replace old storage
//             if (own_fields) {
//                 delete storage;
//             }
//             storage = new_storage.release();
//             own_fields = true;
//         } else if (storage_type != -1) {
//             throw std::runtime_error("Unsupported storage type");
//         }
//
//     } catch (const std::exception& e) {
//         // Clean up on error
//         if (own_fields) {
//             delete storage;
//             storage = nullptr;
//         }
//         rnndescent.final_graph.clear();
//         rnndescent.offsets.clear();
//         ntotal = 0;
//         rnndescent.has_built = false;
//
//         std::string error_msg = "Error loading index: ";
//         error_msg += e.what();
//         throw std::runtime_error(error_msg);
//     }
// }
//
// // Convenience methods remain the same
// void IndexRNNDescent::save(const std::string& fname) const
// {
//     faiss::FileIOWriter writer(fname.c_str());
//     write(&writer);
// }
//
// void IndexRNNDescent::load(const std::string& fname)
// {
//     faiss::FileIOReader reader(fname.c_str());
//     read(&reader);
// }

} // namespace rnndescent
