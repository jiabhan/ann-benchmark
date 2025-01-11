import numpy as np
import h5py
import os
import time
import scann
from pathlib import Path
import argparse
import json
from typing import List, Dict, Tuple

# Constants for directories
DATA_DIR = "../../../dataset/"
INDEX_DIR = "../../index/"
RESULT_DIR = "../../result/"

# Constants
K = 10  # Number of nearest neighbors to retrieve

DATASETS = [
    "audio", 
    "cifar", 
    "deep", 
    "enron", 
    "gist", 
    "glove",
    "imagenet", 
    "millionsong", 
    "mnist", 
    "notre", 
    "nuswide",
    "sift", 
    "siftsmall", 
    "sun", 
    "trevi", 
    "ukbench",
    "wikipedia-2024-06-bge-m3-zh"
]

def read_fvecs(filename: str) -> np.ndarray:
    """Read float vectors from .fvecs file"""
    with open(filename, 'rb') as f:
        # Read dimension
        dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
        
        # Get file size to calculate number of vectors
        f.seek(0, 2)
        file_size = f.tell()
        num_vectors = file_size // ((dim + 1) * 4)
        
        # Read all data at once
        f.seek(0)
        data = np.frombuffer(f.read(), dtype=np.float32)
        
        # Reshape and return vectors without dimension values
        vectors = data.reshape(-1, dim + 1)[:, 1:]
        
        print(f"Read {num_vectors} vectors of dimension {dim} from {filename}")
        return vectors

def read_ivecs(filename: str) -> np.ndarray:
    """Read integer vectors from .ivecs file"""
    with open(filename, 'rb') as f:
        # Read dimension
        dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
        
        # Get file size to calculate number of vectors
        f.seek(0, 2)
        file_size = f.tell()
        num_vectors = file_size // ((dim + 1) * 4)
        
        # Read all data at once
        f.seek(0)
        data = np.frombuffer(f.read(), dtype=np.int32)
        
        # Reshape and return vectors without dimension values
        vectors = data.reshape(-1, dim + 1)[:, 1:]
        
        print(f"Read {num_vectors} vectors of dimension {dim} from {filename}")
        return vectors

class ScaNNQueryParam:
    def __init__(self, leaves_to_search: int, reorder_k: int):
        self.leaves_to_search = leaves_to_search
        self.reorder_k = reorder_k

class ScaNNParams:
    def __init__(self, n_leaves: int, avq_threshold: float, dims_per_block: int, metric: str):
        self.n_leaves = n_leaves
        self.avq_threshold = float('nan') if isinstance(avq_threshold, str) and avq_threshold == '.nan' else avq_threshold
        self.dims_per_block = dims_per_block
        self.metric = metric
        self.query_params: List[ScaNNQueryParam] = []

def get_scann_param_sets() -> List[ScaNNParams]:
    """Get parameter sets from config.yml"""
    params = []
    
    # scann1 - euclidean
    p1 = ScaNNParams(600, float('nan'), 2, "squared_l2")
    p1.query_params = [ScaNNQueryParam(l, r) for l, r in [
        (4, 40), (3, 30), (6, 60), (8, 74), (9, 78), (10, 82),
        (11, 85), (13, 100), (16, 120), (20, 140), (30, 180),
        (35, 240), (50, 360)
    ]]
    params.append(p1)
    
    # scann2 - euclidean
    p2 = ScaNNParams(2000, float('nan'), 4, "squared_l2")
    p2.query_params = [ScaNNQueryParam(l, r) for l, r in [
        (10, 100), (15, 140), (25, 160), (35, 190), (40, 200),
        (45, 220), (50, 240), (60, 250), (70, 300), (80, 400),
        (100, 500), (120, 600), (150, 800), (200, 900)
    ]]
    params.append(p2)
    
    # scann3 - euclidean
    p3 = ScaNNParams(100, float('nan'), 4, "squared_l2")
    p3.query_params = [ScaNNQueryParam(l, r) for l, r in [
        (2, 20), (3, 20), (3, 30), (4, 30), (5, 40), (8, 80)
    ]]
    params.append(p3)
    
    return params

class ScaNN:
    def __init__(self, params: ScaNNParams, dataset_name: str):
        self.n_leaves = params.n_leaves
        self.avq_threshold = params.avq_threshold
        self.dims_per_block = params.dims_per_block
        self.metric = params.metric
        self.query_params = params.query_params
        self.dataset_name = dataset_name
        self.index = None
        
        print(f"Dataset name: {dataset_name}")
        print(f"ScaNN: n_leaves={self.n_leaves}")
        print(f"ScaNN: avq_threshold={self.avq_threshold}")
        print(f"ScaNN: dims_per_block={self.dims_per_block}")
        print(f"ScaNN: metric={self.metric}")

    def fit(self, X: np.ndarray):
        print("ScaNN: start indexing...")
        if X.size == 0:
            raise RuntimeError("Input data is empty")

        print(f"ScaNN: # of data={len(X)}")
        print(f"ScaNN: dimensionality={X.shape[1]}")

        # Following the pattern from module.py
        self.index = scann.scann_ops_pybind.builder(X, K, self.metric).\
            tree(
                num_leaves=self.n_leaves,
                num_leaves_to_search=1,  # Default to 1, will be changed during search
                training_sample_size=len(X),
                spherical=False,
                quantize_centroids=True
            ).\
            score_ah(
                self.dims_per_block,  # Using dims_per_block from parameters
                anisotropic_quantization_threshold=self.avq_threshold  # Using avq_threshold from parameters
            ).\
            reorder(1).build()  # Default to 1, will be changed during search

    def set_query_arguments(self, param: ScaNNQueryParam):
        print(f"ScaNN: Setting leaves_to_search={param.leaves_to_search}, reorder_k={param.reorder_k}")
        self.current_leaves = param.leaves_to_search
        self.current_reorder = param.reorder_k

    def query(self, v: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
        neighbors, distances = self.index.search_batched(
            v.reshape(1, -1),
            final_num_neighbors=n,
            leaves_to_search=self.current_leaves,
            pre_reorder_num_neighbors=self.current_reorder
        )
        return neighbors[0], distances[0]

def compute_recall(neighbors: np.ndarray, true_neighbors: np.ndarray) -> float:
    """Compute recall score"""
    total = 0
    for gt_row, row in zip(true_neighbors, neighbors):
        total += len(set(row[:K]).intersection(set(gt_row[:K])))
    return total / (len(true_neighbors) * K)

def process_dataset(dataset_name: str):
    try:
        # Setup paths and create result directory
        base_path = Path(DATA_DIR) / dataset_name
        result_path = Path(RESULT_DIR) / "scann-test" / dataset_name
        result_file = result_path / f"{dataset_name}_recall_qps_result.csv"
        
        result_path.mkdir(parents=True, exist_ok=True)
        
        # Open result file
        with open(result_file, 'w') as f:
            f.write("Recall,QPS\n")
            
        # Load dataset files
        base_file = base_path / f"{dataset_name}_base.fvecs"
        query_file = base_path / f"{dataset_name}_query.fvecs"
        groundtruth_file = base_path / f"{dataset_name}_groundtruth.ivecs"

        if not all(f.exists() for f in [base_file, query_file, groundtruth_file]):
            raise FileNotFoundError(f"Missing required files for dataset {dataset_name}")

        base_data = read_fvecs(str(base_file))
        query_data = read_fvecs(str(query_file))
        ground_truth = read_ivecs(str(groundtruth_file))
            
        print(f"Dataset loaded:\n"
              f"Base objects: {len(base_data)}\n"
              f"Queries: {len(query_data)}\n"
              f"Ground truth: {len(ground_truth)}")

        # Process each parameter set
        for params in get_scann_param_sets():
            print(f"\nTesting parameter set:\n"
                  f"n_leaves={params.n_leaves}\n"
                  f"dims_per_block={params.dims_per_block}")
            
            try:
                index = ScaNN(params, dataset_name)
                index.fit(base_data)
                
                for qparam in params.query_params:
                    index.set_query_arguments(qparam)
                    
                    # Run queries and measure time
                    start = time.time()
                    all_neighbors = np.zeros((len(query_data), K), dtype=np.int32)
                    
                    for i, query in enumerate(query_data):
                        neighbors, _ = index.query(query, K)
                        all_neighbors[i] = neighbors
                        
                    end = time.time()
                    duration = end - start
                    qps = len(query_data) / duration
                    
                    # Calculate recall
                    recall = compute_recall(all_neighbors, ground_truth)
                    
                    # Log results
                    print(f"Results for leaves_to_search={qparam.leaves_to_search} "
                          f"reorder_k={qparam.reorder_k}:\n"
                          f"Recall={recall:.4f}\n"
                          f"QPS={qps:.1f}")
                    
                    # Write to CSV with all parameters
                    with open(result_file, 'a') as f:
                        f.write(f"{recall},{qps}\n")
                        
            except Exception as e:
                print(f"Error processing parameter set: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {str(e)}")

def main():
    print("ScaNN Benchmark Started")
    print("=====================")
    
    # Check if specific datasets are provided as command line arguments
    parser = argparse.ArgumentParser(description='ScaNN Benchmark')
    parser.add_argument('datasets', nargs='*', help='Specific datasets to process')
    args = parser.parse_args()
    
    datasets_to_process = args.datasets if args.datasets else DATASETS
    
    for dataset in datasets_to_process:
        print(f"\nProcessing dataset: {dataset}")
        print("----------------------------------------")
        
        process_dataset(dataset)
        
        # Give system time to free memory
        time.sleep(2)
        
    print("\nScaNN Benchmark Completed")
    print("========================")

if __name__ == "__main__":
    main()
