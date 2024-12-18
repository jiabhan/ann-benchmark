import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Get the script directory and set up paths
script_dir = os.getcwd()  # Should be in tools/
parent_dir = os.path.dirname(script_dir)  # Go up to parent
result_dir = os.path.join(parent_dir, 'result')  # Point to results/
plot_dir = os.path.join(result_dir, 'plot')  # Create plot directory path

# Create plot directory if it doesn't exist
os.makedirs(plot_dir, exist_ok=True)

print("Result directory:", result_dir)
print("Plot directory:", plot_dir)

# Algorithm display names mapping
algo_name = {
    'hnswlib': "HNSW (hnswlib)",
    'panng': "NGT-ANNG",
    'qg': "NGT-QG",
    'hcnng': "HCNNG",
    'nsg': "NSG",
    'onng': "ONNG",
    'rnndescent': "RNN-Descent",
    'suco': "SUCO"
}

def read_data(root_dir):
    data = {}
    for algo_dir in os.listdir(root_dir):
        if not algo_dir.endswith('-test'):
            continue
            
        algo_path = os.path.join(root_dir, algo_dir)
        if os.path.isdir(algo_path):
            algo_name = algo_dir.replace('-test', '')
            data[algo_name] = {}
            
            for dataset_dir in os.listdir(algo_path):
                dataset_path = os.path.join(algo_path, dataset_dir)
                if os.path.isdir(dataset_path):
                    csv_files = [
                        f"{dataset_dir}_recall_qps_result.csv",
                        f"{dataset_dir}_recall_qps_results.csv"
                    ]
                    
                    for csv_file in csv_files:
                        csv_path = os.path.join(dataset_path, csv_file)
                        if os.path.exists(csv_path):
                            df = pd.read_csv(csv_path)
                            if 'recall' in df.columns:
                                df = df.rename(columns={'recall': 'Recall', 'qps': 'QPS'})
                            data[algo_name][dataset_dir] = df
                            break
    return data

def remove_local_minima(df, window_size=3):
    """
    Remove local minima using a sliding window approach and ensure monotonic QPS decrease
    with increasing recall.
    
    Args:
        df: DataFrame with 'Recall' and 'QPS' columns
        window_size: Size of the window for smoothing (odd number)
    """
    df = df.sort_values('Recall').copy()
    
    # First pass: Remove clear local minima
    mask = np.ones(len(df), dtype=bool)
    half_window = window_size // 2
    
    for i in range(half_window, len(df) - half_window):
        window = df['QPS'].iloc[i-half_window:i+half_window+1]
        if df['QPS'].iloc[i] < max(window.iloc[:half_window]) and \
           df['QPS'].iloc[i] < max(window.iloc[half_window+1:]):
            mask[i] = False
    
    df = df[mask]
    
    # Second pass: Ensure monotonic decrease in QPS as recall increases
    final_points = []
    current_max_qps = float('-inf')
    
    for idx in range(len(df)-1, -1, -1):
        if df['QPS'].iloc[idx] > current_max_qps:
            final_points.append(idx)
            current_max_qps = df['QPS'].iloc[idx]
    
    return df.iloc[sorted(final_points)]

def keep_highest_qps(df):
    return df.groupby('Recall').agg({'QPS': 'max'}).reset_index()

def plot_dataset(data, dataset):
    plt.figure(figsize=(12, 8))

    styles = {
        'hnswlib': ('magenta', '-.'),
        'panng': ('cyan', '--'),
        'qg': ('navy', '-'),
        'hcnng': ('green', ':'),
        'nsg': ('red', '--'),
        'onng': ('orange', '-'),
        'rnndescent': ('purple', '-.'),
        'suco': ('brown', '--')
    }

    # Create two subplot regions
    gs = plt.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.1)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    
    for algo, algo_data in data.items():
        if dataset in algo_data:
            df = algo_data[dataset]
            df_highest = keep_highest_qps(df)
            df_filtered = remove_local_minima(df_highest, window_size=5)
            
            color, linestyle = styles.get(algo, ('gray', '-'))
            
            # Plot on both axes
            for ax in [ax1, ax2]:
                ax.plot(df_filtered['Recall'], df_filtered['QPS'],
                       label=algo_name.get(algo, algo),
                       color=color,
                       linestyle=linestyle,
                       marker='o',
                       markersize=4)

    # Configure main plot (0.0 to 1.0 recall)
    ax1.set_xscale('linear')
    ax1.set_yscale('log')
    ax1.set_xlim(0, 1.001)
    ax1.set_ylim(0.7, 5e4)
    ax1.grid(True, which='both', linestyle=':', linewidth=0.5)
    ax1.set_xticklabels([])  # Remove x-axis labels from top plot

    # Configure zoomed plot (0.95 to 1.0 recall)
    ax2.set_xscale('linear')
    ax2.set_yscale('log')
    ax2.set_xlim(0.95, 1.001)
    ax2.set_ylim(0.7, 5e4)
    ax2.grid(True, which='both', linestyle=':', linewidth=0.5)
    
    # Set labels and title
    ax2.set_xlabel('Recall')
    ax1.set_ylabel('Queries per second (1/s)')
    ax2.set_ylabel('QPS')
    plt.suptitle(f'Recall-QPS tradeoff - {dataset.capitalize()} dataset')
    
    # Add legend
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{dataset}.png'), 
                bbox_inches='tight', 
                dpi=300)
    plt.close()

# Read and plot the data
data = read_data(result_dir)

# Print the structure of the data to verify
for algo, datasets in data.items():
    print(f"Algorithm: {algo}")
    for dataset, df in datasets.items():
        print(f"  Dataset: {dataset}, Shape: {df.shape}")
    print()

# Plot for each dataset
datasets = set(dataset for algo_data in data.values() for dataset in algo_data.keys())
for dataset in sorted(datasets):
    print(f"Plotting {dataset}...")
    plot_dataset(data, dataset)
print("All plots have been saved to:", plot_dir)