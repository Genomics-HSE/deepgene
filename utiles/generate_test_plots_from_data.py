#!/usr/bin/env python3
"""
Generate test plots using real data from result.json with new styling.
Standalone version that doesn't require torch.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay
import msprime

# Import limits from data_gen_np without importing the whole module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "data_gen_np", 
    Path(__file__).parent / "deepgene" / "data" / "data_gen_np.py"
)
dgn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dgn)

limits = dgn.limits

def dem2Ne(demography, limits=limits):
    """Convert demography to Ne trajectory"""
    debug = demography.debug()
    return debug.population_size_trajectory(limits)

def create_NE_plot(NE_pred, NE_true, dem, file_name):
    """Create Ne plot without stripes (new styling)"""
    fig, ax = plt.subplots()
    
    # No stripes - removed fill_between loop
    
    # Настройка осей и отображение графика
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.xlim(0, 100000)
    plt.ylim(0, 25000)
    
    if dem is not None: 
        Ne = dem2Ne(dem)
        plt.step(limits, Ne, where='post', label="True", color="red")
    
    if NE_true is not None:     
        # Handle -Infinity values
        NE_true_clean = [v if np.isfinite(v) and v > 0 else NE_true[0] for v in NE_true]
        plt.step(limits, NE_true_clean + [NE_true_clean[-1]], where='post', label="From data", color="blue")
    
    plt.step(limits, NE_pred + [NE_pred[-1]], where='post', label="NN", color="black", linestyle='-.', linewidth=1.5)
    
    plt.legend()
    plt.savefig(file_name, dpi=200, bbox_inches='tight')
    plt.close(fig)

def create_confusion_matrix_from_data(data, output_path):
    """Create confusion matrix with gray to blue colormap"""
    confusion_matrix = np.array(data['confusion_matrix'])
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor('lightgray')
    
    # Create custom colormap from gray to current color (blue)
    colors = ['lightgray', 'steelblue']
    n_bins = 100
    cmap_custom = LinearSegmentedColormap.from_list('gray_to_blue', colors, N=n_bins)
    
    disp.plot(ax=ax, include_values=False, cmap=cmap_custom)
    fig.savefig(str(output_path), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Confusion matrix generated: {output_path}")

def create_heatmap_from_data(data, output_path):
    """Create heatmap with reduced length, no purple, gray zeros"""
    # For heatmap, we need y_pred and y_true which aren't in result.json
    # We'll create a synthetic heatmap based on the model_dist probabilities
    # This simulates what the heatmap would look like with grouped regions
    
    # Create synthetic heatmap data: simulate probabilities across positions
    length = 5000  # Back to 5000
    n_classes = 32
    
    # Use model_dist as base probabilities
    model_dist = np.array(data['model_dist'])
    prior_dist = np.array(data['prior_dist'])
    prior_dist_normalized = prior_dist / (prior_dist.sum() + 1e-10)
    
    np.random.seed(42)
    
    # Create grouped regions (like real data from tree structure)
    # Real data has regions with similar values due to tree intervals
    y_pred = np.zeros((length, n_classes))
    y_true = np.zeros(length, dtype=int)
    
    # Create several regions with different characteristics
    region_sizes = [800, 600, 700, 900, 1000, 500, 500]  # Variable region sizes
    region_starts = np.cumsum([0] + region_sizes[:-1])
    
    for i, start in enumerate(region_starts):
        if start >= length:
            break
        end = min(start + region_sizes[i], length)
        region_len = end - start
        
        # Each region has a dominant class (like tree intervals)
        dominant_class = np.random.choice(n_classes, p=prior_dist_normalized)
        
        # Create probabilities: high for dominant class, lower for others
        for pos in range(start, end):
            # Base probabilities from model_dist
            probs = model_dist.copy()
            
            # Boost the dominant class for this region
            probs[dominant_class] *= np.random.uniform(2.0, 4.0)
            
            # Add some variation but keep structure
            probs *= np.random.uniform(0.8, 1.2, size=n_classes)
            probs = np.maximum(probs, 1e-6)  # Avoid zeros
            probs = probs / probs.sum()
            
            y_pred[pos] = probs
            y_true[pos] = dominant_class
        
        # Add some transitions between regions (smooth boundaries)
        if i < len(region_starts) - 1 and end < length:
            transition_len = min(50, (region_sizes[i+1] if i+1 < len(region_sizes) else 100))
            for t in range(transition_len):
                if end + t >= length:
                    break
                # Blend between current and next region
                alpha = t / transition_len
                next_dominant = np.random.choice(n_classes, p=prior_dist_normalized)
                probs = model_dist.copy()
                probs[dominant_class] *= (1 - alpha) * np.random.uniform(2.0, 4.0)
                probs[next_dominant] *= alpha * np.random.uniform(2.0, 4.0)
                probs *= np.random.uniform(0.8, 1.2, size=n_classes)
                probs = np.maximum(probs, 1e-6)
                probs = probs / probs.sum()
                y_pred[end + t] = probs
                y_true[end + t] = int(dominant_class * (1 - alpha) + next_dominant * alpha)
    
    # Create heatmap
    sns.set()
    plt.clf()
    
    sns.set_context("paper")
    sns.set(rc={'figure.figsize': (11.7, 8.27),
                'axes.labelsize': 20,
                'axes.titlesize': 20,
                'figure.dpi': 100})
    
    # Prepare data
    data_heatmap = np.transpose(y_pred)
    
    # Create custom colormap from gray to plasma color (without red)
    # Get plasma colormap colors, but exclude red tones (use only up to 0.8 to avoid red)
    try:
        plasma_cmap = matplotlib.colormaps.get_cmap('plasma')
    except AttributeError:
        plasma_cmap = plt.cm.get_cmap('plasma')
    # Create gradient from lightgray to plasma colors (without red - stop at 0.8)
    colors = ['lightgray', plasma_cmap(0.0), plasma_cmap(0.4), plasma_cmap(0.8)]
    n_bins = 256
    cmap_custom = LinearSegmentedColormap.from_list('gray_to_plasma_no_red', colors, N=n_bins)
    
    # Set axes background to gray for zero values
    fig, ax = plt.subplots(figsize=(11.7, 8.27))
    ax.set_facecolor('lightgray')
    
    # Use custom colormap (gray to plasma without red) - no need for mask, colormap handles it
    ax = sns.heatmap(data_heatmap, cmap=cmap_custom, yticklabels=5, xticklabels=1000, 
                     ax=ax, cbar_kws={'label': 'Probability'})
    
    # Plot true labels - red color
    ax.plot(y_true, c='red', linewidth=1.5)
    ax.set(xlabel='Позиция вдоль генома', ylabel='Интервал коалесенции')
    
    fig.savefig(str(output_path), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Heatmap generated: {output_path}")

def main():
    # Path to result.json
    json_path = Path("/Users/evgen/Desktop/newnewnew/deepgene_new_10_v11/test_all_v1/results/gru/gru_set2_ce/const/result.json")
    
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        return
    
    # Load data
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create output directory
    output_dir = Path("test_plots")
    output_dir.mkdir(exist_ok=True)
    
    print("\nGenerating plots with new styling...")
    print("=" * 60)
    
    # Generate Ne plot
    try:
        dem = dgn.get_const_demographcs()
        create_NE_plot(data['Ne_pred'], data['Ne_true'], dem, str(output_dir / "NE_test.png"))
        print(f"✓ Ne plot generated: {output_dir / 'NE_test.png'}")
    except Exception as e:
        print(f"✗ Ne plot failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate confusion matrix
    try:
        create_confusion_matrix_from_data(data, output_dir / "confusion_matrix_test.png")
    except Exception as e:
        print(f"✗ Confusion matrix failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate heatmap
    try:
        create_heatmap_from_data(data, output_dir / "heatmap_test.png")
    except Exception as e:
        print(f"✗ Heatmap failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)
    print("\nAll plots generated! Opening files...")
    
    # Open plots
    import subprocess
    plots = [
        output_dir / "NE_test.png",
        output_dir / "confusion_matrix_test.png",
        output_dir / "heatmap_test.png"
    ]
    
    for plot_path in plots:
        if plot_path.exists():
            subprocess.run(["open", str(plot_path)])
            print(f"  Opened: {plot_path}")

if __name__ == "__main__":
    main()
