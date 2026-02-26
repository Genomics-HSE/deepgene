#!/usr/bin/env python3
"""
Test script to verify plot styling changes work correctly.
Generates the three types of plots with new styling.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import the plotting functions from mark_all.py
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Mock data for testing
def create_test_data():
    """Create mock data for testing plots"""
    # Mock limits (time intervals)
    limits = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 
              250, 300, 350, 400, 500, 600, 700, 800, 1000, 1200, 1400, 1600, 2000, 
              2500, 3000, 4000, 5000, 10000]
    
    # Mock Ne predictions and true values
    np.random.seed(42)
    NE_pred = [10000 + np.random.randn() * 1000 for _ in range(len(limits) - 1)]
    NE_true = [10000 + np.random.randn() * 500 for _ in range(len(limits) - 1)]
    
    # Mock heatmap data (probabilities for each position and class)
    y_pred = np.random.rand(5000, 32)  # 5000 positions, 32 classes
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)  # Normalize to probabilities
    y_true = np.random.randint(0, 32, size=5000)  # True class labels
    
    # Mock confusion matrix data
    full_predicted = np.random.randint(0, 32, size=10000)
    full_ground_true = np.random.randint(0, 32, size=10000)
    
    return limits, NE_pred, NE_true, y_pred, y_true, full_predicted, full_ground_true

def test_ne_plot():
    """Test Ne plot without stripes"""
    from mark_all import create_NE_plot, dem2Ne
    import msprime
    
    limits, NE_pred, NE_true, _, _, _, _ = create_test_data()
    
    # Create a simple demography for testing
    dem = msprime.Demography()
    dem.add_population(name="pop", initial_size=10000)
    
    output_dir = Path("test_plots")
    output_dir.mkdir(exist_ok=True)
    
    create_NE_plot(NE_pred, NE_true, dem, str(output_dir / "NE_test.png"))
    print(f"✓ Ne plot generated: {output_dir / 'NE_test.png'}")
    return output_dir / "NE_test.png"

def test_heatmap():
    """Test heatmap with reduced length, no purple, gray zeros"""
    from mark_all import create_heatmap
    
    _, _, _, y_pred, y_true, _, _ = create_test_data()
    
    output_dir = Path("test_plots")
    output_dir.mkdir(exist_ok=True)
    
    create_heatmap(y_pred, y_true, str(output_dir / "heatmap_test.png"), length=5000)
    print(f"✓ Heatmap generated: {output_dir / 'heatmap_test.png'}")
    return output_dir / "heatmap_test.png"

def test_confusion_matrix():
    """Test confusion matrix with gray background"""
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    _, _, _, _, _, full_predicted, full_ground_true = create_test_data()
    
    final_confusion_matrix = confusion_matrix(full_ground_true, full_predicted)
    disp = ConfusionMatrixDisplay(confusion_matrix=final_confusion_matrix)
    
    output_dir = Path("test_plots")
    output_dir.mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('lightgray')
    disp.plot(ax=ax, include_values=False)
    fig.savefig(str(output_dir / "confusion_matrix_test.png"))
    plt.close(fig)
    
    print(f"✓ Confusion matrix generated: {output_dir / 'confusion_matrix_test.png'}")
    return output_dir / "confusion_matrix_test.png"

if __name__ == "__main__":
    print("Testing plot styling changes...")
    print("=" * 50)
    
    try:
        ne_path = test_ne_plot()
        print()
    except Exception as e:
        print(f"✗ Ne plot failed: {e}")
        ne_path = None
    
    try:
        heatmap_path = test_heatmap()
        print()
    except Exception as e:
        print(f"✗ Heatmap failed: {e}")
        heatmap_path = None
    
    try:
        cm_path = test_confusion_matrix()
        print()
    except Exception as e:
        print(f"✗ Confusion matrix failed: {e}")
        cm_path = None
    
    print("=" * 50)
    print("Test complete!")
    
    # Open plots if they were generated
    if ne_path and ne_path.exists():
        import subprocess
        subprocess.run(["open", str(ne_path)])
    if heatmap_path and heatmap_path.exists():
        import subprocess
        subprocess.run(["open", str(heatmap_path)])
    if cm_path and cm_path.exists():
        import subprocess
        subprocess.run(["open", str(cm_path)])
