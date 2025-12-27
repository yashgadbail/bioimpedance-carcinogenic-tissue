import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "data.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "notebooks", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_circle_parameters(R0, R_inf, X_max):
    """
    Calculate the center (h, k) and radius (r) of the Cole-Cole circle segment.
    The circle passes through (R_inf, 0), (R0, 0) and has a peak height X_max.
    
    Math:
    width = R0 - R_inf
    h = (R0 + R_inf) / 2
    k is derived from geometric constraints: (width/2)^2 + k^2 = (X_max - k)^2
    """
    width = R0 - R_inf
    if width <= 0 or X_max <= 0:
        return 0, 0, 0

    h = (R0 + R_inf) / 2
    
    # solving for k: (w/2)^2 + k^2 = X_max^2 - 2*X_max*k + k^2
    # 2*X_max*k = X_max^2 - (w/2)^2
    # k = (X_max^2 - (w/2)^2) / (2*X_max)
    k = (X_max**2 - (width/2)**2) / (2 * X_max)
    
    r = np.sqrt((width/2)**2 + k**2)
    
    return h, k, r

def generate_arc_points(h, k, r, R_inf, R0, num_points=100):
    """Generate x, y points for the circular arc segment above y=0."""
    if r <= 0: return np.array([]), np.array([])

    x = np.linspace(R_inf, R0, num_points)
    
    # (x-h)^2 + (y-k)^2 = r^2
    term = r**2 - (x - h)**2
    term[term < 0] = 0 # Avoid numerical noise
    y = k + np.sqrt(term)
    
    return x, y

def plot_cole_cole_comparison(df, y_true_labels, y_pred_labels, class_names, title_suffix=""):
    """
    Plot Cole-Cole arcs colored by True Class and Predicted Class side-by-side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=True, sharey=True)
    
    # Prepare data arrays
    # Assuming df has 'I0', 'DR', 'Max.IP'
    # R0 = I0
    # R_inf = I0 - DR
    # X_max = Max.IP
    
    # We will plot centroids for clarity, and maybe thin lines for all?
    # Plotting thousands of arcs is messy. Let's plot points for (R_center, X_max) 
    # and average arcs for each class.
    
    plot_types = [('Ground Truth', y_true_labels), ('Model Prediction', y_pred_labels)]
    
    unique_classes = sorted(list(set(y_true_labels) | set(y_pred_labels)))
    colors = sns.color_palette("bright", len(unique_classes))
    color_map = {cls: colors[i] for i, cls in enumerate(unique_classes)}

    for ax_idx, (title, labels) in enumerate(plot_types):
        ax = axes[ax_idx]
        
        # Plot Scatter of Peaks
        # X-coord of peak is 'h' (center real), Y-coord is 'Max.IP + k'? No, peak is X_max.
        # Wait, peak x is h? No, peak x is h ONLY if h is between R_inf and R0?
        # Typically h is exactly in the middle of R_inf and R0. Yes.
        
        for cls in unique_classes:
            mask = np.array(labels) == cls
            if not np.any(mask): continue
            
            subset = df[mask]
            
            # Scatter
            # Calculate h for all
            R0s = subset['I0'].values
            DRs = subset['DR'].values
            R_infs = R0s - DRs
            hs = (R0s + R_infs) / 2
            X_maxs = subset['Max.IP'].values
            
            ax.scatter(hs, X_maxs, label=f"{cls}", alpha=0.5, s=20, color=color_map[cls])
            
            # Plot Median Arc for the class
            med_R0 = np.median(R0s)
            med_X_max = np.median(X_maxs)
            med_R_inf = np.median(R_infs)
            
            h_med, k_med, r_med = calculate_circle_parameters(med_R0, med_R_inf, med_X_max)
            x_arc, y_arc = generate_arc_points(h_med, k_med, r_med, med_R_inf, med_R0)
            
            ax.plot(x_arc, y_arc, color=color_map[cls], linewidth=3, linestyle='--')

        ax.set_title(f'{title} {title_suffix}')
        ax.set_xlabel('Resistance (R) [Ohms]')
        ax.set_ylabel('Reactance (Xc) [Ohms]')
        ax.axhline(0, color='black', linewidth=1)
        ax.grid(True, alpha=0.3)
        if ax_idx == 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f'cole_cole_classification{title_suffix}.png')
    plt.savefig(save_path)
    print(f"Cole-Cole plot saved to {save_path}")

if __name__ == "__main__":
    # Test run
    df = pd.read_csv(DATA_PATH)
    # Dummy predictions for test
    plot_cole_cole_comparison(df, df['Class'], df['Class'], df['Class'].unique(), title_suffix="_test")
