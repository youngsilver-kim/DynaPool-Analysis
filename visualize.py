# visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_pooling_heatmap(csv_path):
    """ Figure 1: 클래스별 Alpha 가중치 Heatmap """
    df = pd.read_csv(csv_path, index_index='class_label')
    
    plt.figure(figsize=(10, 18))
    # Y축은 클래스 이름(종류), X축은 4개의 풀링 브랜치
    sns.heatmap(df, annot=False, cmap='YlGnBu', cbar_kwas={'label': 'Gating Coefficient (Alpha)'})
    plt.title("Class-conditioned Adaptive Pooling Preference", fontsize=14)
    plt.xlabel("Pooling Mechanism Branch", fontsize=12)
    plt.ylabel("Class Label ID", fontsize=12)
    plt.tight_layout()
    plt.savefig("outputs/figure1_pooling_heatmap.png", dpi=300)
    plt.close()

def plot_branch_histogram(npy_alpha_path):
    """ Figure 2: 각 브랜치별 가중치 할당 분포 (Histogram) """
    alphas = np.load(npy_alpha_path) # [Samples, 4]
    branches = ['Average', 'Max', 'GeM', 'Attention']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.ravel()
    
    for i, branch in enumerate(branches):
        sns.histplot(alphas[:, i], bins=30, ax=axes[i], kde=True, color='skyblue')
        axes[i].set_title(f"{branch} Branch Usage Distribution")
        axes[i].set_xlabel("Alpha Value")
        axes[i].set_ylabel("Sample Count")
        
    plt.suptitle("Overall Pooling Branch Activation Histogram", fontsize=16)
    plt.tight_layout()
    plt.savefig("outputs/figure2_branch_histogram.png", dpi=300)
    plt.close()
