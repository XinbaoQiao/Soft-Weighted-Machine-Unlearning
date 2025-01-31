import os
import time
import argparse
import numpy as np

from dataset import fetch_data, DataTemplate
from eval import Evaluator
from model import LogisticRegression, NNLastLayerIF, MLPClassifier
from fair_fn import grad_ferm, grad_dp, loss_ferm, loss_dp
from utils import fix_seed

import json

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
import pandas as pd
import matplotlib.colors as mcolors

def parse_args():
    parser = argparse.ArgumentParser(description='Influence Fairness')
    parser.add_argument('--dataset', type=str, default="adult", help="name of the dataset")
    parser.add_argument('--num_data', type=int, default=200, help="number of data")
    parser.add_argument('--metric', type=str, default="dp", help="eop or dp")
    parser.add_argument('--seed', type=float, default=42, help="random seed")
    parser.add_argument('--save_model', type=str, default="n", help="y/n")
    parser.add_argument('--type', type=str, default="util", help="util/fair/")
    parser.add_argument('--strategy', type=str, default="dec", help="inc/dec/random")
    parser.add_argument('--points_to_delete', type=int, default=200, help="points to delete (num or index)")
    parser.add_argument('--random_seed', type=int, default=42, help="seed for random strategy")
    parser.add_argument('--only_pre', type=str, default="n", help="y/n")
    parser.add_argument('--model_type', type=str, default="logreg", help="logreg/nn/resnet")

    args = parser.parse_args()

    return args


# Remove outliers (using 3 standard deviations as threshold)
def Speedupdrawing(x, y):
    x_mean, x_std = np.mean(x), np.std(x)
    y_mean, y_std = np.mean(y), np.std(y)
    mask = (abs(x - x_mean) < 3 * x_std) & (abs(y - y_mean) < 3 * y_std)
    return x[mask], y[mask]

def RemoveExtremeOutlier(x, y):
    # Calculate z-scores for both x and y
    x_zscore = abs((x - np.mean(x)) / np.std(x))
    y_zscore = abs((y - np.mean(y)) / np.std(y))
    
    # Combine z-scores to find the most extreme point
    combined_zscore = x_zscore + y_zscore
    most_extreme_idx = np.argmax(combined_zscore)
    
    # Create a mask that keeps all points except the most extreme one
    mask = np.ones(len(x), dtype=bool)
    mask[most_extreme_idx] = False
    
    return x[mask], y[mask]

def calculate_diagonal_distance(x, y):
    # Normalize the points to [0,1] range for fair distance calculation
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
    y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    
    # Calculate distance to diagonal line y=x
    distance = np.abs(y_norm - x_norm) / np.sqrt(2)
    
    # Convert distance to alpha values (closer = more opaque)
    alpha = 1 - distance
    return alpha

if __name__ == "__main__":
    args = parse_args()

    args.save_model = 'y'
    args.metric = 'dp'
    args.seed = 1
    args.num_data = 30162

    print("Patience Is All You Need...")

    os.makedirs('results/SEC-5-1-1-1', exist_ok=True)

    # Load data
    fair_dp_actual = np.load('results/SEC-5-1-0-1/fair_dp_actual.npy')
    fair_dp_infl = np.load('results/SEC-5-1-0-1/fair_dp_infl.npy')
    # fair_eop_actual = np.load('results/SEC-5-1-0-1/fair_eop_actual.npy')
    # fair_eop_infl = np.load('results/SEC-5-1-0-1/fair_eop_infl.npy')
    util_actual = np.load('results/SEC-5-1-0-1/util_actual.npy')
    util_infl = np.load('results/SEC-5-1-0-1/util_infl.npy')
    robust_actual = np.load('results/SEC-5-1-0-1/robust_actual.npy')
    robust_infl = np.load('results/SEC-5-1-0-1/robust_infl.npy')
    
    # Calculate Pearson correlations
    pearson_util_fair, _ = pearsonr(util_actual, fair_dp_actual)
    pearson_util_robust, _ = pearsonr(util_actual, robust_actual)
    print(f"Pearson correlation between utility and fairness: {pearson_util_fair:.4f}")
    print(f"Pearson correlation between utility and robustness: {pearson_util_robust:.4f}")
    # Calculate Spearman correlations
    spearman_util_fair, _ = spearmanr(util_actual, fair_dp_actual)
    spearman_util_robust, _ = spearmanr(util_actual, robust_actual)
    print(f"Spearman correlation between utility and fairness: {spearman_util_fair:.4f}")
    print(f"Spearman correlation between utility and robustness: {spearman_util_robust:.4f}")

    # Remove outliers
    fair_dp_infl, fair_dp_actual = RemoveExtremeOutlier(fair_dp_infl, fair_dp_actual)
    util_infl, util_actual = RemoveExtremeOutlier(util_infl, util_actual)
    robust_infl, robust_actual = Speedupdrawing(robust_infl, robust_actual)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(32.5, 11))  # Horizontal layout
    plt.style.use('fast')  # Use fast style   

    # Set common parameters
    base_color = '#836796'  # Base color in hex
    
    # Convert hex to RGB
    rgb = mcolors.hex2color(base_color)

    # Utility
    alpha_util = calculate_diagonal_distance(util_infl, util_actual)
    colors_util = np.array([[rgb[0], rgb[1], rgb[2], a] for a in alpha_util])
    axes[0].scatter(util_infl, util_actual, c=colors_util, s=50, edgecolors='none')
    axes[0].plot([min(util_infl), max(util_infl)], [min(util_infl), max(util_infl)], 
                 color='silver', linestyle='--')
    axes[0].set_xlabel('Actual Utility Change', fontsize=40, fontweight='bold')
    axes[0].set_ylabel('Approximate Utility Change', fontsize=40, fontweight='bold')
    spearman_util, _ = spearmanr(util_infl, util_actual)
    pearson_util, _ = pearsonr(util_infl, util_actual)
    axes[0].text(0.05, 0.95, f'Spearman: {spearman_util:.2f}\nPearson: {pearson_util:.2f}', 
                 transform=axes[0].transAxes, fontsize=50, verticalalignment='top')

    # Fairness
    alpha_fair = calculate_diagonal_distance(fair_dp_infl, fair_dp_actual)
    colors_fair = np.array([[rgb[0], rgb[1], rgb[2], a] for a in alpha_fair])
    axes[1].scatter(fair_dp_infl, fair_dp_actual, c=colors_fair, s=50, edgecolors='none')
    axes[1].plot([min(fair_dp_infl), max(fair_dp_infl)], [min(fair_dp_infl), max(fair_dp_infl)], 
                 color='silver')
    axes[1].set_xlabel('Actual Fairness Change', fontsize=40, fontweight='bold')
    axes[1].set_ylabel('Approximate Fairness Change', fontsize=40, fontweight='bold')
    spearman_dp, _ = spearmanr(fair_dp_infl, fair_dp_actual)
    pearson_dp, _ = pearsonr(fair_dp_infl, fair_dp_actual)
    axes[1].text(0.05, 0.95, f'Spearman: {spearman_dp:.2f}\nPearson: {pearson_dp:.2f}', 
                 transform=axes[1].transAxes, fontsize=50, verticalalignment='top')

    # Robust
    alpha_robust = calculate_diagonal_distance(robust_infl, robust_actual)
    colors_robust = np.array([[rgb[0], rgb[1], rgb[2], a] for a in alpha_robust])
    axes[2].scatter(robust_infl, robust_actual, c=colors_robust, s=50, edgecolors='none')
    axes[2].plot([min(robust_infl), max(robust_infl)], [min(robust_infl), max(robust_infl)], 
                 color='silver', linestyle='--')
    axes[2].set_xlabel('Actual Robustness Change', fontsize=40, fontweight='bold')
    axes[2].set_ylabel('Approximate Robustness Change', fontsize=40, fontweight='bold')
    # Adjust y-axis label position
    axes[2].yaxis.set_label_coords(-0.08, 0.45)  # Move y-axis label down
    spearman_robust, _ = spearmanr(robust_infl, robust_actual)
    pearson_robust, _ = pearsonr(robust_infl, robust_actual)
    axes[2].text(0.05, 0.95, f'Spearman: {spearman_robust:.2f}\nPearson: {pearson_robust:.2f}', 
                 transform=axes[2].transAxes, fontsize=50, verticalalignment='top')

    # Set uniform format for all subplots
    for ax in axes:
        ax.tick_params(axis='both', labelsize=29.5)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.xaxis.get_offset_text().set_fontsize(29.5)
        ax.yaxis.get_offset_text().set_fontsize(29.5)
        # Adjust x-axis label position
        ax.xaxis.set_label_coords(0.5, -0.08)  # Move x-axis label up (from -0.15 to -0.1)
        # Adjust scientific notation position
        ax.xaxis.get_offset_text().set_y(-0.6)  # Move scientific notation up (from -0.15 to -0.1)

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Increase bottom space
    plt.savefig('results/SEC-5-1-1-1/correlation_plot.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to release memory