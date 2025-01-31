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
    parser.add_argument('--model_type', type=str, default="nn", help="logreg/nn/resnet")

    args = parser.parse_args()

    return args


# 移除异常点（使用3个标准差作为阈值）
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


if __name__ == "__main__":
    args = parse_args()

    args.save_model = 'y'
    args.metric = 'eop'  # Can be 'dp/EOP' or 'robust'    
    args.seed = 1
    args.num_data = 30162
    
    print("Patience Is All You Need...")
    os.makedirs('results/SEC-5-1-1-3', exist_ok=True)

    # Load data
    util_actual = np.load('results/SEC-5-1-0-3/util_actual.npy')
    util_infl = np.load('results/SEC-5-1-0-3/util_infl.npy')
    
    # Load fairness data based on metric type
    if args.metric == 'eop':
        metric_actual = np.load('results/SEC-5-1-0-3/fair_eop_actual.npy')
        metric_infl = np.load('results/SEC-5-1-0-3/fair_eop_infl.npy')
        metric_name = 'Fairness(EOP)'
    else:
        metric_actual = np.load('results/SEC-5-1-0-3/fair_dp_actual.npy')
        metric_infl = np.load('results/SEC-5-1-0-3/fair_dp_infl.npy')
        metric_name = 'Fairness(DP)'

    # Remove outliers
    metric_infl, metric_actual = Speedupdrawing(metric_infl, metric_actual)
    util_infl, util_actual = Speedupdrawing(util_infl, util_actual)

    # Ensure consistent data lengths
    min_len = min(len(metric_actual), len(util_actual))
    metric_actual = metric_actual[:min_len]
    metric_infl = metric_infl[:min_len]
    util_actual = util_actual[:min_len]
    util_infl = util_infl[:min_len]

    # Sort metric data by actual values and reorder utility data accordingly
    metric_sort_idx = np.argsort(metric_actual)
    metric_actual = metric_actual[metric_sort_idx]
    metric_infl = metric_infl[metric_sort_idx]
    util_actual = util_actual[metric_sort_idx]
    util_infl = util_infl[metric_sort_idx]

    # Downsample, take one point every k points
    k = 10  # Downsample rate
    x_indices = np.arange(len(metric_actual))[::k]
    metric_actual = metric_actual[::k]
    metric_infl = metric_infl[::k]
    util_actual = util_actual[::k]
    util_infl = util_infl[::k]

    # Create vertically split figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11))
    plt.style.use('fast')
    plt.subplots_adjust(hspace=0.1)  # Reduce subplot spacing

    # Set colors and point parameters
    actual_color = '#704DA8'  
    infl_color = '#7DC99F'    
    
    # Set scatter plot parameters
    actual_params = {
        'alpha': 0.6,     # Slightly transparent
        's': 120,         # Increase point size
        'rasterized': True,
        'marker': 'o'     # Circle marker
    }
    
    infl_params = {
        'alpha': 0.85,     # Transparency
        's': 300,         # Increase x size
        'rasterized': True,
        'marker': 'x',    # Use x marker
        'linewidth': 4    # Increase line width
    }

    # Metric 
    ax1.scatter(x_indices, metric_infl, c=infl_color, label='Approximate Change', **infl_params)
    ax1.scatter(x_indices, metric_actual, c=actual_color, label='Actual Change', zorder=2, **actual_params)
    
    ax1.set_xlabel('')  # Remove upper plot x-axis label
    ax1.tick_params(axis='x', labelbottom=False)  # Hide x-axis labels this way
    ax1.set_ylabel(f'{metric_name}', fontsize=45, fontweight='bold')
    # Adjust y-axis label position based on metric type
    if args.metric == 'robust':
        ax1.yaxis.set_label_coords(-0.08, 0.4)  # Lower and more left position for robustness
    else:
        ax1.yaxis.set_label_coords(-0.06, 0.5)  # Position for fairness
    legend = ax1.legend(fontsize=45, loc='upper left', handletextpad=0.1, borderpad=0.2, labelspacing=0.2)
    legend.get_frame().set_linewidth(0.0)  # Remove legend frame

    # Utility (lower plot)
    ax2.scatter(x_indices, util_infl, c=infl_color, **infl_params)
    ax2.scatter(x_indices, util_actual, c=actual_color, zorder=2, **actual_params)
    
    ax2.set_xlabel('Sample Index', fontsize=45, fontweight='bold')
    ax2.set_ylabel('Utility', fontsize=45, fontweight='bold')
    # Adjust y-axis label position based on metric type, keep consistent with upper plot
    if args.metric == 'robust':
        ax2.yaxis.set_label_coords(-0.08, 0.4)  # Position for robustness
    else:
        ax2.yaxis.set_label_coords(-0.06, 0.5)  # Position for fairness

    # Set uniform format for all subplots
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', labelsize=35)  # Reduce tick label size
        ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        ax.xaxis.get_offset_text().set_fontsize(35)  # Reduce scientific notation font size
        ax.yaxis.get_offset_text().set_fontsize(35)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
        
        # Reduce number of ticks
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))  # Maximum 5 ticks on x-axis
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))  # Maximum 5 ticks on y-axis

    plt.tight_layout()
    plt.subplots_adjust(left=0.12)  # Reduce left margin
    plt.savefig(f'results/SEC-5-1-1-3/index_series_{args.metric}.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

    # Calculate correlation coefficients
    pearson_metric, _ = pearsonr(metric_infl, metric_actual)
    spearman_metric, _ = spearmanr(metric_infl, metric_actual)

    # Create correlation plot
    plt.figure(figsize=(11, 11))
    plt.style.use('fast')

    # Set colors and point parameters
    scatter_color = '#836796'
    scatter_params = {
        'alpha': 0.6,
        's': 120,
        'rasterized': True
    }

    # Metric correlation plot
    plt.scatter(metric_infl, metric_actual, c=scatter_color, **scatter_params)
    plt.plot([min(metric_infl), max(metric_infl)], [min(metric_infl), max(metric_infl)], 
             color='silver', linestyle='--')
    plt.xlabel('Approximate Change', fontsize=40, fontweight='bold')
    plt.ylabel('Actual Change', fontsize=40, fontweight='bold')
    plt.text(0.05, 0.95, f'Spearman: {spearman_metric:.2f}\nPearson: {pearson_metric:.2f}', 
             transform=plt.gca().transAxes, fontsize=50, verticalalignment='top')
    plt.title(f'{metric_name}', fontsize=45, fontweight='bold', pad=20)

    # Set format
    plt.tick_params(axis='both', labelsize=29.5)
    plt.gca().ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.gca().xaxis.get_offset_text().set_fontsize(29.5)
    plt.gca().yaxis.get_offset_text().set_fontsize(29.5)
    plt.gca().xaxis.set_label_coords(0.5, -0.08)
    plt.gca().xaxis.get_offset_text().set_y(-0.6)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f'results/SEC-5-1-1-3/correlation_plot_{args.metric}.pdf', dpi=300, bbox_inches='tight')
    plt.close()