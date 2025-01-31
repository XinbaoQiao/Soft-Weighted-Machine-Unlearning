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
    parser.add_argument('--metric', type=str, default="eop", help="eop or dp")
    parser.add_argument('--seed', type=float, default=42, help="random seed")
    parser.add_argument('--save_model', type=str, default="n", help="y/n")
    parser.add_argument('--type', type=str, default="util", help="util/fair/")
    parser.add_argument('--strategy', type=str, default="dec", help="inc/dec/random")
    parser.add_argument('--points_to_delete', type=int, default=200, help="points to delete (num or index)")
    parser.add_argument('--random_seed', type=int, default=42, help="seed for random strategy")
    parser.add_argument('--only_pre', type=str, default="n", help="y/n")
    parser.add_argument('--model_type', type=str, default="logreg", help="logreg/nn")

    args = parser.parse_args()

    return args


# Remove outliers (using 3 standard deviations as threshold)
def Speedupdrawing(x, y):
    x_mean, x_std = np.mean(x), np.std(x)
    y_mean, y_std = np.mean(y), np.std(y)
    mask = (abs(x - x_mean) < 3 * x_std) & (abs(y - y_mean) < 3 * y_std)
    return x[mask], y[mask]


if __name__ == "__main__":
    args = parse_args()

    args.save_model = 'y'
    args.metric = 'dp'  # can be 'dp/EOP' or 'robust'    
    args.seed = 1
    args.num_data = 30162
    
    os.makedirs('results/SEC-1-1', exist_ok=True)

    # Load data
    util_actual = np.load('results/SEC-5-1-0-1/util_actual.npy')
    util_infl = np.load('results/SEC-5-1-0-1/util_infl.npy')
    
    # Load fairness data
    fair_actual = np.load('results/SEC-5-1-0-1/fair_dp_actual.npy')
    # Load robustness data
    robust_actual = np.load('results/SEC-5-1-0-1/robust_actual.npy')

    # Remove outliers and ensure consistent data length (fairness)
    fair_util_actual = util_actual.copy()  # Create separate utility copy for fairness
    fair_actual, fair_util_actual = Speedupdrawing(fair_actual, fair_util_actual)

    # Remove outliers and ensure consistent data length (robustness)
    robust_util_actual = util_actual.copy()  # Create separate utility copy for robustness
    robust_actual, robust_util_actual = Speedupdrawing(robust_actual, robust_util_actual)

    # Sort fairness data
    fair_sort_idx = np.argsort(fair_actual)
    fair_actual = fair_actual[fair_sort_idx]
    fair_util_actual = fair_util_actual[fair_sort_idx]

    # Sort robustness data
    robust_sort_idx = np.argsort(robust_actual)
    robust_actual = robust_actual[robust_sort_idx]
    robust_util_actual = robust_util_actual[robust_sort_idx]

    x_indices_fair = np.arange(len(fair_actual))    
    x_indices_robust = np.arange(len(robust_actual))

    # Create 2x2 plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(25, 16))
    plt.style.use('fast')
    
    # Set color and point parameters
    default_color = '#704DA8'  # Default purple
    highlight_color = '#E8655A'  # Highlight red
    
    # Set scatter plot parameters
    actual_params = {
        'alpha': 0.6,     
        's': 120,         
        'rasterized': True,
        'marker': 'o'     
    }

    # Create color mask (fairness)
    fair_colors = np.where((fair_util_actual > 0) & (fair_actual < 0), 
                          highlight_color, default_color)

    # Create color mask (robustness)
    robust_colors = np.where((robust_util_actual > 0) & (robust_actual < 0), 
                            highlight_color, default_color)

    # Fairness (top left)
    ax1.scatter(x_indices_fair, fair_actual, c=fair_colors, **actual_params)
    ax1.set_xlabel('')
    ax1.tick_params(axis='x', labelbottom=False)
    ax1.set_ylabel('Fairness', fontsize=60, fontweight='bold')
    ax1.yaxis.set_label_coords(-0.1, 0.5)

    # Robustness (top right)
    ax2.scatter(x_indices_robust, robust_actual, c=robust_colors, **actual_params)
    ax2.set_xlabel('')
    ax2.tick_params(axis='x', labelbottom=False)
    ax2.set_ylabel('Robustness', fontsize=60, fontweight='bold')
    ax2.yaxis.set_label_coords(-0.1, 0.5)

    # Utility for Fairness (bottom left)
    ax3.scatter(x_indices_fair, fair_util_actual, c=fair_colors, **actual_params)
    ax3.set_xlabel('Sample Index', fontsize=45, fontweight='bold')
    ax3.set_ylabel('Utility', fontsize=60, fontweight='bold')
    ax3.yaxis.set_label_coords(-0.1, 0.5)

    # Utility for Robustness (bottom right)
    ax4.scatter(x_indices_robust, robust_util_actual, c=robust_colors, **actual_params)
    ax4.set_xlabel('Sample Index', fontsize=45, fontweight='bold')
    ax4.set_ylabel('Utility', fontsize=60, fontweight='bold')
    ax4.yaxis.set_label_coords(-0.1, 0.5)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(axis='both', labelsize=35)
        ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        ax.xaxis.get_offset_text().set_fontsize(35)
        ax.yaxis.get_offset_text().set_fontsize(35)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
        
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.1)  
    plt.savefig('results/SEC-1-1/actual_series_all.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.close()