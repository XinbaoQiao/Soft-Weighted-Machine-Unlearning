# -*- coding: utf-8 -*-

import os
import time
import argparse
import numpy as np
from typing import Sequence
from dataset import fetch_data, DataTemplate
from dataset2 import fetch_data2, DataTemplate2
from eval import Evaluator
from model import LogisticRegression, NNLastLayerIF, MLPClassifier
from fair_fn import grad_ferm, grad_dp, loss_ferm, loss_dp
from utils import fix_seed, save2csv
import json
from robust_fn import grad_robust, calc_robust_acc
from robust_fn_nn import grad_robust_nn, calc_robust_acc_nn
import pickle
import random
import copy
import matplotlib
import matplotlib.colors as mcolors
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def parse_args():
    parser = argparse.ArgumentParser(description='Influence Fairness')
    parser.add_argument('--dataset', type=str, default="adult", help="name of the dataset")
    parser.add_argument('--num_data', type=int, default=200, help="number of data")
    parser.add_argument('--metric', type=str, default="dp", help="robust or dp/eop")
    parser.add_argument('--seed', type=float, default=42, help="random seed")
    parser.add_argument('--save_model', type=str, default="n", help="y/n")
    parser.add_argument('--type', type=str, default="util", help="util/fair/robust")
    parser.add_argument('--model_type', type=str, default="logreg", help="logreg/nn")
    parser.add_argument('--points_to_delete', type=int, default=0, help="points to delete")
    parser.add_argument('--weights_regularizer', type=float, default=0.1, help="weights regularizer")
    parser.add_argument('--weight_scheme', type=str, default="soft", help="soft/hard")

    args = parser.parse_args()
    return args

def normalize_weights(weights):
    abs_max = max(abs(np.max(weights)), abs(np.min(weights)))
    return weights / abs_max

def plot_weights_comparison():
    # Define colors
    soft_color = '#9F0000' 
    hard_color = '#003A75'  

    # Load weight data
    # Fairness weights
    fair_soft = np.load('results/SEC-5-1-2-1/Analytical_weights_2_fair_0.00175751.npy')
    fair_hard = np.load('results/SEC-5-1-2-1/hard_weights_fair.npy')
    # Robustness weights
    robust_soft = np.load('results/SEC-5-1-2-2/Analytical_weights_3_robust_1000.00000000.npy')
    robust_hard = np.load('results/SEC-5-1-2-2/hard_weights_robust.npy')

    # Load influence values
    fair_infl = pd.read_csv('results/SEC-5-1-2-1/fair/fair_infl.csv')['Influence Fair Difference'].values
    robust_infl = pd.read_csv('results/SEC-5-1-2-2/robust/robust_infl.csv')['Influence Robust Difference'].values

    # Print weight statistics for checking
    print("Fairness soft weights stats:", np.min(fair_soft), np.max(fair_soft), np.mean(fair_soft))
    print("Robustness soft weights stats:", np.min(robust_soft), np.max(robust_soft), np.mean(robust_soft))

    # Normalize soft weights (preserving positive/negative properties)
    fair_soft = normalize_weights(fair_soft)
    robust_soft = normalize_weights(robust_soft)

    print("After normalization:")
    print("Fairness soft weights stats:", np.min(fair_soft), np.max(fair_soft), np.mean(fair_soft))
    print("Robustness soft weights stats:", np.min(robust_soft), np.max(robust_soft), np.mean(robust_soft))

    # Fairness plot
    plt.figure(figsize=(22, 12))
    # Get sort indices based on influence values
    fair_sort_idx = np.argsort(fair_infl)
    # Sort weights and influence values based on influence values
    fair_hard_sorted = fair_hard[fair_sort_idx]
    fair_soft_sorted = fair_soft[fair_sort_idx]
    fair_infl_sorted = fair_infl[fair_sort_idx]
    
    # Create dual Y axes
    fig, ax1 = plt.subplots(figsize=(22, 12))
    ax2 = ax1.twinx()
    
    # Plot weights (left Y axis)
    ax1.plot(range(len(fair_soft)), fair_soft_sorted, color=soft_color, label='Soft Weights', linewidth=15, alpha=0.75)
    ax1.plot(range(len(fair_hard)), fair_hard_sorted, color=hard_color, label='Hard Weights', linewidth=15)
    
    # Scale influence values (divide by 1e-2)
    scaled_fair_infl = fair_infl_sorted / 1e-2
    
    # Plot influence values (right Y axis)
    ax2.plot(range(len(fair_infl)), scaled_fair_infl, color='#482957', label='Influence Value', linewidth=15, linestyle='--')
    
    ax1.set_xlabel('Sample Index', fontsize=60, fontweight='bold')
    ax1.set_ylabel('Weight Value', fontsize=60, fontweight='bold')
    ax2.set_ylabel('$\\mathcal{I}_{\\rm fair}$', fontsize=95, fontweight='bold', color='#482957', labelpad=-20)
    
    # Add scale notation in top right
    plt.text(0.98, 0.98, '$\\times 10^{-2}$', fontsize=48, color='#482957', 
             horizontalalignment='right', verticalalignment='top', transform=ax2.transAxes)
    
    # Merge legends of both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=56, loc='upper left')
    
    ax1.grid(True)
    ax1.tick_params(axis='both', which='major', labelsize=48)
    ax2.tick_params(axis='y', which='major', labelsize=42, colors='#482957')
    
    # Add vertical Fairness title
    plt.text(-0.31, 0.5, 'Fairness', fontsize=100, fontweight='bold', rotation=90, va='center', transform=ax1.transAxes)
    plt.tight_layout()
    plt.savefig('results/SEC-5-1-2-3/weights_comparison_fairness.pdf')
    plt.close()

    # Robustness plot
    plt.figure(figsize=(22, 12))
    # Get sort indices based on influence values
    robust_sort_idx = np.argsort(robust_infl)
    # Sort weights and influence values based on influence values
    robust_hard_sorted = robust_hard[robust_sort_idx]
    robust_soft_sorted = robust_soft[robust_sort_idx]
    robust_infl_sorted = robust_infl[robust_sort_idx]
    
    # Create dual Y axes
    fig, ax1 = plt.subplots(figsize=(22, 12))
    ax2 = ax1.twinx()
    
    # Plot weights (left Y axis)
    ax1.plot(range(len(robust_soft)), robust_soft_sorted, color=soft_color, label='Soft Weights', linewidth=15, alpha=0.75)
    ax1.plot(range(len(robust_hard)), robust_hard_sorted, color=hard_color, label='Hard Weights', linewidth=15)
    
    # Plot influence values (right Y axis)
    ax2.plot(range(len(robust_infl)), robust_infl_sorted, color='#482957', label='Influence Value', linewidth=15, linestyle='--')
    
    ax1.set_xlabel('Sample Index', fontsize=60, fontweight='bold')
    ax1.set_ylabel('Weight Value', fontsize=60, fontweight='bold')
    ax2.set_ylabel('$\\mathcal{I}_{\\rm robust}$', fontsize=95, fontweight='bold', color='#482957', labelpad=-20)
    
    # Merge legends of both axes and place in bottom right
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=56, loc='lower right')
    
    ax1.grid(True)
    ax1.tick_params(axis='both', which='major', labelsize=48)
    ax2.tick_params(axis='y', which='major', labelsize=42, colors='#482957')
    
    # Add vertical Robustness title
    plt.text(-0.3, 0.5, 'Robustness', fontsize=100, fontweight='bold', rotation=90, va='center', transform=ax1.transAxes)
    plt.tight_layout()
    plt.savefig('results/SEC-5-1-2-3/weights_comparison_robustness.pdf')
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    
    # Create save directory
    os.makedirs('results/SEC-5-1-2-3', exist_ok=True)
    
    # Draw weight comparison plots
    plot_weights_comparison()
