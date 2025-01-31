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
    parser.add_argument('--num_data', type=int, default=600, help="number of data")
    parser.add_argument('--metric', type=str, default='dp', help="robust or dp/eop")
    parser.add_argument('--seed', type=float, default=0, help="random seed")
    parser.add_argument('--save_model', type=str, default='y', help="y/n")
    parser.add_argument('--type', type=str, default="robust", help="util/fair/robust")
    parser.add_argument('--model_type', type=str, default="logreg", help="logreg/nn")
    parser.add_argument('--points_to_delete', type=int, default=0, help="points to delete")
    parser.add_argument('--weights_regularizer', type=float, default=0.1, help="weights regularizer")
    parser.add_argument('--weight_scheme', type=str, default='soft-2.2', help="'hard', 'soft-2.3','soft-2.2','soft-2.1'")
    args = parser.parse_args()
    return args

def plot_decision_boundary(model, X, X_original, ax, linestyle='-', alpha=1.0, color='#496C88', pca_model=None):
    """
    Plot decision boundary for high dimensional data using PCA
    """
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    if pca_model is not None:
        mesh_points_original = pca_model.inverse_transform(mesh_points)
        Z, _ = model.pred(mesh_points_original)
    else:
        Z, _ = model.pred(mesh_points)
    
    Z = Z.reshape(xx.shape)
    
    ax.contour(xx, yy, Z, levels=[0.5], 
               colors=color, alpha=alpha, linestyles=linestyle, linewidths=2)



def pre_main(args):
    tik = time.time()

    if args.seed is not None:
        fix_seed(args.seed)
    
    with open('data/' + args.dataset  + '/meta.json', 'r+') as f:
        json_data = json.load(f)
        json_data['train_path'] = './data/' + args.dataset + '/train.csv'
        f.seek(0)        
        json.dump(json_data, f, indent=4)
        f.truncate()


    """ initialization"""
    data: DataTemplate = fetch_data(args.dataset)
    model = LogisticRegression(l2_reg=data.l2_reg)
    if args.dataset != "adult" or args.num_data != 600:
        args.num_data = 1000 if args.dataset == "bank" or args.dataset == "adult" else -1

    if args.model_type == 'nn':
        model = NNLastLayerIF(input_dim=data.dim, base_model_cls=MLPClassifier, l2_reg=1e-4)

    if args.num_data == -1:
        n_samples = len(data.x_train)
        n_val = len(data.x_val)
        n_test = len(data.x_test)
    else:
        n_samples = args.num_data
        n_val = int(n_samples * 0.2)
        n_test = int(n_samples * 0.2)


    train_idx = np.random.choice(len(data.x_train), n_samples, replace=False)
    data.x_train = data.x_train[train_idx]
    data.y_train = data.y_train[train_idx]

    train_data_df = pd.DataFrame(data.x_train)
    train_data_df['label'] = data.y_train
    # train_data_df.to_csv('results/SEC-1/train_data_pre.csv', index=True)

    val_idx = np.random.choice(len(data.x_val), n_val, replace=False)
    data.x_val = data.x_val[val_idx]
    data.y_val = data.y_val[val_idx]
    data.s_val = data.s_val[val_idx]
    
    test_idx = np.random.choice(len(data.x_test), n_test, replace=False)
    data.x_test = data.x_test[test_idx]
    data.y_test = data.y_test[test_idx]
    data.s_test = data.s_test[test_idx]


    # Add data standardization
    global scaler_model
    scaler_model = StandardScaler()
    x_train_scaled = scaler_model.fit_transform(data.x_train)
    
    # Modify PCA part to use standardized data
    global pca
    pca = PCA(n_components=2)
    x_train_pca = pca.fit_transform(x_train_scaled)
    
    # Ensure other data is also standardized
    x_val_scaled = scaler_model.transform(data.x_val)
    x_val_pca = pca.transform(x_val_scaled)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    val_evaluator, test_evaluator = Evaluator(data.s_val, "val"), Evaluator(data.s_test, "test")

    """ vanilla training """

    model.fit(data.x_train, data.y_train)
    #if args.dataset == "toy" and args.save_model == "y":
    #    pickle.dump(model.model, open("toy/model_pre.pkl", "wb"))

    if args.metric == "eop":
        ori_fair_loss_val = loss_ferm(model.log_loss, data.x_val, data.y_val, data.s_val)
    elif args.metric == "dp":
        pred_val, _ = model.pred(data.x_val)
        ori_fair_loss_val = loss_dp(data.x_val, data.s_val, pred_val)
    else:
        raise ValueError
    ori_util_loss_val = model.log_loss(data.x_val, data.y_val)

    """ compute the influence and save data """
    pred_train, _ = model.pred(data.x_train)

    train_total_grad, train_indiv_grad = model.grad(data.x_train, data.y_train)
    util_loss_total_grad, acc_loss_indiv_grad = model.grad(data.x_val, data.y_val)
    if args.metric == "eop":
        fair_loss_total_grad = grad_ferm(model.grad, data.x_val, data.y_val, data.s_val)
    elif args.metric == "dp":
        fair_loss_total_grad = grad_dp(model.grad_pred, data.x_val, data.s_val)
    else:
        raise ValueError

    hess = model.hess(data.x_train)
    util_grad_hvp = model.get_inv_hvp(hess, util_loss_total_grad)
    fair_grad_hvp = model.get_inv_hvp(hess, fair_loss_total_grad)

    # Calculate influence values
    util_pred_infl = train_indiv_grad.dot(util_grad_hvp)
    fair_pred_infl = train_indiv_grad.dot(fair_grad_hvp)

    # Save influence values as .npy files
    np.save('results/SEC-5-1-2-1/fair/util_infl.npy', util_pred_infl)
    np.save('results/SEC-5-1-2-1/fair/fair_infl.npy', fair_pred_infl)

    # Save as CSV files for better readability
    util_infl_df = pd.DataFrame(util_pred_infl, columns=['Influence Util Difference'])
    fair_infl_df = pd.DataFrame(fair_pred_infl, columns=['Influence Fair Difference'])
    
    util_infl_df.to_csv('results/SEC-5-1-2-1/fair/util_infl.csv', index=False)
    fair_infl_df.to_csv('results/SEC-5-1-2-1/fair/fair_infl.csv', index=False)

    # Save combined influence values
    combined_df = pd.DataFrame({
        'util_influence': util_pred_infl,
        'fair_influence': fair_pred_infl
    })
    combined_df.to_csv('results/SEC-5-1-2-1/fair/influence_values.csv', index=False)

    _, pred_label_val = model.pred(data.x_val)
    _, pred_label_test = model.pred(data.x_test)


    np.save('traindata.npy', np.append(data.x_train, data.y_train.reshape((-1,1)), 1))
    return data, x_train_pca, model, ori_util_loss_val, ori_fair_loss_val





def post_main(args, model, data, weights):
    """ initialization"""
    model = LogisticRegression(l2_reg=data.l2_reg)

    if args.model_type == 'nn':
        model = NNLastLayerIF(input_dim=data.dim, base_model_cls=MLPClassifier, l2_reg=1e-4)

    val_evaluator, test_evaluator = Evaluator(data.s_val, "val"), Evaluator(data.s_test, "test")

    """ vanilla training with weights """
    model.fit(data.x_train, data.y_train, sample_weight=None)

    train_total_grad, train_indiv_grad = model.grad(data.x_train, data.y_train, sample_weight=None)
    hess = model.hess(data.x_train, sample_weight=None)
    
    # Calculate weighted gradient
    weighted_grad = np.sum(train_indiv_grad * weights.reshape(-1, 1), axis=0)
    # Calculate hvp
    delta = model.get_inv_hvp(hess, weighted_grad)  # Update weights directly
    
    # Create new model and update parameters
    post_model = copy.deepcopy(model)
    post_model.weight = model.weight - delta  # Update weights directly
    if hasattr(model, 'bias'):  # If model has bias term
        post_model.bias = model.bias
    
    # Update sklearn model parameters
    post_model.model.coef_ = post_model.weight.reshape(1, -1)

    if args.metric == "eop":
        new_fair_loss_val = loss_ferm(post_model.log_loss, data.x_val, data.y_val, data.s_val)
    elif args.metric == "dp":
        pred_val, _ = post_model.pred(data.x_val)
        new_fair_loss_val = loss_dp(data.x_val, data.s_val, pred_val)
    else:
        raise ValueError
    new_util_loss_val = post_model.log_loss(data.x_val, data.y_val)
    # print("New Utility Loss Value:", new_util_loss_val)  # Print the new utility loss value

    return data, post_model, new_util_loss_val, new_fair_loss_val

def plot_results(data, x_train_pca, pre_model, post_model, x_train_post_pca, weights, args):
    # Define color constants
    BLUE_COLOR = '#5572cc'
    RED_COLOR = '#e0531e'
    
    if args.weight_scheme == "hard":
        # For hard scheme, create only main plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        scatter = ax.scatter(x_train_pca[:, 0], x_train_pca[:, 1],
                           c=[BLUE_COLOR if y == 0 else RED_COLOR for y in data.y_train],
                           alpha=0.85, s=200)
        deleted_mask = weights == -1
        for i in range(len(x_train_pca)):
            if deleted_mask[i]:
                ax.scatter(x_train_pca[i, 0], x_train_pca[i, 1],
                          c=[BLUE_COLOR if data.y_train[i] == 0 else RED_COLOR],
                          marker='x', s=400, linewidths=3, alpha=0.8)
    else:
        # For soft scheme, create main plot and colorbar subplots
        fig, (ax, ax_colorbar1, ax_colorbar2) = plt.subplots(1, 3, figsize=(12, 8), 
                                             gridspec_kw={'width_ratios': [6, 0.2, 0.2]})
        
        weights_norm = (weights - weights.min()) / (weights.max() - weights.min())
        alphas = np.where(weights >= 0,
                         0.5 + 0.5 * weights_norm,
                         0.01 + 0.495 * weights_norm)
        for i in range(len(x_train_pca)):
            ax.scatter(x_train_pca[i, 0], x_train_pca[i, 1],
                      c=[BLUE_COLOR if data.y_train[i] == 0 else RED_COLOR],
                      alpha=alphas[i], s=200)
        # Blue gradient
        colors_blue = [(1, 1, 1, 1), tuple([int(BLUE_COLOR[1:][i:i+2], 16)/255 for i in (0, 2, 4)] + [1])]
        cmap_blue = mcolors.LinearSegmentedColormap.from_list('custom_blue', colors_blue)
        norm_blue = mcolors.Normalize(vmin=0, vmax=1)
        cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=norm_blue, cmap=cmap_blue), 
                        cax=ax_colorbar1,
                        ticks=[])
        cb1.ax.set_ylabel('')
        
        # Red gradient
        colors_red = [(1, 1, 1, 1), tuple([int(RED_COLOR[1:][i:i+2], 16)/255 for i in (0, 2, 4)] + [1])]
        cmap_red = mcolors.LinearSegmentedColormap.from_list('custom_red', colors_red)
        norm_red = mcolors.Normalize(vmin=0, vmax=1)
        cb2 = plt.colorbar(plt.cm.ScalarMappable(norm=norm_red, cmap=cmap_red), 
                        cax=ax_colorbar2,
                        ticks=[])
        cb2.ax.set_ylabel('')
        
        # Add annotation between the two colorbars
        midpoint = (ax_colorbar1.get_position().x1 + ax_colorbar2.get_position().x0) / 2
        top_pos = ax_colorbar2.get_position().y1 + 0.105
        bottom_pos = ax_colorbar2.get_position().y0 - 0.007
        
        fig.text(midpoint , top_pos, 'High Weight', 
                fontsize=29, fontweight='bold', va='bottom')
        fig.text(midpoint , bottom_pos, 'Low Weight', 
                fontsize=29, fontweight='bold', va='top')
    
    # Calculate normal range (for setting display range)
    q1_x = np.percentile(x_train_pca[:, 0], 5)
    q3_x = np.percentile(x_train_pca[:, 0], 95)
    q1_y = np.percentile(x_train_pca[:, 1], 5)
    q3_y = np.percentile(x_train_pca[:, 1], 95)
    
    # Set display range
    margin = 0.1
    x_range = q3_x - q1_x
    y_range = q3_y - q1_y
    ax.set_xlim(q1_x - margin * x_range, q3_x + margin * x_range)
    ax.set_ylim(q1_y - margin * y_range, q3_y + margin * y_range)

    # Plot decision boundaries
    plot_decision_boundary(pre_model, x_train_pca, data.x_train, ax, 
                         linestyle='--', alpha=1.0, color='#181849', pca_model=pca)
    plot_decision_boundary(post_model, x_train_pca, data.x_train, ax, 
                         linestyle='-', alpha=1.0, color='#8E0F31', pca_model=pca)

    # Display accuracy and fairness comparison before and after
    metrics_text = (f'Pre-Unlearning:\n'
                   f'  Test Accuracy: {pre_accuracy*100:.2f}%\n'
                   f'  Fairness Loss: {pre_fair_loss:.4f}\n'
                   f'Post-Unlearning:\n'
                   f'  Test Accuracy: {post_accuracy*100:.2f}%\n'
                   f'  Fairness Loss: {(post_fair_loss):.4f}')
    
    ax.text(0.02, 0.98, metrics_text,
            transform=ax.transAxes,
            fontsize=29,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set labels and ticks
    ax.set_xlabel('Feature One', fontsize=32, fontweight='bold')
    if args.weight_scheme == "hard":
        ax.set_ylabel('Feature Two', fontsize=32, fontweight='bold')
    ax.tick_params(axis='both', labelsize=27)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='#024163', linestyle='--', linewidth=2,
                  label='Pre-Decision Boundary'),
        plt.Line2D([0], [0], color='#8E0F31', linestyle='-', linewidth=2,
                  label='Post-Decision Boundary')
    ]
    
    ax.legend(handles=legend_elements, fontsize=27.5, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('results/SEC-5-1-2-1/fair/decision_boundaries_{}_{}.pdf'.format(args.weight_scheme,args.metric), 
                format='pdf', bbox_inches='tight', 
                pad_inches=0.3)  # Increase margin
    plt.close()

if __name__ == "__main__":
    args = parse_args()

    fix_seed(args.seed)
    os.makedirs('results/SEC-5-1-2-1/fair', exist_ok=True)
    
    # Run pre-training
    data, x_train_pca, pre_model, ori_util_loss_val, ori_fair_loss_val = pre_main(args)

    # Load influence values
    fair_infl = np.load('results/SEC-5-1-2-1/fair/fair_infl.npy')
    util_infl = np.load('results/SEC-5-1-2-1/fair/util_infl.npy')
    weights = np.zeros(len(fair_infl))
    
    k = 0.1 if args.dataset == "adult" and args.num_data == 600 else 0.2
    num_points_to_delete = int(k * len(fair_infl))
    points_to_delete = np.argsort(fair_infl)[:num_points_to_delete]
    weights[points_to_delete] = -1
    data_post, post_model, hard_post_util_loss_val, hard_post_fair_loss_val = post_main(args, pre_model, data, weights)
   
    if args.weight_scheme == "hard":
        # Create weight array, initialize to 0
        data_post, post_model, post_util_loss_val, post_fair_loss_val = post_main(args, pre_model, data, weights)
        
        # Save hard weights
        np.save(f'results/SEC-5-1-2-1/hard_weights_fair.npy', weights)
        weights_df = pd.DataFrame(weights)
        weights_df.to_csv(f'results/SEC-5-1-2-1/hard_weights_fair.csv', index=False, header=False)

    elif args.weight_scheme == "soft-2.0":
        """ Find best lambda for Soft weights2.0"""
        # Generate a sequence of lambda values from small to large, with a base of 10
        iter_num = 50
        lambda_candidates = np.logspace(-3, 1, num=iter_num)  
        best_lambda = None
        best_fair_loss = float('inf')
        best_util_loss = ori_util_loss_val

        best_weights = None
        best_results = None
        
        print("\nSearching for best lambda for Scheme 2.0...")
        print(f"Testing {len(lambda_candidates)} lambda values from {lambda_candidates[0]:.8f} to {lambda_candidates[-1]:.8f}")
        
        for lambda_reg in lambda_candidates:
            weights = (fair_infl ) / (2 * lambda_reg)
            # print('Weighted Fairness',-np.dot(weights.T, fair_infl))
            # print('Weighted Utility',-np.dot(weights.T, util_infl))
            # Run post_main to get results
            data_post, post_model, post_util_loss_val, post_fair_loss_val = post_main(args, pre_model, data, weights)
            # print(f"\nλ = {lambda_reg:.8f}:")
            # print(f"Utility Loss: {post_util_loss_val:.8f}")
            # print(f"Original Utility Loss: {ori_util_loss_val:.8f}")  # Print the original utility loss
            # print(f"Fairness Loss: {post_fair_loss_val:.8f}")
            # print(f"Best Fairness Loss: {best_fair_loss:.8f}")  # Print the best fairness loss

            # if post_util_loss_val <= best_util_loss:  # Compare fairness loss
            if abs(post_fair_loss_val) <= abs(best_fair_loss):  # Compare fairness loss
                    best_fair_loss = post_fair_loss_val
                    best_fair_loss = post_fair_loss_val
                    best_lambda = lambda_reg
                    best_weights = weights
                    best_results = (data_post, post_model, post_util_loss_val, post_fair_loss_val)
            
        print(f"\nBest lambda for Scheme 2.2: {best_lambda:.8f}")
        print(f"Best Utility Loss: {best_results[2]:.8f}")
        print(f"Best Fairness Loss: {best_results[3]:.8f}")

        # Use best results
        args.weights_regularizer = best_lambda
        weights = best_weights
        data_post, post_model, post_util_loss_val, post_fair_loss_val = best_results
        np.save(f'results/SEC-5-1-2-1/Analytical_weights_2_fair_{best_lambda:.8f}.npy', weights)
        weights_df = pd.DataFrame(weights)
        weights_df.to_csv(f'results/SEC-5-1-2-1/Analytical_weights_2_fair_{best_lambda:.8f}.csv', index=False, header=False)

    elif args.weight_scheme == "soft-2.1":
        """ Soft weights2.1 with fixed lambda """
        lambda_reg = 0.1
        beta2 = 1- 2* lambda_reg * abs(ori_fair_loss_val) / (np.linalg.norm(fair_infl)**2)
        weights = (fair_infl * abs(ori_fair_loss_val) ) /  (np.linalg.norm(fair_infl)**2)
        print('Weighted Fairness',-np.dot(weights.T, fair_infl))
        print('Weighted Utility',-np.dot(weights.T, util_infl))            
        
        # Run post_main to get results
        data_post, post_model, post_util_loss_val, post_fair_loss_val = post_main(args, pre_model, data, weights)
        # print(f"\nλ = {lambda_reg:.8f}:")
        # print(f"Utility Loss: {post_util_loss_val:.8f}")
        # print(f"Original Utility Loss: {ori_util_loss_val:.8f}")
        # print(f"Fairness Loss: {post_fair_loss_val:.8f}")

        # Save results
        np.save(f'results/SEC-5-1-2-1/Analytical_weights_1_fair_{lambda_reg:.8f}.npy', weights)
        weights_df = pd.DataFrame(weights)
        weights_df.to_csv(f'results/SEC-5-1-2-1/Analytical_weights_1_fair_{lambda_reg:.8f}.csv', index=False, header=False)

    elif args.weight_scheme == "soft-2.2":
        """ Find best lambda for Soft weights2.2"""
        # Generate a sequence of lambda values from small to large, with a base of 10
        iter_num = 50
        lambda_candidates = np.logspace(-3, 1, num=iter_num)  
        best_lambda = None
        best_fair_loss = float('inf')
        best_util_loss = ori_util_loss_val

        best_weights = None
        best_results = None
        
        print("\nSearching for best lambda for Scheme 2.2...")
        print(f"Testing {len(lambda_candidates)} lambda values from {lambda_candidates[0]:.8f} to {lambda_candidates[-1]:.8f}")
        
        for lambda_reg in lambda_candidates:
            beta1 = - np.dot(fair_infl.T, util_infl) / (np.linalg.norm(util_infl)**2)
            if beta1 < 0:
                print(f"Skipping λ = {lambda_reg:.8f} because beta2 = {beta1:.8f} < 0")
                continue
            weights = (fair_infl + beta1 * util_infl) / (2 * lambda_reg)
            # print('Weighted Fairness',-np.dot(weights.T, fair_infl))
            # print('Weighted Utility',-np.dot(weights.T, util_infl))
            # Run post_main to get results
            data_post, post_model, post_util_loss_val, post_fair_loss_val = post_main(args, pre_model, data, weights)
            # print(f"\nλ = {lambda_reg:.8f}:")
            # print(f"Utility Loss: {post_util_loss_val:.8f}")
            # print(f"Original Utility Loss: {ori_util_loss_val:.8f}")  # Print the original utility loss
            # print(f"Fairness Loss: {post_fair_loss_val:.8f}")
            # print(f"Best Fairness Loss: {best_fair_loss:.8f}")  # Print the best fairness loss

            # if post_util_loss_val <= best_util_loss:  # Compare fairness loss
            if abs(post_fair_loss_val) <= abs(best_fair_loss):  # Compare fairness loss
                    best_fair_loss = post_fair_loss_val
                    best_fair_loss = post_fair_loss_val
                    best_lambda = lambda_reg
                    best_weights = weights
                    best_results = (data_post, post_model, post_util_loss_val, post_fair_loss_val)
            
        print(f"\nBest lambda for Scheme 2.2: {best_lambda:.8f}")
        print(f"Best Utility Loss: {best_results[2]:.8f}")
        print(f"Best Fairness Loss: {best_results[3]:.8f}")

        # Use best results
        args.weights_regularizer = best_lambda
        weights = best_weights
        data_post, post_model, post_util_loss_val, post_fair_loss_val = best_results
        np.save(f'results/SEC-5-1-2-1/Analytical_weights_2_fair_{best_lambda:.8f}.npy', weights)
        weights_df = pd.DataFrame(weights)
        weights_df.to_csv(f'results/SEC-5-1-2-1/Analytical_weights_2_fair_{best_lambda:.8f}.csv', index=False, header=False)

    elif args.weight_scheme == "soft-2.3":
        """ Soft weights2.3 with fixed lambda """
        lambda_reg = 0.1
        
        # Calculate denominator
        denominator = (np.linalg.norm(fair_infl)**2 * np.linalg.norm(util_infl)**2 - 
                      (np.dot(fair_infl, util_infl))**2) 
        # Calculate beta1
        beta1 = -(2 * lambda_reg * abs(ori_fair_loss_val) * np.linalg.norm(util_infl)**2 * np.dot(fair_infl.T, util_infl)) / (denominator * np.linalg.norm(util_infl)**2)
        # Calculate beta2
        beta2 = 1- (2 * lambda_reg * abs(ori_fair_loss_val) * np.linalg.norm(util_infl)**2) / denominator
        # Calculate weights epsilon
        weights = ((1 - beta2) * fair_infl + beta1 * util_infl) / (2 * lambda_reg)
        print('Weighted Fairness',-np.dot(weights.T, fair_infl))
        print('Weighted Utility',-np.dot(weights.T, util_infl))

        # Run post_main to get results
        weights = weights / {'adult': 1, 'celeba': 1, 'bank': 1}.get(args.dataset, 1.8)
        data_post, post_model, post_util_loss_val, post_fair_loss_val = post_main(args, pre_model, data, weights)
        print(f"\nλ = {lambda_reg:.8f}:")
        print(f"Utility Loss: {post_util_loss_val:.8f}")
        print(f"Original Utility Loss: {ori_util_loss_val:.8f}")
        print(f"Fairness Loss: {post_fair_loss_val:.8f}")

        # Save results
        np.save(f'results/SEC-5-1-2-1/Analytical_weights_3_fair_{lambda_reg:.8f}.npy', weights)
        weights_df = pd.DataFrame(weights)
        weights_df.to_csv(f'results/SEC-5-1-2-1/Analytical_weights_3_fair_{lambda_reg:.8f}.csv', index=False, header=False)


    
    print("\nWeight Statistics:")
    print(f"Mean weight: {np.mean(weights):.6f}")
    print(f"Max weight: {np.max(weights):.6f}")
    print(f"Min weight: {np.min(weights):.6f}")

    # Calculate pre-model accuracy and fairness loss
    pred_val_pre, _ = pre_model.pred(data.x_val)
    pred_val_pre_binary = (pred_val_pre > 0.5).astype(int)
    pre_accuracy = np.mean(pred_val_pre_binary == data.y_val)
    if args.metric == "dp":
        pre_fair_loss = loss_dp(data.x_val, data.s_val, pred_val_pre)
    else:
        pre_fair_loss = loss_ferm(pre_model.log_loss, data.x_val, data.y_val, data.s_val)

    # Calculate post-model accuracy and fairness loss
    pred_val_post, _ = post_model.pred(data.x_val)
    pred_val_post_binary = (pred_val_post > 0.5).astype(int)
    post_accuracy = np.mean(pred_val_post_binary == data.y_val)
    if args.metric == "dp":
        post_fair_loss = loss_dp(data.x_val, data.s_val, pred_val_post)
    else:
        post_fair_loss = loss_ferm(post_model.log_loss, data.x_val, data.y_val, data.s_val)
    pre_util_loss = pre_model.log_loss(data.x_val, data.y_val)
    post_util_loss = post_model.log_loss(data.x_val, data.y_val)

    # Print results for debugging
    print("\nModel Performance:")
    print(f"Pre-model accuracy: {pre_accuracy:.4f}")    
    print(f"Pre-model fairness loss: {pre_fair_loss:.4f}")
    # print(f"Pre-model utility loss: {pre_util_loss:.4f}")
    print(f"Post-model accuracy: {post_accuracy:.4f}")
    print(f"Post-model fairness loss: {post_fair_loss:.4f}")
    # print(f"Post-model utility loss: {post_util_loss:.4f}")

    # Call plot_results to draw
    plot_results(data, x_train_pca, pre_model, post_model, x_train_pca, 
                fair_infl if weights is None else weights, args)