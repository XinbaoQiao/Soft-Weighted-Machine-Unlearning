import os
import time
import argparse
import numpy as np
from typing import Sequence

from dataset import fetch_data, DataTemplate
from dataset2 import fetch_data2, DataTemplate2
from eval import Evaluator
from model import LogisticRegression, NNLastLayerIF, MLPClassifier, ResNetLastLayerIF, ResNetBase
from fair_fn import grad_ferm, grad_dp, loss_ferm, loss_dp
from utils import fix_seed, save2csv
from robust_fn import grad_robust, calc_robust_acc
from robust_fn_nn import grad_robust_nn1, calc_robust_acc_nn1

import json
import pickle
import random
import copy
import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Influence Fairness')
    parser.add_argument('--dataset', type=str, default="adult", help="name of the dataset")
    parser.add_argument('--num_data', type=int, default=30162, help="number of data")
    parser.add_argument('--metric', type=str, default="dp", help="eop or dp")
    parser.add_argument('--seed', type=float, default=1, help="random seed")
    parser.add_argument('--save_model', type=str, default="n", help="y/n")
    parser.add_argument('--type', type=str, default="util", help="util/fair/")
    parser.add_argument('--strategy', type=str, default="dec", help="inc/dec/random")
    parser.add_argument('--points_to_delete', type=int, default=-1, help="points to delete (num or index)")
    parser.add_argument('--random_seed', type=int, default=1, help="seed for random strategy")
    parser.add_argument('--only_pre', type=str, default="y", help="y/n")
    parser.add_argument('--model_type', type=str, default="nn", help="logreg/nn/resnet")
    parser.add_argument('--split_ratio', type=float, default=0.5, help="ratio of data for training feature extractor")

    args = parser.parse_args()

    return args


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

    if args.model_type == 'nn':
        model = NNLastLayerIF(input_dim=data.dim, base_model_cls=MLPClassifier, l2_reg=1e-4)
    elif args.model_type == 'resnet':
        model = ResNetLastLayerIF(input_dim=data.dim, base_model_cls=ResNetBase, l2_reg=1e-4)

    val_evaluator, test_evaluator = Evaluator(data.s_val, "val"), Evaluator(data.s_test, "test")

    """ Split data for feature extractor and linear layer training """
    num_samples = len(data.x_train)
    indices = np.random.permutation(num_samples)
    split_idx = int(num_samples * args.split_ratio)
    
    # Data for feature extractor training
    feature_train_indices = indices[:split_idx]
    x_train_feature = data.x_train[feature_train_indices]
    y_train_feature = data.y_train[feature_train_indices]
    
    # Data for linear layer training
    linear_train_indices = indices[split_idx:]
    x_train_linear = data.x_train[linear_train_indices]
    y_train_linear = data.y_train[linear_train_indices]

    """ Train feature extractor """
    model.fit_feature_extractor(x_train_feature, y_train_feature)

    """ Train linear layer """
    model.fit_linear_layer(x_train_linear, y_train_linear)
    
    # Synchronize weights after training
    w = model.model.last_fc.weight.data.cpu().numpy().flatten()
    model.w = w.copy()  # Update logistic regression weights
    b = 0

    if args.metric == "eop":
        ori_fair_loss_val = loss_ferm(model.log_loss, data.x_val, data.y_val, data.s_val)
    elif args.metric == "dp":
        pred_val, _ = model.pred(data.x_val)
        ori_fair_loss_val = loss_dp(data.x_val, data.s_val, pred_val)
    else:
        raise ValueError
    ori_util_loss_val = model.log_loss(data.x_val, data.y_val)

    # 根据模型类型选择不同的鲁棒性计算函数
    if args.model_type == 'logreg':
        val_rob_acc, val_rob_loss = calc_robust_acc(model, data.x_val, data.y_val, 'val', 'pre')
        robust_loss_total_grad = grad_robust(model, data.x_val, data.y_val)
    else:  # 'nn' or 'resnet'
        val_rob_acc, val_rob_loss = calc_robust_acc_nn1(model, data.x_val, data.y_val, 'val', 'pre')
        robust_loss_total_grad = grad_robust_nn1(model, data.x_val, data.y_val)
    ori_robust_loss_val = val_rob_loss

    """ compute the influence and save data """
    # Only compute influence for data used in linear layer training
    pred_train, _ = model.pred(x_train_linear)

    train_total_grad, train_indiv_grad = model.grad(x_train_linear, y_train_linear)
    util_loss_total_grad, acc_loss_indiv_grad = model.grad(data.x_val, data.y_val)
    if args.metric == "eop":
        fair_loss_total_grad = grad_ferm(model.grad, data.x_val, data.y_val, data.s_val)
    elif args.metric == "dp":
        fair_loss_total_grad = grad_dp(model.grad_pred, data.x_val, data.s_val)
    else:
        raise ValueError

    hess = model.hess(x_train_linear)
    util_grad_hvp = model.get_inv_hvp(hess, util_loss_total_grad)
    fair_grad_hvp = model.get_inv_hvp(hess, fair_loss_total_grad)
    robust_grad_hvp = model.get_inv_hvp(hess, robust_loss_total_grad)

    util_pred_infl = train_indiv_grad.dot(util_grad_hvp)
    fair_pred_infl = train_indiv_grad.dot(fair_grad_hvp)
    robust_pred_infl = train_indiv_grad.dot(robust_grad_hvp)

    _, pred_label_val = model.pred(data.x_val)
    _, pred_label_test = model.pred(data.x_test)

    val_res = val_evaluator(data.y_val, pred_label_val)
    test_res = test_evaluator(data.y_test, pred_label_test)

    # Save the indices for post processing
    np.save('linear_train_indices.npy', linear_train_indices)
    
    return val_res, test_res, ori_util_loss_val, ori_fair_loss_val, ori_robust_loss_val, util_pred_infl, fair_pred_infl, robust_pred_infl, linear_train_indices


def post_main(args):
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

    if args.model_type == 'nn':
        model = NNLastLayerIF(input_dim=data.dim, base_model_cls=MLPClassifier, l2_reg=1e-4)
    elif args.model_type == 'resnet':
        model = ResNetLastLayerIF(input_dim=data.dim, base_model_cls=ResNetBase, l2_reg=1e-4)

    val_evaluator, test_evaluator = Evaluator(data.s_val, "val"), Evaluator(data.s_test, "test")

    # Load the indices used for linear layer training
    linear_train_indices = np.load('linear_train_indices.npy')
    
    # Get the data used for linear layer training
    x_train_linear = data.x_train[linear_train_indices]
    y_train_linear = data.y_train[linear_train_indices]

    # Remove the point to delete from linear layer training data
    delete_idx = args.points_to_delete  # This index is relative to linear_train_indices
    x_train_linear = np.delete(x_train_linear, delete_idx, axis=0)
    y_train_linear = np.delete(y_train_linear, delete_idx, axis=0)

    # Train feature extractor with the same data as in pre_main
    num_samples = len(data.x_train)
    indices = np.random.permutation(num_samples)
    split_idx = int(num_samples * args.split_ratio)
    x_train_feature = data.x_train[indices[:split_idx]]
    y_train_feature = data.y_train[indices[:split_idx]]
    model.fit_feature_extractor(x_train_feature, y_train_feature)

    # Train only the linear layer with the remaining data (excluding the deleted point)
    model.fit_linear_layer(x_train_linear, y_train_linear)
    
    # Synchronize weights after training
    w = model.model.last_fc.weight.data.cpu().numpy().flatten()
    model.w = w.copy()  # Update logistic regression weights
    b = 0

    if args.metric == "eop":
        new_fair_loss_val = loss_ferm(model.log_loss, data.x_val, data.y_val, data.s_val)
    elif args.metric == "dp":
        pred_val, _ = model.pred(data.x_val)
        new_fair_loss_val = loss_dp(data.x_val, data.s_val, pred_val)
    else:
        raise ValueError
    new_util_loss_val = model.log_loss(data.x_val, data.y_val)

    if args.model_type == 'logreg':
        val_rob_acc, val_rob_loss = calc_robust_acc(model, data.x_val, data.y_val, 'val', 'pre')
    else:   # 'nn' or 'resnet'
        val_rob_acc, val_rob_loss = calc_robust_acc_nn1(model, data.x_val, data.y_val, 'val', 'pre')
    new_robust_loss_val = val_rob_loss

    _, pred_label_val = model.pred(data.x_val)
    _, pred_label_test = model.pred(data.x_test)

    val_res = val_evaluator(data.y_val, pred_label_val)
    test_res = test_evaluator(data.y_test, pred_label_test)

    return val_res, test_res, new_util_loss_val, new_fair_loss_val, new_robust_loss_val


if __name__ == "__main__":
    args = parse_args()

    os.makedirs('results/SEC-5-1-0-2', exist_ok=True)

    pre_val_res, pre_test_res, ori_util_loss_val, ori_fair_loss_val, ori_robust_loss_val, util_pred_infl, fair_pred_infl, robust_pred_infl, linear_train_indices = pre_main(args)

    util_diff = []
    fair_diff = []
    robust_diff = []
    
    # Determine number of points to calculate based on points_to_delete
    if args.points_to_delete == -1:
        # Calculate all points used for linear layer training
        indices_to_check = range(len(linear_train_indices))
    else:
        # Only calculate specified number of points
        indices_to_check = range(args.points_to_delete)
    
    # Calculate actual values for specified sample points
    print(f"\nCalculating actual values for first {len(indices_to_check)} points")
    for i, idx in enumerate(indices_to_check):
        print(f"Processing point {i} (of {len(indices_to_check)} points)")
        # Use actual index from linear_train_indices
        args.points_to_delete = idx
        post_val_res, post_test_res, post_util_loss_val, post_fair_loss_val, post_robust_loss_val = post_main(args)
        
        # Record differences
        util_diff.append(post_util_loss_val - ori_util_loss_val)
        fair_diff.append(post_fair_loss_val - ori_fair_loss_val)
        robust_diff.append(post_robust_loss_val - ori_robust_loss_val)

    # Save correspondence between actual values and influence values
    results_df = pd.DataFrame({
        'Index': list(range(len(util_diff))),
        'Actual Fair Difference': fair_diff,
        'Influence Fair Difference': fair_pred_infl[:len(fair_diff)],
        'Correlation': np.corrcoef(fair_diff, fair_pred_infl[:len(fair_diff)])[0,1]
    })

    print("\nCorrelation Analysis:")
    print(f"Fair metric correlation: {np.corrcoef(fair_diff, fair_pred_infl[:len(fair_diff)])[0,1]:.4f}")

    # Save influence values
    print("\nSaving influence values:")
    print(f"util_pred_infl shape: {util_pred_infl.shape}")
    print(f"fair_pred_infl shape: {fair_pred_infl.shape}")
    print(f"robust_pred_infl shape: {robust_pred_infl.shape}")

    # Print array length information for debugging
    print("\nArray lengths:")
    print(f"util_diff length: {len(util_diff)}")
    print(f"fair_diff length: {len(fair_diff)}")
    print(f"robust_diff length: {len(robust_diff)}")
    print(f"util_pred_infl length: {len(util_pred_infl)}")
    print(f"fair_pred_infl length: {len(fair_pred_infl)}")
    print(f"robust_pred_infl length: {len(robust_pred_infl)}")
    print(f"points_to_delete: {args.points_to_delete}")

    # Ensure all array lengths are consistent
    util_df = pd.DataFrame({
        'Index': list(range(len(util_diff))),
        'Actual Util Difference': util_diff,
        'Influence Util Difference': util_pred_infl[:len(util_diff)]
    })
    fair_df = pd.DataFrame({
        'Index': list(range(len(fair_diff))),
        'Actual Fair Difference': fair_diff,
        'Influence Fair Difference': fair_pred_infl[:len(fair_diff)]
    })
    robust_df = pd.DataFrame({
        'Index': list(range(len(robust_diff))),
        'Actual Robust Difference': robust_diff,
        'Influence Robust Difference': robust_pred_infl[:len(robust_diff)]
    })

    # Save CSV files
    util_df.to_csv('results/SEC-5-1-0-2/util_actual.csv', index=False)
    fair_df.to_csv('results/SEC-5-1-0-2/fair_{}_actual.csv'.format(args.metric), index=False)
    robust_df.to_csv('results/SEC-5-1-0-2/robust_actual.csv', index=False)

    np.save('results/SEC-5-1-0-2/util_infl_full.npy', util_pred_infl)
    np.save('results/SEC-5-1-0-2/fair_{}_infl_full.npy'.format(args.metric), fair_pred_infl)
    np.save('results/SEC-5-1-0-2/robust_infl_full.npy', robust_pred_infl)