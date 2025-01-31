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
from utils import fix_seed, save2csv, AverageMeter, accuracy  # 添加需要的utils函数
from robust_fn import grad_robust, calc_robust_acc
from robust_fn_nn import grad_robust_nn, calc_robust_acc_nn
from unlearn import CU_k, EU_k, GA, IF  # Import directly from unlearn package

import json
import pickle
import random
import copy
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch.nn as nn

# Clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser(description='Influence Fairness')
    # Required dataset and training parameters
    parser.add_argument('--dataset', type=str, default="bank", help="name of the dataset")
    parser.add_argument('--num_data', type=int, default=-1, help="number of data")
    parser.add_argument('--metric', type=str, default="eop", help="dp/eop/robust")
    parser.add_argument('--seed', type=float, default=0, help="random seed")
    parser.add_argument('--model_type', type=str, default="nn", help="nn/resnet")
    
    # Unlearning related parameters
    parser.add_argument('--gpu_id', type=int, default=1, help="GPU ID to use")
    parser.add_argument('--lr', type=float, default=0.005, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-3, help="weight decay")
    parser.add_argument('--batch_size', type=int, default=None, help="batch size (default: size of training set)")
    parser.add_argument('--weight_scheme', type=str, default="hard", help="hard/soft-2.1/soft-2.2/soft-2.3")
    parser.add_argument('--feature_data_ratio', type=float, default=0.01, help="ratio of data for training feature extractor")
    parser.add_argument('--arch', type=str, default='mlp', help="network architecture")
    
    args = parser.parse_args()
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        args.device = f"cuda:{args.gpu_id}"
    else:
        args.device = "cpu"
        
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
    args.num_data = 15000 if args.dataset == 'bank' else -1
    # Limit training set size
    if args.num_data > 0 and args.num_data < len(data.x_train):
        indices = np.random.choice(len(data.x_train), args.num_data, replace=False)
        data.x_train = data.x_train[indices]
        data.y_train = data.y_train[indices]
        data.s_train = data.s_train[indices]
        print(f"Limiting training set size to {args.num_data} samples")
    
    model = LogisticRegression(l2_reg=data.l2_reg)

    if args.model_type == 'nn':
        model = NNLastLayerIF(input_dim=data.dim, base_model_cls=MLPClassifier, l2_reg=1e-4, device=args.device)
    elif args.model_type == 'resnet':
        model = ResNetLastLayerIF(input_dim=data.dim, base_model_cls=ResNetBase, l2_reg=1e-4, device=args.device)

    val_evaluator, test_evaluator = Evaluator(data.s_val, "val"), Evaluator(data.s_test, "test")

    """ Split data for feature extractor and linear layer training """
    num_samples = len(data.x_train)
    indices = np.random.permutation(num_samples)
    args.feature_data_ratio = 0.2 if args.dataset == 'celeba' else 0.01
    split_idx = int(num_samples * args.feature_data_ratio)
    
    # Data for feature extractor training
    feature_train_indices = indices[:split_idx]
    x_train_feature = data.x_train[feature_train_indices]
    y_train_feature = data.y_train[feature_train_indices]
    
    # Data for linear layer training
    linear_train_indices = indices[split_idx:]
    x_train_linear = data.x_train[linear_train_indices]
    y_train_linear = data.y_train[linear_train_indices]
    
    print(f"\nData split:")
    print(f"Feature extractor training: {len(x_train_feature)} samples")
    print(f"Linear layer training: {len(x_train_linear)} samples")

    """ Train feature extractor """
    print("\nTraining feature extractor...")
    model.fit_feature_extractor(x_train_feature, y_train_feature)

    """ Train linear layer """
    print("\nTraining last layer...")
    model.fit_linear_layer(x_train_linear, y_train_linear)

    # Save indices for later use
    np.save('results/SEC-5-2-0-3/linear_train_indices.npy', linear_train_indices)

    if args.metric == "eop":
        ori_fair_loss_val = loss_ferm(model.log_loss, data.x_val, data.y_val, data.s_val)
        fair_loss_total_grad = grad_ferm(model.grad, data.x_val, data.y_val, data.s_val)
    elif args.metric == "dp":
        pred_val, _ = model.pred(data.x_val)
        ori_fair_loss_val = loss_dp(data.x_val, data.s_val, pred_val)
        fair_loss_total_grad = grad_dp(model.grad_pred, data.x_val, data.s_val)
    elif args.metric == "robust":
        # For robust metric, use robustness loss
        if args.model_type == 'logreg':
            val_rob_acc, ori_fair_loss_val = calc_robust_acc(model, data.x_val, data.y_val, 'val', 'pre')
            fair_loss_total_grad = grad_robust(model, data.x_val, data.y_val)
        else:  # 'nn' or 'resnet'
            val_rob_acc, ori_fair_loss_val = calc_robust_acc_nn(model, data.x_val, data.y_val, 'val', 'pre')
            fair_loss_total_grad = grad_robust_nn(model, data.x_val, data.y_val)
    else:
        raise ValueError(f"Unsupported metric: {args.metric}")
    ori_util_loss_val = model.log_loss(data.x_val, data.y_val)

    # Choose robustness calculation function based on model type
    if args.model_type == 'logreg':
        val_rob_acc, val_rob_loss = calc_robust_acc(model, data.x_val, data.y_val, 'val', 'pre')
        robust_loss_total_grad = grad_robust(model, data.x_val, data.y_val)
    else:  # 'nn' or 'resnet'
        val_rob_acc, val_rob_loss = calc_robust_acc_nn(model, data.x_val, data.y_val, 'val', 'pre')
        robust_loss_total_grad = grad_robust_nn(model, data.x_val, data.y_val)
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
    elif args.metric == "robust":
        # For robust metric, use robustness gradient
        if args.model_type == 'logreg':
            fair_loss_total_grad = grad_robust(model, data.x_val, data.y_val)
        else:  # 'nn' or 'resnet'
            fair_loss_total_grad = grad_robust_nn(model, data.x_val, data.y_val)
    else:
        raise ValueError(f"Unsupported metric: {args.metric}")

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

    return val_res, test_res, ori_util_loss_val, ori_fair_loss_val, ori_robust_loss_val, util_pred_infl, fair_pred_infl, robust_pred_infl, model, data, x_train_linear, y_train_linear, val_rob_acc


def create_weighted_dataloaders(data, weights, x_train_linear, y_train_linear, args):
    """Create data loaders with weights and sensitive attributes for unlearning"""
    
    # Convert numpy arrays to torch tensors
    x_train = torch.FloatTensor(x_train_linear)  # Use linear layer training data only
    y_train = torch.LongTensor(y_train_linear)   # Use linear layer training data only
    
    # Get corresponding sensitive attributes
    linear_train_indices = np.load('results/SEC-5-2-0-3/linear_train_indices.npy')
    s_train = torch.LongTensor(data.s_train[linear_train_indices])
    
    weights = torch.FloatTensor(weights)
    # Save sample information to CSV file
    import pandas as pd
    
    # Get indices of linear training data
    linear_train_indices = np.load('results/SEC-5-2-0-3/linear_train_indices.npy')
    
    # Create DataFrame
    sample_info = pd.DataFrame({
        'index': linear_train_indices,
        'weight': weights.numpy(),
        'util_infl': util_pred_infl,
        'fair_infl': fair_pred_infl, 
        'robust_infl': robust_pred_infl
    })
    
    # Sort by metric influence
    if args.metric == 'robust':
        sample_info = sample_info.sort_values('robust_infl', ascending=False)
    else:
        sample_info = sample_info.sort_values('fair_infl', ascending=False)
        
    # Save to CSV file
    weight_type = "hard" if args.weight_scheme == "hard" else "soft" if args.weight_scheme.startswith("soft-") else args.weight_scheme
    save_path = f'results/SEC-5-2-0-3/sample_info_{weight_type}.csv'
    sample_info.to_csv(save_path, index=False)
    print(f"Sample information saved to: {save_path}")

    
    # If batch_size is None, use the size of the training set
    if args.batch_size is None:
        args.batch_size = len(x_train_linear)
        print(f"Using full dataset as batch size: {args.batch_size}")
    
    # Create complete training dataset
    train_dataset = TensorDataset(x_train, y_train, s_train, weights)
    
    # Create validation and test datasets
    val_dataset = TensorDataset(
        torch.FloatTensor(data.x_val),
        torch.LongTensor(data.y_val),
        torch.LongTensor(data.s_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(data.x_test),
        torch.LongTensor(data.y_test),
        torch.LongTensor(data.s_test)
    )
    
    if args.weight_scheme == "hard":
        # Separate forget and retain data based on weights
        forget_mask = weights == -1  # Weights of -1 are data to be forgotten
        retain_mask = weights == 0  # Weights of 0 are data to be retained
        
        # Create forget and retain datasets
        forget_dataset = TensorDataset(
            x_train[forget_mask], 
            y_train[forget_mask], 
            s_train[forget_mask], 
            weights[forget_mask]
        )
        
        retain_dataset = TensorDataset(
            x_train[retain_mask], 
            y_train[retain_mask], 
            s_train[retain_mask], 
            weights[retain_mask]
        )
        
        data_loaders = {
            "train": DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True),
            "val": DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False),
            "test": DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False),
            "forget": DataLoader(forget_dataset, batch_size=args.batch_size, shuffle=True),
            "retain": DataLoader(retain_dataset, batch_size=args.batch_size, shuffle=True)
        }
    else:  # Soft scheme
        data_loaders = {
            "train": DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True),
            "val": DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False),
            "test": DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False),
            "forget": DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True),  # Soft scheme uses same dataset
            "retain": DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)   # Differentiate by weights
        }
    
    if "val" not in data_loaders:
        print("Warning: Validation loader not found in data_loaders")
    
    return data_loaders, weights


def calculate_weights_scheme_2_0(fair_infl, util_infl, ori_fair_loss_val, ori_util_loss_val):
    """Calculate weights using scheme 2.0"""
    iter_num = 50
    lambda_candidates = np.logspace(-3, 1, num=iter_num)
    best_lambda = None
    best_fair_loss = float('inf')
    best_weights = None
    
    for lambda_reg in lambda_candidates:
        
        weights = (fair_infl ) / (2 * lambda_reg)
        
        # Check weight validity
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            continue
            
        # Calculate new fairness loss
        new_fair_loss = -np.dot(weights.T, fair_infl)
        if abs(new_fair_loss) < abs(best_fair_loss):
            best_fair_loss = new_fair_loss
            best_lambda = lambda_reg
            best_weights = weights
    
    # If no suitable weights are found, use uniform weights
    if best_weights is None:
        print("Warning: Invalid weights found, using uniform weights")
        weights = np.zeros_like(fair_infl)
        weights[(fair_pred_infl < 0) & (util_pred_infl < 0)] = -1
        best_weights = weights
    
    return best_weights, best_lambda

def calculate_weights_scheme_2_1(fair_infl, util_infl, ori_fair_loss_val, ori_util_loss_val):
    """Calculate weights using scheme 2.1 with fixed lambda"""
    lambda_reg = 0.1  # Fixed lambda value
    
    beta2 = 2 * lambda_reg * abs(ori_fair_loss_val) / (np.linalg.norm(fair_infl)**2)
    weights = (fair_infl * abs(ori_fair_loss_val)) / (np.linalg.norm(fair_infl)**2)
    
    return weights, lambda_reg

def calculate_weights_scheme_2_2(fair_infl, util_infl, ori_fair_loss_val, ori_util_loss_val):
    """Calculate weights using scheme 2.2"""
    iter_num = 50
    lambda_candidates = np.logspace(-3, 1, num=iter_num)
    best_lambda = None
    best_fair_loss = float('inf')
    best_weights = None
    
    for lambda_reg in lambda_candidates:
        beta1 = -np.dot(fair_infl.T, util_infl) / (np.linalg.norm(util_infl)**2)
        
        # Remove beta1 < 0 check, as negative values are also valid
        weights = (fair_infl + beta1 * util_infl) / (2 * lambda_reg)
        
        # Check weight validity
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            continue
        # Calculate new fairness loss
        new_fair_loss = -np.dot(weights.T, fair_infl)
        if abs(new_fair_loss) < abs(best_fair_loss):
            best_fair_loss = new_fair_loss
            best_lambda = lambda_reg
            best_weights = weights
    
    # If no suitable weights are found, use uniform weights
    if best_weights is None:
        print("Warning: Invalid weights found, using uniform weights")
        weights = np.zeros_like(fair_infl)
        weights[(fair_pred_infl < 0) & (util_pred_infl < 0)] = -1
        best_weights = weights
    
    return best_weights, best_lambda

def calculate_weights_scheme_2_3(fair_infl, util_infl, ori_fair_loss_val, ori_util_loss_val):
    """Calculate weights using scheme 2.3 with fixed lambda"""
    lambda_reg = 0.1  # Fixed lambda value
    
    denominator = (np.linalg.norm(fair_infl)**2 * np.linalg.norm(util_infl)**2 - 
                  (np.dot(fair_infl, util_infl))**2)
    
    # Add numerical stability check
    if abs(denominator) < 1e-10:
        print("Warning: Denominator too small, using uniform weights")
        return np.ones_like(fair_infl) / len(fair_infl), lambda_reg
        
    beta1 = -(2 * lambda_reg * abs(ori_fair_loss_val) * np.linalg.norm(util_infl)**2 * 
             np.dot(fair_infl.T, util_infl)) / (denominator * np.linalg.norm(util_infl)**2)
    beta2 = (2 * lambda_reg * abs(ori_fair_loss_val) * np.linalg.norm(util_infl)**2) / denominator - 1
    
    weights = ((1 + beta2) * fair_infl + beta1 * util_infl) / (2 * lambda_reg)
    
    # Check for invalid weights
    if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
        print("Warning: Invalid weights found, using uniform weights")
        weights = np.zeros_like(fair_infl)
        weights[(fair_pred_infl < 0) & (util_pred_infl < 0)] = -1
        
    return weights, lambda_reg

if __name__ == "__main__":
    args = parse_args()

    #######################
    args.dataset = 'adult'
    args.model_type = 'nn'  
    args.weight_scheme = 'hard'
    args.weight_scheme = 'soft-2.3'
    args.lr = 0.01
    args.weight_decay = 1e-3
    args.unlearn_epochs = 30
    args.unlearn_method = "euk" 
    #######################
    # Create results directory if it doesn't exist
    os.makedirs('results/SEC-5-2-0-3', exist_ok=True)
    # Phase 1: Train original model and calculate influence weights
    print("Phase 1: Training original model and calculating influence weights...")
    pre_val_res, pre_test_res, ori_util_loss_val, ori_fair_loss_val, ori_robust_loss_val, util_pred_infl, fair_pred_infl, robust_pred_infl, model, data, x_train_linear, y_train_linear, val_rob_acc = pre_main(args)

    # Create method-specific directory
    base_dir = 'results/SEC-5-2-0-3'
    method_dir = os.path.join(base_dir, args.unlearn_method.upper())
    os.makedirs(method_dir, exist_ok=True)

    # Save to CSV file
    weight_type = "hard" if args.weight_scheme == "hard" else "soft" if args.weight_scheme.startswith("soft-") else args.weight_scheme
    model_type = "logreg" if args.model_type == "logreg" else "nn"

    # Store pre-unlearning metrics
    pre_metrics = {
        'utility': ori_util_loss_val,
        'fairness': ori_fair_loss_val if args.metric == "dp" else None,
        'robustness': ori_robust_loss_val
    }

    # Print pre-unlearning metrics
    print("\n------------------------------ Pre-unlearning Results")
    _, pred_label_val = model.pred(data.x_val)
    pre_accuracy = np.mean(pred_label_val == data.y_val) * 100
    print(f"Utility - Val Accuracy: {pre_accuracy:.2f}%")
    if args.metric == "dp":
        print(f"Fairness (DP) - Val Loss: {ori_fair_loss_val:.4f}")
    elif args.metric == "eop":
        print(f"Fairness (EOP) - Val Loss: {ori_fair_loss_val:.4f}")
    else:
        print(f"Robustness - Val Accuracy: {val_rob_acc:.2f}%")

    args.weight_scheme = 'soft-2.1' if args.dataset == 'nlp' and args.unlearn_method == 'if' else args.weight_scheme
    # Calculate weights based on scheme
    print("\nPhase 2: Calculating weights using scheme {}...".format(args.weight_scheme))
    # Choose task influence based on task type
    if args.metric == "dp" or args.metric == "eop":
        task_infl = fair_pred_infl
    else:  # robust
        task_infl = robust_pred_infl
    
    if args.weight_scheme == "hard":
        # Initialize weights
        weights = np.zeros_like(task_infl)
        # Sort indices based on task influence
        sorted_indices = np.argsort(task_infl)
        # Select top k% as forgetting points (weight = 0)
        k= {('bank'): 0.001}.get(args.dataset, 0.2)   # Modified to if method
        num_forget = int(len(weights) * k)
        forget_indices = sorted_indices[:num_forget]
        retain_indices = sorted_indices[num_forget:]
        
        # Set weights: 1 for retain, 0 for forget
        weights[retain_indices] = 0
        weights[forget_indices] = -1
        
        print(f"Number of points to forget: {num_forget}")
        print(f"Number of points to retain: {len(retain_indices)}")
        
    elif args.weight_scheme == "soft-2.0":
        weights, best_lambda = calculate_weights_scheme_2_0(
            task_infl, util_pred_infl, 
            ori_fair_loss_val if (args.metric == "dp" or args.metric == "eop") else ori_robust_loss_val, 
            ori_util_loss_val)

    elif args.weight_scheme == "soft-2.1":
        weights, best_lambda = calculate_weights_scheme_2_1(
            task_infl, util_pred_infl, 
            ori_fair_loss_val if (args.metric == "dp" or args.metric == "eop") else ori_robust_loss_val, 
            ori_util_loss_val)
        
    elif args.weight_scheme == "soft-2.2":
        weights, best_lambda = calculate_weights_scheme_2_2(
            task_infl, util_pred_infl, 
            ori_fair_loss_val if (args.metric == "dp" or args.metric == "eop") else ori_robust_loss_val, 
            ori_util_loss_val)
        
    elif args.weight_scheme == "soft-2.3":
        weights, best_lambda = calculate_weights_scheme_2_3(
            task_infl, util_pred_infl, 
            ori_fair_loss_val if (args.metric == "dp" or args.metric == "eop") else ori_robust_loss_val, 
            ori_util_loss_val)

    # Create weighted data loaders
    data_loaders, retain_weights = create_weighted_dataloaders(data, weights, x_train_linear, y_train_linear, args)

    # Initialize criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss(reduction='none')  # Use reduction='none' to apply weights manually

    # Ensure model is initialized
    if not hasattr(model, 'model'):
        # Use random input to initialize model
        dummy_input = torch.randn(1, data.dim).to(args.device)
        model.model = model.base_model_cls(model.input_dim).to(args.device)
        _ = model.model(dummy_input)

    # Freeze feature extraction layers
    for param in model.model.parameters():
        param.requires_grad = False

    # Create PyTorch version of linear layer for unlearning
    feature_dim = model.model.emb(torch.randn(1, data.dim).to(args.device)).shape[1]
    last_layer = torch.nn.Linear(feature_dim, 1, bias=False).to(args.device)
    # Initialize with original logistic regression weights
    last_layer.weight.data = torch.from_numpy(model.logistic.coef_).float().to(args.device)

    # Optimize only the last linear layer
    optimizer = torch.optim.Adam(last_layer.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Modify model's forward method to use new linear layer
    def new_forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        with torch.no_grad():
            features = self.model.emb(x)
        return torch.sigmoid(last_layer(features)).squeeze()

    model.forward = new_forward.__get__(model)

    # Execute unlearning
    if args.unlearn_method.lower() == "euk":
        unlearn_model, unlearn_metrics = EU_k(
            data_loaders=data_loaders,
            model=model,
            criterion=None,
            optimizer=None,
            epoch=0,
            args=args
        )
    elif args.unlearn_method.lower() == "cuk":
        unlearn_model, unlearn_metrics = CU_k(
            data_loaders=data_loaders,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=0,
            args=args
        )
    elif args.unlearn_method.lower() == "if":
        unlearn_model, unlearn_metrics = IF(
            data_loaders=data_loaders,
            model=model,
            criterion=None,
            optimizer=None,
            epoch=0,
            args=args,
            data=data
        )
    else:
        raise ValueError(f"Unsupported unlearning method: {args.unlearn_method}")

    # Update original logistic regression weights
    model.logistic.coef_ = model.last_fc_weight.reshape(1, -1)  # Use updated weights directly

    # Print post-unlearning metrics
    print("\n------------------------------ Post-unlearning Results")
    final_metrics = unlearn_metrics[-1]  # Get metrics from last epoch
    print(f"Utility - Accuracy: {final_metrics['val']['utility']['accuracy']:.2f}%")
    if args.metric == "dp":
        print(f"Fairness (DP) - Loss: {final_metrics['val']['fairness']['dp']:.4f}")
    elif args.metric == "eop":
        print(f"Fairness (EOP) - Loss: {final_metrics['val']['fairness']['eop']:.4f}")
    elif 'robustness' in final_metrics['val']:
        print(f"Robustness - Accuracy: {final_metrics['val']['robustness']['accuracy']:.2f}%")

    # Print comparison of pre and post unlearning metrics
    print("\n------------------------------ Unlearning Results Comparison")
    print("Metric               Pre-unlearning    Post-unlearning    Improvement")
    print("-" * 65)
    
    # Accuracy comparison
    acc_improve = final_metrics['val']['utility']['accuracy'] - pre_accuracy
    print(f"Utility Accuracy:    {pre_accuracy:.2f}%         {final_metrics['val']['utility']['accuracy']:.2f}%         {acc_improve:>7.2f}%")
    
    # Save results to CSV file in method-specific folder
    base_dir = 'results/SEC-5-2-0-3'
    method_dir = os.path.join(base_dir, args.unlearn_method.upper())
    os.makedirs(method_dir, exist_ok=True)
    
    # Format results in a more readable way
    formatted_results = {
        'Metric': [],
        'Pre-unlearning': [],
        'Post-unlearning': [],
        'Improvement': []
    }
    
    # Add utility metrics
    formatted_results['Metric'].append('Utility Accuracy')
    formatted_results['Pre-unlearning'].append(f"{pre_accuracy:.2f}%")
    formatted_results['Post-unlearning'].append(f"{final_metrics['val']['utility']['accuracy']:.2f}%")
    formatted_results['Improvement'].append(f"{acc_improve:+.2f}%")
    
    # Fairness or Robustness comparison
    if args.metric == "dp":
        # Preserve original sign for display, but use absolute value for relative change calculation
        pre_dp = ori_fair_loss_val
        post_dp = final_metrics['val']['fairness']['dp']
        # Calculate improvement percentage = (|original value| - |new value|) / |original value| * 100
        dp_improve = ((abs(pre_dp) - abs(post_dp)) / abs(pre_dp)) * 100 if abs(pre_dp) != 0 else 0
        
        print(f"Fairness (DP)         {pre_dp:.6f}          {post_dp:.6f}       {dp_improve:+.2f}%")
        
        formatted_results['Metric'].append('Fairness (DP)')
        formatted_results['Pre-unlearning'].append(f"{pre_dp:.6f}")  # Preserve original sign
        formatted_results['Post-unlearning'].append(f"{post_dp:.6f}")  # Preserve original sign
        formatted_results['Improvement'].append(f"{dp_improve:+.2f}%")  # Display percentage improvement
    elif args.metric == "eop":
        fair_improve = ((abs(ori_fair_loss_val) - abs(final_metrics['val']['fairness']['eop'])) / abs(ori_fair_loss_val)) * 100 if abs(ori_fair_loss_val) != 0 else 0
        print(f"Fairness (EOP):      {ori_fair_loss_val:.6f}          {final_metrics['val']['fairness']['eop']:.6f}          {fair_improve:>7.2f}%")
        
        formatted_results['Metric'].append('Fairness (EOP)')
        formatted_results['Pre-unlearning'].append(f"{ori_fair_loss_val:.4f}")
        formatted_results['Post-unlearning'].append(f"{final_metrics['val']['fairness']['eop']:.4f}")
        formatted_results['Improvement'].append(f"{fair_improve:+.2f}%")
    else:
        rob_acc_improve = final_metrics['val']['robustness']['accuracy'] - val_rob_acc
        print(f"Robustness Acc:      {val_rob_acc:.2f}%         {final_metrics['val']['robustness']['accuracy']:.2f}%         {rob_acc_improve:>7.2f}%")
        
        formatted_results['Metric'].append('Robustness Accuracy')
        formatted_results['Pre-unlearning'].append(f"{val_rob_acc:.2f}%")
        formatted_results['Post-unlearning'].append(f"{final_metrics['val']['robustness']['accuracy']:.2f}%")
        formatted_results['Improvement'].append(f"{rob_acc_improve:+.2f}%")
    
    # Save formatted results
    result_filename = f"{args.dataset}_{args.metric}_{weight_type}_{model_type}_comparison.csv"
    result_path = os.path.join(method_dir, result_filename)
    
    df = pd.DataFrame(formatted_results)
    df.to_csv(result_path, index=False)
    
    print(f"\nResults saved to {result_path}")
    print("\nSaved results format:")
    print(df.to_string(index=False))

    # Save hyperparameters (can be removed code block)
    hyperparams = vars(args)  # Convert args object to dictionary containing all parameters
    
    # Remove parameters not to be saved (e.g., device)
    if 'device' in hyperparams:
        del hyperparams['device']
    
    # Save hyperparameters based on task type
    task_type = 'fairness' if args.metric == "dp" else 'robustness'
    hyperparam_filename = f"{args.dataset}_{task_type}_{weight_type}_{model_type}_hyperparams.json"
    hyperparam_path = os.path.join(method_dir, hyperparam_filename)
    
    with open(hyperparam_path, 'w') as f:
        json.dump(hyperparams, f, indent=4)
    
    print(f"Hyperparameters saved to {hyperparam_path}")
