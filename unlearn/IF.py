import numpy as np
import copy
import torch
from typing import Dict, List, Any, Optional, Tuple
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from model import MLPClassifier, NNLastLayerIF
from fair_fn import loss_dp, loss_ferm
from robust_fn import calc_robust_acc
from robust_fn_nn import calc_robust_acc_nn

def IF(
    data_loaders: Dict[str, DataLoader],
    model: Any,
    criterion: Optional[torch.nn.Module],
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    args: Any,
    data: Any  
) -> Tuple[Any, List[Dict[str, Dict[str, Dict[str, float]]]]]:
    """
    Influence Function based unlearning method.
    
    Args:
        data_loaders: Dictionary containing DataLoaders for 'train', 'val', 'test', 'forget', 'retain'
        model: The model to be unlearned
        criterion: Loss function (not used in IF)
        optimizer: Optimizer (not used in IF)
        epoch: Current epoch number (not used in IF)
        args: Arguments containing training configurations
        data: Original dataset object containing x_train, x_val, x_test etc.
        
    Returns:
        tuple: (unlearned_model, metrics_list)
            - unlearned_model: The model after unlearning
            - metrics_list: List of dictionaries containing metrics for each epoch
    """
    # Collect all training data
    all_data = []
    all_targets = []
    all_weights = []
    all_sensitive = []
    
    # Use DataLoader's batch_size to ensure correct data order
    train_loader = data_loaders['train']
    for batch in train_loader:
        all_data.append(batch[0].numpy())
        all_targets.append(batch[1].numpy())
        all_sensitive.append(batch[2].numpy())
        all_weights.append(batch[3].numpy())
    
    # Merge data from all batches
    x_train = np.concatenate(all_data, axis=0)
    y_train = np.concatenate(all_targets, axis=0)
    s_train = np.concatenate(all_sensitive, axis=0)
    weights = np.concatenate(all_weights, axis=0)
    
    # Calculate gradients
    print(f"x_train shape: {x_train.shape}")
    train_total_grad, train_indiv_grad = model.grad(x_train, y_train, sample_weight=None)
    
    # Calculate Hessian
    hess = model.hess(x_train, sample_weight=None)
    
    # Calculate weighted gradient
    weighted_grad = np.sum(train_indiv_grad * weights.reshape(-1, 1), axis=0)
    
    # Calculate HVP (Hessian-vector product)
    delta = model.get_inv_hvp(hess, weighted_grad)
    
    # Create new model and update parameters
    post_model = copy.deepcopy(model)
    
    if isinstance(model, (NNLastLayerIF)):  # Neural network model
        # Update logistic regression weights
        post_model.logistic.coef_ = (model.logistic.coef_ - delta.reshape(1, -1)).copy()
        post_model.last_fc_weight = post_model.logistic.coef_.flatten().copy()
        
        # Synchronize to neural network's last layer
        with torch.no_grad():
            w_tensor = torch.from_numpy(post_model.last_fc_weight.reshape(1, -1)).to(
                dtype=post_model.model.last_fc.weight.dtype,
                device=post_model.model.last_fc.weight.device
            )
            post_model.model.last_fc.weight.data.copy_(w_tensor)
    else:  # Logistic regression model
        post_model.weight = model.weight - delta
        # Update sklearn model parameters
        post_model.model.coef_ = post_model.weight.reshape(1, -1)
        if hasattr(model.model, 'intercept_'):
            post_model.model.intercept_ = model.model.intercept_
    
    # Calculate metrics
    metrics = []
    split_names = ['train', 'val', 'test']
    
    # Evaluate model on all splits
    curr_metrics = {split: {'utility': {}, 'fairness': {}, 'robustness': {}} for split in split_names}
    
    for split in split_names:
        if split in data_loaders:
            # Use original dataset for evaluation
            if split == 'train':
                x = x_train  # Use collected training data
                y = y_train
                s = s_train
            elif split == 'val':
                x = data.x_val
                y = data.y_val
                s = data.s_val
            else:  # test
                x = data.x_test
                y = data.y_test
                s = data.s_test
            
            # Calculate accuracy
            pred = post_model.pred(x)[1]
            accuracy = (pred == y).mean() * 100
            curr_metrics[split]['utility']['accuracy'] = accuracy
            
            # Calculate fairness metric if specified
            if hasattr(args, 'metric') and args.metric in ['dp', 'eop']:
                pred_probs = post_model.pred(x)[0]
                if args.metric == 'dp':
                    fairness_loss = loss_dp(x, s, pred_probs)
                    curr_metrics[split]['fairness']['dp'] = fairness_loss
                else:
                    fairness_loss = loss_ferm(post_model.log_loss, x, y, s)
                    curr_metrics[split]['fairness']['eop'] = fairness_loss
            # Calculate robustness metric if specified
            if hasattr(args, 'metric') and args.metric == 'robust':
                if split == 'val':  # Only evaluate robustness on validation set
                    if args.model_type == 'logreg':
                        rob_acc, rob_loss = calc_robust_acc(post_model, x, y, 'val', 'post')
                    else:
                        rob_acc, rob_loss = calc_robust_acc_nn(post_model, x, y, 'val', 'post')
                    curr_metrics[split]['robustness'] = {
                        'accuracy': rob_acc * 100,
                        'loss': rob_loss
                    }
                else:  # Do not evaluate robustness on other datasets
                    curr_metrics[split]['robustness'] = {
                        'accuracy': None,
                        'loss': None
                    }
    
    metrics.append(curr_metrics)
    
    return post_model, metrics