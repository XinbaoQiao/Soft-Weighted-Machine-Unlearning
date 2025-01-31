import numpy as np
from sklearn.metrics import accuracy_score
import random
import torch

def grad_fn(clf, x, y, sample_weight=None, l2_reg=False):
    """
    Compute the gradients: grad_wo_reg = (pred - y) * x
    """
    if sample_weight is None:
        sample_weight = np.ones(x.shape[0])
    sample_weight = np.array(sample_weight)

    pred = clf.predict_proba(x)[:, 1]

    indiv_grad = x * (pred - y).reshape(-1, 1)
    weighted_indiv_grad = indiv_grad * sample_weight.reshape(-1, 1)
    if clf.fit_intercept:
        weighted_indiv_grad = np.concatenate([weighted_indiv_grad, (pred - y).reshape(-1, 1)], axis=1)

    total_grad = np.sum(weighted_indiv_grad, axis=0)

    return total_grad, weighted_indiv_grad

def grad_robust_nn(model, x_val, y_val):
    """Calculate gradient of the same loss function as robust_loss"""
    # 1. Use logistic regression weights
    clf = model.logistic
    w = clf.coef_[0]
    b = clf.intercept_
    
    # 2. Feature extraction
    x_val_emb = model.emb(x_val)
    
    # 3. Generate adversarial samples using logistic regression weights (in feature space)
    x_val_adv = []
    for i,x0 in enumerate(x_val_emb):
        perturbation = 1.3 * (np.dot(w, x0) + b) / np.dot(w, w) * w
        x1 = x0 - perturbation
        x_val_adv.append(x1)
    x_val_adv = np.array(x_val_adv)
    
    # 4. Use grad_fn to calculate gradient directly (since x_val_adv is already in feature space)
    total_loss_grad, t = grad_fn(clf, x=x_val_adv, y=y_val)
    return total_loss_grad


def calc_robust_acc_nn(model, x_val, y_val, type, stage):
    """Calculate robustness metric"""
    # 1. Use logistic regression weights
    clf = model.logistic
    w = clf.coef_[0]
    b = clf.intercept_
    
    # 2. Feature extraction
    x_val_emb = model.emb(x_val)
    
    if stage == 'pre':
        # 3. Generate adversarial samples using logistic regression weights (in feature space)
        x_val_adv = []
        for i,x0 in enumerate(x_val_emb):
            perturbation = 1.3 * (np.dot(w, x0) + b) / np.dot(w, w) * w
            x1 = x0 - perturbation
            x_val_adv.append(x1)
        x_val_adv = np.array(x_val_adv)
        # Save adversarial samples
        np.save('xadv_'+type+'.npy', x_val_adv)
    elif stage == 'post':
        # Load adversarial samples from pre stage
        x_val_adv = np.load('xadv_'+type+'.npy')
        # Check if sample counts match
        if len(x_val_adv) != len(y_val):
            return 0.0, 0.0  # Return 0 to indicate error condition

    # 4. Use logistic regression to predict directly (since x_val_adv is already in feature space)
    y_val_adv_prob = clf.predict_proba(x_val_adv)
    y_val_adv = clf.predict(x_val_adv)
    robust_acc = accuracy_score(y_val_adv, y_val)
    
    # 5. Calculate total loss for adversarial samples
    robust_loss = -np.sum(y_val * np.log(y_val_adv_prob[:, 1] + 1e-10) + 
                         (1 - y_val) * np.log(y_val_adv_prob[:, 0] + 1e-10))
    
    return robust_acc, robust_loss


def grad_robust_nn1(model, x_val, y_val):
    w = model.model.last_fc.weight.data.cpu().numpy().flatten()
    b = 0  
    

    x_val_emb = model.emb(x_val)
    

    x_val_adv = []
    for i,x0 in enumerate(x_val_emb):
        perturbation = 1.3 * (np.dot(w, x0) + b) / np.dot(w, w) * w
        x1 = x0 - perturbation
        x_val_adv.append(x1)
    x_val_adv = np.array(x_val_adv)
    

    total_loss_grad, _ = model.grad(x=x_val_adv, y=y_val)
    return total_loss_grad


def calc_robust_acc_nn1(model, x_val, y_val, type, stage):
    w = model.model.last_fc.weight.data.cpu().numpy().flatten()
    b = 0  
    
    x_val_emb = model.emb(x_val)

    if stage == 'pre':
        x_val_adv = []
        for i,x0 in enumerate(x_val_emb):
            perturbation = 1.3 * (np.dot(w, x0) + b) / np.dot(w, w) * w
            x1 = x0 - perturbation
            x_val_adv.append(x1)
        x_val_adv = np.array(x_val_adv)
        np.save('xadv_'+type+'.npy', x_val_adv)
    elif stage == 'post':
        x_val_adv = np.load('xadv_'+type+'.npy')
        if len(x_val_adv) != len(y_val):
            return 0.0, 0.0  

    # 4. 计算鲁棒性指标
    pred, pred_label = model.pred(x_val_adv)
    robust_acc = accuracy_score(pred_label, y_val)
    
    robust_loss = -np.sum(y_val * np.log(pred + 1e-10) + 
                         (1 - y_val) * np.log(1 - pred + 1e-10))
    
    return robust_acc, robust_loss
