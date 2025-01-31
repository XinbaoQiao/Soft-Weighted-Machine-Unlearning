import numpy as np
from sklearn.metrics import accuracy_score
import random

# def grad_fn(clf, x, y, sample_weight=None, l2_reg=False):
#     """
#     Compute the gradients: grad_wo_reg = (pred - y) * x
#     """
#     if sample_weight is None:
#         sample_weight = np.ones(x.shape[0])
#     sample_weight = np.array(sample_weight)

#     pred = clf.predict_proba(x)[:, 1]

#     indiv_grad = x * (pred - y).reshape(-1, 1)
#     weighted_indiv_grad = indiv_grad * sample_weight.reshape(-1, 1)
#     if clf.fit_intercept:
#         weighted_indiv_grad = np.concatenate([weighted_indiv_grad, (pred - y).reshape(-1, 1)], axis=1)

#     total_grad = np.sum(weighted_indiv_grad, axis=0)

#     return total_grad, weighted_indiv_grad

def grad_robust(model, x_val, y_val):

    clf = model.model
    grad_fn = model.grad

    w = clf.coef_[0]
    b = clf.intercept_
    x_val_adv = []
    for i,x0 in enumerate(x_val):
        perturbation = 1.3 * (np.dot(w, x0) + b) / np.dot(w, w) * w 
        x1 = x0 - perturbation
        x_val_adv.append(x1)
    x_val_adv = np.array(x_val_adv)

    total_loss_grad, t = grad_fn(x=x_val_adv, y=y_val)
    return total_loss_grad


def calc_robust_acc(model, x_val, y_val, type, stage):

    clf = model.model

    if stage == 'pre':
        w = clf.coef_[0]
        b = clf.intercept_
        x_val_adv = []
        for i,x0 in enumerate(x_val):
            perturbation = 1.3 * (np.dot(w, x0) + b) / np.dot(w, w) * w
            x1 = x0 - perturbation
            x_val_adv.append(x1)
        x_val_adv = np.array(x_val_adv)

        np.save('xadv_'+type+'.npy', x_val_adv)

    elif stage == 'post':
        x_val_adv = np.load('xadv_'+type+'.npy')

    y_val_adv = clf.predict(x_val_adv)
    robust_acc = accuracy_score(y_val_adv, y_val)
    
    y_val_adv_prob = clf.predict_proba(x_val_adv)
    robust_loss = -np.sum(y_val * np.log(y_val_adv_prob[:, 1] + 1e-10) + 
                         (1 - y_val) * np.log(y_val_adv_prob[:, 0] + 1e-10))
    
    return robust_acc, robust_loss
