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

import matplotlib.pyplot as plt

import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Influence Fairness')
    parser.add_argument('--dataset', type=str, default="adult", help="name of the dataset")
    parser.add_argument('--num_data', type=int, default=30162, help="number of data")
    parser.add_argument('--metric', type=str, default="eop", help="eop or dp")
    parser.add_argument('--seed', type=float, default=1, help="random seed")
    parser.add_argument('--save_model', type=str, default="n", help="y/n")
    parser.add_argument('--type', type=str, default="util", help="util/fair/")
    parser.add_argument('--strategy', type=str, default="dec", help="inc/dec/random")
    parser.add_argument('--points_to_delete', type=int, default=-1, help="points to delete (num or index)")
    parser.add_argument('--only_pre', type=str, default="n", help="y/n")
    parser.add_argument('--model_type', type=str, default="logreg", help="logreg/nn/resnet")

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

    val_rob_acc, val_rob_loss = calc_robust_acc(model, data.x_val, data.y_val, 'val', 'pre')
    ori_robust_loss_val = val_rob_loss

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
    robust_loss_total_grad = grad_robust(model, data.x_val, data.y_val)

    hess = model.hess(data.x_train)
    util_grad_hvp = model.get_inv_hvp(hess, util_loss_total_grad)
    fair_grad_hvp = model.get_inv_hvp(hess, fair_loss_total_grad)
    robust_grad_hvp = model.get_inv_hvp(hess, robust_loss_total_grad)

    # ()/(args.num_data*0.2): avg loss change
    util_pred_infl =  train_indiv_grad.dot(util_grad_hvp)
    fair_pred_infl =  train_indiv_grad.dot(fair_grad_hvp)
    robust_pred_infl = train_indiv_grad.dot(robust_grad_hvp)

    # Print influence values shapes and content for debugging
    print("Influence values shapes:")
    print(f"util_pred_infl shape: {util_pred_infl.shape}")
    print(f"fair_pred_infl shape: {fair_pred_infl.shape}")
    print(f"robust_pred_infl shape: {robust_pred_infl.shape}")

    _, pred_label_val = model.pred(data.x_val)
    _, pred_label_test = model.pred(data.x_test)

    val_res = val_evaluator(data.y_val, pred_label_val)
    test_res = test_evaluator(data.y_test, pred_label_test)

    np.save('traindata.npy', np.append(data.x_train, data.y_train.reshape((-1,1)), 1))
    return val_res, test_res, ori_util_loss_val, ori_fair_loss_val, ori_robust_loss_val, util_pred_infl, fair_pred_infl, robust_pred_infl


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

    val_evaluator, test_evaluator = Evaluator(data.s_val, "val"), Evaluator(data.s_test, "test")

    data.x_train = np.delete(data.x_train, args.points_to_delete, axis=0)
    data.y_train = np.delete(data.y_train, args.points_to_delete, axis=0)

    train_data_df = pd.DataFrame(data.x_train)
    train_data_df['label'] = data.y_train
    # train_data_df.to_csv('results/SEC-5-1-0-3/train_data_{}.csv'.format(args.points_to_delete), index=True)

    if args.points_to_delete > len(data.x_train):
        raise ValueError(f"points_to_delete ({args.points_to_delete}) must be less than the size of x_train ({len(data.x_train)})")

    """ vanilla training """
    model.fit(data.x_train, data.y_train)
    if args.metric == "eop":
        new_fair_loss_val = loss_ferm(model.log_loss, data.x_val, data.y_val, data.s_val)
    elif args.metric == "dp":
        pred_val, _ = model.pred(data.x_val)
        new_fair_loss_val = loss_dp(data.x_val, data.s_val, pred_val)
    else:
        raise ValueError
    new_util_loss_val = model.log_loss(data.x_val, data.y_val)

    val_rob_acc, val_rob_loss = calc_robust_acc(model, data.x_val, data.y_val, 'val', 'post')
    new_robust_loss_val = val_rob_loss

    _, pred_label_val = model.pred(data.x_val)
    _, pred_label_test = model.pred(data.x_test)

    val_res = val_evaluator(data.y_val, pred_label_val)
    test_res = test_evaluator(data.y_test, pred_label_test)


    np.save('traindata.npy', np.append(data.x_train, data.y_train.reshape((-1,1)), 1))
    return val_res, test_res, new_util_loss_val, new_fair_loss_val, new_robust_loss_val


if __name__ == "__main__":
    args = parse_args()

    os.makedirs('results/SEC-5-1-0-3', exist_ok=True)

    pre_val_res, pre_test_res, ori_util_loss_val, ori_fair_loss_val, ori_robust_loss_val, util_pred_infl, fair_pred_infl, robust_pred_infl = pre_main(args) #Run pre code

    # Calculate actual values
    util_diff = []
    fair_diff = []
    robust_diff = []
    args.points_to_delete=0
    for i in range(args.num_data):
        print("Index: ", args.points_to_delete)
        post_val_res, post_test_res, post_util_loss_val, post_fair_loss_val, post_robust_loss_val = post_main(args) #Run post code
        util_diff.append((post_util_loss_val - ori_util_loss_val))
        fair_diff.append(post_fair_loss_val - ori_fair_loss_val)
        robust_diff.append(post_robust_loss_val - ori_robust_loss_val)
        args.points_to_delete += 1

    # Only take the first args.num_data influence values
    util_pred_infl = util_pred_infl[:args.num_data]
    fair_pred_infl = fair_pred_infl[:args.num_data]
    robust_pred_infl = robust_pred_infl[:args.num_data]

    # Save actual values
    np.save('results/SEC-5-1-0-3/util_actual.npy', np.array(util_diff))
    np.save('results/SEC-5-1-0-3/fair_{}_actual.npy'.format(args.metric), np.array(fair_diff))
    np.save('results/SEC-5-1-0-3/robust_actual.npy', np.array(robust_diff))

    # Save influence values
    print("\nSaving influence values:")
    print(f"util_pred_infl shape: {util_pred_infl.shape}")
    print(f"fair_pred_infl shape: {fair_pred_infl.shape}")
    print(f"robust_pred_infl shape: {robust_pred_infl.shape}")
    
    np.save('results/SEC-5-1-0-3/util_infl.npy', util_pred_infl)
    np.save('results/SEC-5-1-0-3/fair_{}_infl.npy'.format(args.metric), fair_pred_infl)
    np.save('results/SEC-5-1-0-3/robust_infl.npy', robust_pred_infl)

    # Save to CSV file - actual values
    util_df = pd.DataFrame(util_diff, columns=['Actual Util Difference'])
    fair_df = pd.DataFrame(fair_diff, columns=['Actual Fair Difference'])
    robust_df = pd.DataFrame(robust_diff, columns=['Actual Robust Difference'])

    util_df.to_csv('results/SEC-5-1-0-3/util_actual.csv', index=False)
    fair_df.to_csv('results/SEC-5-1-0-3/fair_{}_actual.csv'.format(args.metric), index=False)
    robust_df.to_csv('results/SEC-5-1-0-3/robust_actual.csv', index=False)

    # Save to CSV file - influence values
    try:
        util_infl_df = pd.DataFrame(util_pred_infl, columns=['Influence Util Difference'])
        fair_infl_df = pd.DataFrame(fair_pred_infl, columns=['Influence Fair Difference'])
        robust_infl_df = pd.DataFrame(robust_pred_infl, columns=['Influence Robust Difference'])
        
        util_infl_df.to_csv('results/SEC-5-1-0-3/util_infl.csv', index=False)
        fair_infl_df.to_csv('results/SEC-5-1-0-3/fair_{}_infl.csv'.format(args.metric), index=False)
        robust_infl_df.to_csv('results/SEC-5-1-0-3/robust_infl.csv', index=False)
        print("Successfully saved influence values to CSV")
    except Exception as e:
        print("Error saving influence values to CSV:", e)