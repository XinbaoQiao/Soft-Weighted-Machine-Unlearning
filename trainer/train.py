import copy
import os
import time
import numpy as np

import torch
import utils
from tqdm import tqdm

def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def get_optimizer_and_scheduler(model, args):
    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1
    )
    return optimizer, scheduler


def validate(val_loader, model, criterion, args, loader_name='val'):
    """在验证集上评估模型"""
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    rob_acc_meter = utils.AverageMeter()  # 添加鲁棒性准确率的计量器
    
    model.eval()
    device = args.device
    
    with torch.no_grad():
        for data in val_loader:
            if len(data) == 4:
                image, target, sensitive, _ = data
            else:
                image, target, sensitive = data
            
            image = image.to(device)
            target = target.to(device).float()
            sensitive = sensitive.to(device)
            
            output = model(image)
            
            if criterion is not None:
                loss = criterion(output, target)
                loss = loss.mean()
                losses.update(loss.item(), image.size(0))
            
            pred = (output > 0.5).float()
            acc = (pred == target).float().mean() * 100
            
            # 根据metric类型计算不同的指标
            if args.metric in ['dp', 'eop']:
                if args.metric == 'dp':
                    from fair_fn import loss_dp
                    metric_value = loss_dp(image.cpu().numpy(), sensitive.cpu().numpy(), output.detach().cpu().numpy())
                else:
                    from fair_fn import loss_ferm
                    metric_value = loss_ferm(model.log_loss, image.cpu().numpy(), target.cpu().numpy(), sensitive.cpu().numpy())
            else:  # robust
                if loader_name == 'val':  # 只在验证集上计算鲁棒性
                    if args.model_type == 'logreg':
                        from robust_fn import calc_robust_acc
                        rob_acc, metric_value = calc_robust_acc(model, image, target, 'val', 'post')
                    else:  # 'nn' or 'resnet'
                        from robust_fn_nn import calc_robust_acc_nn
                        # 确保数据是numpy格式
                        image_np = image.cpu().numpy() if isinstance(image, torch.Tensor) else image
                        target_np = target.cpu().numpy() if isinstance(target, torch.Tensor) else target
                        rob_acc, metric_value = calc_robust_acc_nn(model, image_np, target_np, 'val', 'post')
                else:  # 其他数据集不计算鲁棒性
                    rob_acc = None
                    metric_value = None
            
            top1.update(acc.item(), image.size(0))
            if args.metric == "robust" and rob_acc is not None:
                rob_acc_meter.update(rob_acc * 100, image.size(0))
    
    return {
        "utility": {"accuracy": top1.avg, "loss": losses.avg if criterion is not None else 0.0},
        "fairness" if args.metric in ['dp', 'eop'] else "robustness": {
            args.metric: metric_value,
            "accuracy": rob_acc_meter.avg if args.metric == "robust" and loader_name == 'val' else None
        }
    }


def train(train_loader, model, criterion, optimizer, epoch, args, forget_loader=None, val_loader=None):
    # 如果是逻辑回归且没有optimizer
    if optimizer is None and hasattr(model, 'fit_linear_layer'):
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        rob_acc_meter = utils.AverageMeter()  # 添加鲁棒性准确率的计量器
        
        all_data = []
        all_targets = []
        all_weights = []
        all_sensitive = []
        
        for image, target, sensitive, weight in train_loader:
            all_data.append(image.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_weights.append(weight.cpu().numpy())
            all_sensitive.append(sensitive.cpu().numpy())
            
        x = np.concatenate(all_data, axis=0)
        y = np.concatenate(all_targets, axis=0)
        weights = np.concatenate(all_weights, axis=0)
        sensitive = np.concatenate(all_sensitive, axis=0)
        
        # Normalize weights for soft unlearning
        if args.weight_scheme.startswith('soft'):
            model.fit_linear_layer(x, y, sample_weight=weights+1)
        if args.weight_scheme == 'hard':
            retain_mask = weights == 0
            x_retain = x[retain_mask]
            y_retain = y[retain_mask]
            # 使用retain数据训练模型
            model.fit_linear_layer(x_retain, y_retain, sample_weight=None)

        pred, pred_label = model.pred(x)
        acc = (pred_label == y).mean() * 100
        
        # 打印验证集的loss和准确率
        val_x = val_loader.dataset.tensors[0].cpu().numpy()
        val_y = val_loader.dataset.tensors[1].cpu().numpy()
        val_pred, val_pred_label = model.pred(val_x)
        val_acc = (val_pred_label == val_y).mean() * 100
        
        # 更新logistic回归的权重
        model.logistic.coef_ = model.last_fc_weight.reshape(1, -1)
        
        # 根据metric类型计算不同的指标
        if args.metric in ['dp', 'eop']:
            if args.metric == 'dp':
                from fair_fn import loss_dp
                metric_value = loss_dp(x, sensitive, pred)
                val_metric_value = loss_dp(val_x, val_loader.dataset.tensors[2].cpu().numpy(), val_pred)
            else:
                from fair_fn import loss_ferm
                metric_value = loss_ferm(model.log_loss, x, y, sensitive)
                val_metric_value = loss_ferm(model.log_loss, val_x, val_y, val_loader.dataset.tensors[2].cpu().numpy())
        else:  # robust
            # 只在验证集上计算鲁棒性
            val_metrics = validate(val_loader, model, criterion, args, 'val')
            val_metric_value = val_metrics["robustness"][args.metric]
            val_rob_acc = val_metrics["robustness"]["accuracy"]
            metric_value = val_metric_value  # 使用验证集的指标
            rob_acc = val_rob_acc  # 使用验证集的鲁棒准确率
        
        evals = {
            "train": {
                "utility": {"accuracy": acc.item(), "loss": model.log_loss(x, y)},
                "fairness" if args.metric in ['dp', 'eop'] else "robustness": {
                    args.metric: metric_value,
                    "accuracy": rob_acc if args.metric == "robust" else None
                }
            },
            "val": {
                "utility": {"accuracy": val_acc.item(), "loss": model.log_loss(val_x, val_y)},
                "fairness" if args.metric in ['dp', 'eop'] else "robustness": {
                    args.metric: val_metric_value,
                    "accuracy": val_rob_acc if args.metric == "robust" else None
                }
            }
        }
        
        if forget_loader is not None:
            forget_metrics = validate(forget_loader, model, criterion, args, 'forget')
            evals["forget"] = forget_metrics
            
        return evals
        
    else:
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        
        model.train()
        
        # 只有神经网络才需要冻结层
        if not hasattr(model, 'fit_linear_layer'):
            for name, param in model.named_parameters():
                if 'logistic' not in name:
                    param.requires_grad = False
        
        device = args.device
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
        
        for i, (image, target, sensitive, weight) in enumerate(pbar):
            image = image.to(device)
            target = target.to(device).float()
            sensitive = sensitive.to(device)
            weight = weight.to(device)

            # 获取模型输出并正确处理logits
            output = model(image)  # 这里的output已经是sigmoid后的结果
            logits = torch.log(torch.stack([1-output, output], dim=1) + 1e-10)  # 转换为log概率
            criterion.reduction = 'none'
            per_sample_loss = criterion(logits, target.long())
            weighted_loss = per_sample_loss * (1 + weight)
            final_loss = weighted_loss.mean()
            
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()
            
            pred = (output > 0.5).float()
            acc = (pred == target).float().mean() * 100
            
            # 根据metric类型计算不同的指标
            if args.metric in ['dp', 'eop']:
                if args.metric == 'dp':
                    from fair_fn import loss_dp
                    metric_value = loss_dp(image.cpu().numpy(), sensitive.cpu().numpy(), output.detach().cpu().numpy())
                else:
                    from fair_fn import loss_ferm
                    metric_value = loss_ferm(model.log_loss, image.cpu().numpy(), target.cpu().numpy(), sensitive.cpu().numpy())
            else:  # robust
                if args.model_type == 'logreg':
                    from robust_fn import calc_robust_acc
                    rob_acc, metric_value = calc_robust_acc(model, image, target, 'val', 'post')
                else:  # 'nn' or 'resnet'
                    from robust_fn_nn import calc_robust_acc_nn
                    # 确保数据是numpy格式
                    image_np = image.cpu().numpy() if isinstance(image, torch.Tensor) else image
                    target_np = target.cpu().numpy() if isinstance(target, torch.Tensor) else target
                    rob_acc, metric_value = calc_robust_acc_nn(model, image_np, target_np, 'val', 'post')
            
            losses.update(final_loss.item(), image.size(0))
            top1.update(acc.item(), image.size(0))
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{top1.avg:.2f}%',
                args.metric: f'{metric_value:.4f}'
            })
        
        pbar.close()
        
        val_metrics = validate(val_loader, model, criterion, args, 'val') if val_loader is not None else None
        
        evals = {
            "train": {
                "utility": {"accuracy": top1.avg, "loss": losses.avg},
                "fairness" if args.metric in ['dp', 'eop'] else "robustness": {
                    args.metric: metric_value
                }
            },
            "val": val_metrics if val_metrics is not None else None
        }
        
        if forget_loader is not None:
            forget_metrics = validate(forget_loader, model, criterion, args, 'forget')
            evals["forget"] = forget_metrics
        
        return evals
