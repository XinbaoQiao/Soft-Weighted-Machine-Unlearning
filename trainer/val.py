import utils

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler  # for feature scaling
from sklearn.model_selection import train_test_split  # for train/test split


class LogisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted


def RecalibrationBaseline(val_loader, cal_loader, model, criterion, args, calibrate="platt", log_model=None):
    cls_ignore = args.class_to_replace

    num_classes = 10

    criterion = nn.NLLLoss()

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    if args.device is None:
        if torch.cuda.is_available():
            torch.cuda.set_device(int(args.gpu))
            device = torch.device(f"cuda:{int(args.gpu)}")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model.eval()

    if calibrate == "platt":
        # Initialise a binary classifier for each class
        if log_model is None:
            log_model = LogisticRegression(1)

            log_model = log_model.to(device)

            log_learning_rate = args.unlearn_lr
            log_criterion = nn.BCELoss()
            log_optimizer = torch.optim.SGD(log_model.parameters(), lr=log_learning_rate)

            for epoch in range(args.unlearn_epochs):
                for i, (image, target) in enumerate(cal_loader):
                    image = image.to(device)
                    target = target.to(device)

                    with torch.no_grad():

                        output = model(image)

                        # check if output is a tuple
                        if isinstance(output, tuple):
                            output = output[0]

                        output = output.float()

                        # Set the output of the class to ignore to 0
                        output[:, cls_ignore] = 0

                        # Normalize the output
                        output = output / output.sum(dim=1, keepdim=True)

                        output[:, cls_ignore] = float('-inf')                          
                        
                        output_soft = nn.functional.softmax(output_soft, dim=1)

                        # get the predicted class and its probability
                        pred_prob, pred_class = torch.max(output_soft, 1)

                        # Reshape pred_prob to be a column vector
                        pred_prob = pred_prob.view(-1, 1)

                    # train the binary classifier
                    for i in range(num_classes):
                        if i == cls_ignore:
                            continue
                        # get the binary target
                        binary_target = torch.where(pred_class == i, 1, 0)
                        log_optimizer.zero_grad()
                        y_predicted = log_model(pred_prob)
                        # Expand the target to be the same shape as the output
                        binary_target = binary_target.view(-1, 1).float()
                        y_predicted = y_predicted.float()
                        loss = log_criterion(y_predicted, binary_target)
                        loss.backward(retain_graph=True)
                        log_optimizer.step()

                    print(loss.float().item())

    # switch to evaluate mode
    model.eval()
    for i, (image, target) in enumerate(val_loader):
        image = image.to(device)
        target = target.to(device)

        # compute output
        with torch.no_grad():
            output = model(image)
            # check if output is a tuple
            if isinstance(output, tuple):
                output = output[0]

        # print(output.shape, output)
        # print(target.shape, target)
        output = output.float()

        # Set the output of the class to ignore to 0
        output[:, cls_ignore] = 0

        # Normalize the output
        output = output / output.sum(dim=1, keepdim=True)

        if calibrate == "platt":
            # Calibrate the output by putting them all through the logistic regression
            new_output = torch.zeros_like(output)
            for i in range(output.shape[1]):
                if i == cls_ignore:
                    continue # Dont calibrate and leave at 0
                i_prob = output[:, i].view(-1, 1)
                new_output[:, i] = log_model(i_prob).view(-1)
            output = new_output

        # output[:, cls_ignore] = float('-inf')

        loss = criterion(output, target)
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = utils.accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Accuracy {top1.val:.3f} ({top1.avg:.3f})".format(
                    i, len(val_loader), loss=losses, top1=top1
                )
            )

    print("valid_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg, losses.avg, log_model


def IgnoreBaseline(val_loader, model, criterion, args):
    cls_ignore = args.class_to_replace

    criterion = nn.CrossEntropyLoss()

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    if args.device is None:
        if torch.cuda.is_available():
            torch.cuda.set_device(int(args.gpu))
            device = torch.device(f"cuda:{int(args.gpu)}")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # switch to evaluate mode
    model.eval()

    for i, (image, target) in enumerate(val_loader):
        image = image.to(device)
        target = target.to(device)

        # compute output
        with torch.no_grad():
            output = model(image)
            # check if output is a tuple
            if isinstance(output, tuple):
                output = output[0]

        # print(output.shape, output)
        # print(target.shape, target)
        output = output.float()

        # output = nn.functional.softmax(output, dim=1)

        # Set the output of the class to ignore to 0
        output[:, cls_ignore] = 0

        # output = nn.functional.softmax(output, dim=1)

        # Set the output of the class to ignore to 0
        # output[:, cls_ignore] = 0

        # Normalize the output
        output = output / output.sum(dim=1, keepdim=True)

        # output[:, cls_ignore] = float('-inf')

        loss = criterion(output, target)

        loss = loss.float()

        # measure accuracy and record loss
        prec1 = utils.accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Accuracy {top1.val:.3f} ({top1.avg:.3f})".format(
                    i, len(val_loader), loss=losses, top1=top1
                )
            )

    print("valid_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args):
    """
    Run evaluation
    """
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    if args.device is None:
        if torch.cuda.is_available():
            torch.cuda.set_device(int(args.gpu))
            device = torch.device(f"cuda:{int(args.gpu)}")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # switch to evaluate mode
    model.eval()
    for i, (image, target) in enumerate(val_loader):
        image = image.to(device)
        target = target.to(device)

        # compute output
        with torch.no_grad():
            output = model(image)
            # check if output is a tuple
            if isinstance(output, tuple):
                output = output[0]
            loss = criterion(output, target)

        # print(output.shape, output)
        # print(target.shape, target)
        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = utils.accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Accuracy {top1.val:.3f} ({top1.avg:.3f})".format(
                    i, len(val_loader), loss=losses, top1=top1
                )
            )

    print("valid_accuracy {top1.avg:.3f}".format(top1=top1))
    model.train()

    return top1.avg, losses.avg
