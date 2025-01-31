import numpy as np

from model import IFBaseClass


def grad_ferm(grad_fn: IFBaseClass.grad, x: np.ndarray, y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Fair empirical risk minimization for binary sensitive attribute
    Exp(L|grp_0) - Exp(L|grp_1)
    """
    # grad_ferm: Computes the gradient for the fairness metric FERM (Equal Opportunity)
    # - Calculates the gradient difference for samples labeled as 1 in two groups
    # - Returns (average gradient of y=1 samples in group 0) - (average gradient of y=1 samples in group 1)

    N = x.shape[0]

    idx_grp_0_y_1 = [i for i in range(N) if s[i] == 0 and y[i] == 1]
    idx_grp_1_y_1 = [i for i in range(N) if s[i] == 1 and y[i] == 1]

    grad_grp_0_y_1, _ = grad_fn(x=x[idx_grp_0_y_1], y=y[idx_grp_0_y_1])
    grad_grp_1_y_1, _ = grad_fn(x=x[idx_grp_1_y_1], y=y[idx_grp_1_y_1])

    return (grad_grp_0_y_1 / len(idx_grp_0_y_1)) - (grad_grp_1_y_1 / len(idx_grp_1_y_1))


def loss_ferm(loss_fn: IFBaseClass.log_loss, x: np.ndarray, y: np.ndarray, s: np.ndarray) -> float:

    # loss_ferm: Computes the FERM loss value
    # - Calculates the loss difference for samples labeled as 1 in two groups
    # - Returns (average loss of y=1 samples in group 0) - (average loss of y=1 samples in group 1)

    N = x.shape[0]

    idx_grp_0_y_1 = [i for i in range(N) if s[i] == 0 and y[i] == 1]
    idx_grp_1_y_1 = [i for i in range(N) if s[i] == 1 and y[i] == 1]

    loss_grp_0_y_1 = loss_fn(x[idx_grp_0_y_1], y[idx_grp_0_y_1])
    loss_grp_1_y_1 = loss_fn(x[idx_grp_1_y_1], y[idx_grp_1_y_1])
    
    eop_diff = (loss_grp_0_y_1 / len(idx_grp_0_y_1)) - (loss_grp_1_y_1 / len(idx_grp_1_y_1))
    
    # if eop_diff > 0:
    #     print('Group 1 (s=1) is privileged, Group 0 (s=0) is unprivileged')
    # else:
    #     print('Group 0 (s=0) is privileged, Group 1 (s=1) is unprivileged')
    
    # print('EOP:', eop_diff)
    return eop_diff


def grad_dp(grad_fn: IFBaseClass.grad_pred, x: np.ndarray, s: np.ndarray) -> np.ndarray:
    # grad_dp: Computes the gradient for Demographic Parity
    # - Calculates the gradient difference for predictions between two groups
    # - Returns (average gradient of group 1) - (average gradient of group 0)

    """ Demographic parity """

    N = x.shape[0]

    idx_grp_0 = [i for i in range(N) if s[i] == 0]
    idx_grp_1 = [i for i in range(N) if s[i] == 1]

    grad_grp_0, _ = grad_fn(x=x[idx_grp_0])
    grad_grp_1, _ = grad_fn(x=x[idx_grp_1])

    return (grad_grp_1 / len(idx_grp_1)) - (grad_grp_0 / len(idx_grp_0))


def loss_dp(x: np.ndarray, s: np.ndarray, pred: np.ndarray) -> float:
    # loss_dp: Computes the DP loss value
    # - Calculates the difference in predicted probabilities between two groups
    # - Returns (average predicted value of group 1) - (average predicted value of group 0)

    N = x.shape[0]

    idx_grp_0 = [i for i in range(N) if s[i] == 0]
    idx_grp_1 = [i for i in range(N) if s[i] == 1]

    pred_grp_0 = np.sum(pred[idx_grp_0])
    pred_grp_1 = np.sum(pred[idx_grp_1])
    
    dp_diff = (pred_grp_1 / len(idx_grp_1)) - (pred_grp_0 / len(idx_grp_0))
    
    # if dp_diff > 0:
    #     print('Group 1 (s=1) is privileged, Group 0 (s=0) is unprivileged')
    # else:
    #     print('Group 0 (s=0) is privileged, Group 1 (s=1) is unprivileged')
    
    # print('DP:', dp_diff)
    return dp_diff
