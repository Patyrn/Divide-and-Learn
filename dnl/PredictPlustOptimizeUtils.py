import numpy as np



def compute_C_k(X, alphas, const, k=3, isSampling=True):
    """
    sum all alpha_j*a_j where j!=k into a constant
    :param isSampling:
    :param X:
    :param alphas:
    :param const:
    :param k:
    :return:
    """
    if isSampling:
        C_k = np.dot(np.delete(alphas, k).T, np.delete(X, k, 0)) + const
    else:
        C_k = np.dot(alphas.T, X) + const
    return C_k


def compute_F_k(X, alpha, C_k, k):
    """
    Compute F_k which was described in 2020 AAAI DP paper. for an alpha_k, F_K = X * alpha_k + C_K.
    Where C_k = sum all X*alpha_i for each alpha_(i!=k)
    :param X:
    :param alpha:
    :param C_k:
    :param k:
    :return:
    """
    F_k = alpha * X[k, :] + C_k
    return F_k


