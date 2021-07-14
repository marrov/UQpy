import numpy as np


def correlated_gaussian(samples, b_eff, d):

    qoi = list()
    for i in range(samples.shape[0]):
        qoi.append(b_eff * np.sqrt(d) - np.sum(samples[i, :]))
    return qoi