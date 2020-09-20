import os
import sys
import tarfile
import time
import pyprind
import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import explained_variance_score



X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
## Kernel PCA
# Kernel PCAで変換
# X_kpca = KernelPCA.fit_transform(X)
# print("Explained var ratio :", np.sum(np.var(X_kpca, axis=0) / np.sum(np.var(X, axis=0)))) #不適切
# print("Explained var score :", explained_variance_score(X, kpca.inverse_transform(X_kpca)))
# ----------------------------hyperopt----------------------------
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def acc_model(params):
    clf = KernelPCA(**params,fit_inverse_transform=True)
    return -1* explained_variance_score(clf, clf.inverse_transform(clf.fit_transform(X))).mean()

param_space = {
    'gamma': hp.choice('gamma', range(1,20)),}

best = 0
def f(params):
    global best
    acc = acc_model(params)
    if acc > best:
        best = acc
    print ('new best:', best, params)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(acc_model, param_space, algo=tpe.suggest, max_evals=100, trials=trials)
print ('best:')
print (best)
