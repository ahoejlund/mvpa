# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:39:51 2017

@author: hojlund
"""

import os
from os.path import join
import numpy as np
import scipy.io as sio
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib
import matplotlib.pyplot as plt

proj_name = 'MINDLAB2012_16-EEG-Phoneme-Longitudinal-MMN'
os.environ['MINDLABPROJ']= proj_name
#scratch_folder = join('/projects', proj_name, 'scratch')
scratch_folder = join('/Volumes/projects', proj_name, 'scratch') # running locally
file_folder = join(scratch_folder, 'for_python')
save_folder = join(scratch_folder, 'MVPA')

subjs = np.arange(1,21,1)

# X and y matrix construction
# OVER TIME
T0_dev = np.load(join(save_folder,'S1_C1_T0_dev3.npy'))
T3_dev = np.load(join(save_folder,'S1_C1_T3_dev3.npy'))

T0_std = np.load(join(save_folder,'S1_C1_T0_std3.npy'))
T3_std = np.load(join(save_folder,'S1_C1_T3_std3.npy'))

T0 = T0_dev
T3 = T3_dev

X = np.concatenate([T0, T3])

y = np.concatenate([np.zeros(len(T0)), np.ones(len(T3))])
#y = np.concatenate([np.zeros(len(T0)), np.ones(len(T1)), np.ones(len(T2))*2, 
#                    np.ones(len(T3))*3])

# MMN
dev = np.load(join(save_folder,'S1_C1_T3_dev3.npy'))
std = np.load(join(save_folder,'S1_C1_T3_std3.npy'))

X = np.concatenate([dev, std])

y = np.concatenate([np.zeros(len(dev)), np.ones(len(std))])


# Cross-validation setup
scaler = []
model = []
score = []
cv = StratifiedKFold(n_splits=10, shuffle=True)
for train, test in cv.split(X, y):
    scl = StandardScaler()
    X_train = scl.fit_transform(X[train])
    X_test = scl.transform(X[test])
    mdl = LogisticRegression(C=1)
    mdl.fit(X_train, y[train])
    scr = roc_auc_score(mdl.predict(X_test), y[test])
    scaler.append(scl)
    model.append(mdl)
    score.append(scr)
    
score = np.asarray(score)
print(score.mean(), score.std(), score)

# for saving
joblib.dump(models, 'FILENAME')
# joblib.load()

# for plotting
coefs = np.asarray([m.coef_ for m in model]).squeeze()
times = np.arange(-100, 501, 4)
plt.plot(times, coefs.mean(axis=0).reshape([32, -1]).T)

