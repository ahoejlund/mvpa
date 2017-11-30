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

stims = ('std', 'dev', 'mmn')    # test std and dev over time and mmn as diff betw dev and std

subjs = np.arange(1,21,1) # 20 subjs
conds = np.arange(1,3,1) # 2 conds (arab vs. dari)
arabs = np.arange(1,9,1) # arab subjs range from 1 to 8
daris = np.arange(9,21,1) # dari subjs range from 9 to 20
times = (0,3) # Ts to be tested, first subtracted from last (e.g. (0,3) = T3 minus T0)
dev_level = 3 # which deviant-level to look at, ranging from 1-3 (if interested in std0, stims should be specified as only 'std')

for s in subjs:
    for c in conds:
        for n, stim in enumerate(stims):   
            # X and y matrix construction
            if con == 'mmn':
                # STD vs DEV
                T0 = np.load(join(save_folder, 
                                  'S%s_C%s_T%s_std3.npy' %(s, c, times[1]))) # 
                T3 = np.load(join(save_folder, 
                                  'S%s_C%s_T%s_dev3.npy' %(s, c, times[1])))
            else:
                # OVER TIME
                T0 = np.load(join(save_folder, 'S%s_C%s_T%s_%s%s.npy' 
                                  %(s, c, times[0], stim, dev_level)))
                T3 = np.load(join(save_folder, 'S%s_C%s_T%s_%s%s.npy' 
                                  %(s, c, times[3], stim, dev_level)))
        X = np.concatenate([T0, T3])        
        y = np.concatenate([np.zeros(len(T0)), np.ones(len(T3))])
        
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

