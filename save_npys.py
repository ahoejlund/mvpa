# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
from os.path import join
import numpy as np
import scipy.io as sio
import glob

proj_name = 'MINDLAB2012_16-EEG-Phoneme-Longitudinal-MMN'
os.environ['MINDLABPROJ']= proj_name
scratch_folder = join('/projects', proj_name, 'scratch')
file_folder = join(scratch_folder, 'for_python')
save_folder = join(scratch_folder, 'MVPA')
if not(os.path.exists(save_folder)):
    os.mkdir(save_folder)
    
files = glob.glob(join(file_folder, 'S*.mat'))

for fi, filename in enumerate(files):
    a = sio.loadmat(filename)['a']
    b = a.reshape((-1,a.shape[-1])).T
    np.save(join(save_folder, (filename[-17:-4] + '.npy')),b)
    
    
    
    
