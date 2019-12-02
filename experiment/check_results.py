import pathlib
from tqdm import trange

import soundfile as sf
import librosa
import librosa.display

import numpy as np
import numpy.random as rnd
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns



results = pd.read_csv('results/results.csv')
metadata = pd.read_csv('exp_metadata.csv')



subjects = results.columns[1:].values
mos_values = results[subjects].values
print(mos_values.shape)


ind = np.where(metadata['rep'])[0]
rep_files = np.unique(metadata['filename'][ind].values)

num_reps = 2
num_subjects = 3
intra_mos = np.empty([num_reps,rep_files.size, num_subjects])
for i, filename in enumerate(rep_files):
    temp_ind = np.where(metadata['filename'] == filename)[0]
    intra_mos[:,i,:] = mos_values[temp_ind,]

print(intra_mos.shape)

intra_conf = np.empty(num_subjects)
for i in range(num_subjects):
    x, y = (intra_mos[0,:,i], intra_mos[1,:,i])
    intra_conf[i], p = stats.pearsonr(x, y)

print('Intra-Avaliador (Pearson):')
print(f'{intra_conf.mean():.3f} +- {intra_conf.std():.3f}')
print(intra_conf)


# print(ind.size)
# print(rep_files)
# print(rep_files.size)


# plt.hist(mos_values)
# plt.show()

icc = 0.8703
print(f'ICC = {icc}')
