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



results = pd.read_csv('../analysis/compare_metadata.csv')

subject_names = ['bruno','celso','felipe']
num_subjects = len(subject_names)

results['mos'] = np.median(results[subject_names],axis=1)

metric_names = ['pesq', 'srmr', 'stoi', 'csii', 'snrg', 'llr']
num_metrics = len(metric_names)

# Normalize LLR
results['llr'] = (2 - results['llr'])/2

metric_values = results[metric_names].values
mos_values = results['mos'].values


metric_conf = np.empty(num_metrics)
for i in range(num_metrics):
    x, y = (mos_values, metric_values[:,i])
    metric_conf[i], p = stats.pearsonr(x, y)

print(metric_names)
print('Pearson per metric:')
# print(f'{metric_conf.mean():.3f} +- {metric_conf.std():.3f}')
print(metric_conf)


# mos_values = results[subject_names].values
# metric_conf = np.empty([num_metrics, num_subjects])
# for i in range(num_metrics):
#     for j in range(num_subjects):
#         x, y = (mos_values[:,j], metric_values[:,i])
#         metric_conf[i,j], p = stats.pearsonr(x, y)

# print('Pearson per metric:')
# # print(f'{metric_conf.mean():.3f} +- {metric_conf.std():.3f}')
# print('mean:', metric_conf.mean(axis=-1), '\nstd: ', metric_conf.std(axis=-1))
# print(metric_conf)


# print(ind.size)
# print(rep_files)
# print(rep_files.size)


# plt.hist(mos_values)
# plt.show()

icc = 0.8703
print(f'ICC = {icc}')
