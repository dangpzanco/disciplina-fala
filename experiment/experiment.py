import pathlib
import soundfile as sf
import librosa
import librosa.display
import numpy as np
import numpy.random as rnd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import trange

import shutil


rnd.seed(0)

metadata = pd.read_csv('../analysis/speechmetrics_results.csv')


metadata = metadata.drop(['pesq', 'stoi', 'srmr'], axis=1)
print(metadata.columns)
print(metadata)



technique_list = ['noisy', 'wiener', 'bayes', 'binary']
num_techniques = len(technique_list)

# exit()

snr_values = np.array([-20,-10,0,10,20])
num_snr = snr_values.size # 5
files_per_snr = 4
num_rep = 1

exp_folder = pathlib.Path('exp_data')
exp_folder.mkdir(parents=True, exist_ok=True)


speech_folder = pathlib.Path('../data/speech/')
noisy_folder = pathlib.Path('../data/speech+noise/')



# id_list = []
# filename = []
# speech_name = []
# noise_name = []
# realization = []
# SNR = []
# technique = []
# rep_list = []

# Metadata definitions
num_files = num_snr * (files_per_snr + num_rep) * num_techniques
exp_metadata = pd.DataFrame(index=np.arange(num_files),
    columns=['id', 'filename', 'speech_name', 'noise_name', 'realization', 
    'SNR', 'technique', 'rep'])

file_list = []
exp_index = 0
for k, tech in enumerate(technique_list):
    for i in range(num_snr):

        ind = metadata['SNR'] == snr_values[i]
        rnd_ind = rnd.permutation(ind.sum())[:files_per_snr]

        filenames = metadata['filename'].values[ind][rnd_ind]


        for j, item in enumerate(filenames):
            exp_metadata['filename'][exp_index] = item
            exp_metadata['speech_name'][exp_index] = item.split('_')[0]
            exp_metadata['noise_name'][exp_index] = item.split('_')[1][:-1]
            exp_metadata['realization'][exp_index] = item.split('_')[1][-1]
            exp_metadata['SNR'][exp_index] = float(item.split('_')[-1])
            exp_metadata['technique'][exp_index] = tech
            exp_metadata['rep'][exp_index] = False

            exp_index +=1
            file_list.append(item)

        exp_metadata['rep'][exp_index-1] = True    

        # Repeated audios
        for j in range(num_rep):
            exp_metadata['filename'][exp_index] = item
            exp_metadata['speech_name'][exp_index] = item.split('_')[0]
            exp_metadata['noise_name'][exp_index] = item.split('_')[1][:-1]
            exp_metadata['realization'][exp_index] = item.split('_')[1][-1]
            exp_metadata['SNR'][exp_index] = float(item.split('_')[-1])
            exp_metadata['technique'][exp_index] = tech
            exp_metadata['rep'][exp_index] = True
            exp_index +=1
            file_list.append(item)

num_files = len(file_list)
print(num_files, file_list)

exp_metadata = exp_metadata.sample(frac=1).reset_index(drop=True)
exp_metadata['id'] = np.arange(num_files)
print(exp_metadata)

exp_metadata.to_csv('exp_metadata.csv', index=False)


subject_metadata = exp_metadata.drop([
    'filename', 'speech_name', 'noise_name', 
    'realization', 'SNR', 'technique', 'rep'], axis=1)

subject_metadata['quality'] = -1

print(subject_metadata)

# exit()

for i in range(num_files):
    tech = exp_metadata['technique'][i]

    if tech is 'noisy':
        processed_folder = pathlib.Path('../data/speech+noise/')
    else:
        processed_folder = pathlib.Path('../data/processed/') / tech

    in_filename = exp_metadata['filename'][i]
    out_filename = exp_metadata['id'][i]

    # print(in_filename, out_filename)

    src = processed_folder / f'{in_filename}.wav'
    dst = exp_folder / f'{out_filename:03}.wav'

    shutil.copy2(str(src), str(dst))
    print(src, dst)



# exp_metadata = pd.DataFrame(index=[],columns=['filename', 'quality'])
# exp_metadata['filename'] = sorted(file_list)
# exp_metadata['quality'] = 0
# print(exp_metadata)

subject_metadata.to_csv(exp_folder / 'subject_metadata.csv', index=False)


