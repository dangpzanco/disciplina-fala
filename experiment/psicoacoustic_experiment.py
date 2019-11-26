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

metadata = pd.read_csv('metadata.csv')

technique_list = ['noisy', 'wiener', 'bayes', 'binary']
num_techniques = len(technique_list)

print(metadata)

snr_values = np.array([-20,-10,-5,0,5,10,20,np.inf])
num_snr = snr_values.size # 8
files_per_snr = 5

exp_folder = pathlib.Path('data/exp_data')
exp_folder.mkdir(parents=True, exist_ok=True)


speech_folder = pathlib.Path('data/speech/')
noisy_folder = pathlib.Path('data/speech+noise/')



rnd_ind = rnd.permutation(450)[:files_per_snr]
file_list = []
for i in range(num_snr):

    ind = metadata['SNR'] == snr_values[i]

    filenames = metadata['filename'].values[ind][rnd_ind]

    for item in filenames:
        file_list.append(item)

print(file_list)
num_files = len(file_list)


for i in range(num_files):
    for k in range(num_techniques):
        if technique_list[k] is 'noisy':
            processed_folder = pathlib.Path('data/speech+noise/')
        else:
            processed_folder = pathlib.Path('data/processed/') / technique_list[k]

        dest_folder = exp_folder / technique_list[k]
        dest_folder.mkdir(parents=True, exist_ok=True)

        src = processed_folder / f'{file_list[i]}.wav'
        dst = dest_folder / f'{file_list[i]}.wav'

        # shutil.copy2(str(src), str(dst))
        # print(src, dst)

exp_metadata = pd.DataFrame(index=[],columns=['filename', 'quality'])
exp_metadata['filename'] = sorted(file_list)
exp_metadata['quality'] = 0


print(exp_metadata)

exp_metadata.to_csv(exp_folder / 'exp_metadata.csv', index=False)


