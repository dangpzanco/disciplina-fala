import pathlib
import soundfile as sf
import librosa
import numpy as np
import numpy.random as rnd

from tqdm import trange

import pandas as pd

import speechmetrics


data_folder = pathlib.Path('../data')

win_length_seconds = 30e-3
hop_length_seconds = win_length_seconds / 2

sr = 8000
win_length_samples = int(win_length_seconds * sr)
hop_length_samples = int(hop_length_seconds * sr)

stft_params = dict(n_fft=2048, 
    hop_length=hop_length_samples, 
    win_length=win_length_samples)

# ['filename', 'speech_name', 'noise_name', 'realization', 'SNR']
metadata = pd.read_csv(data_folder / 'metadata.csv')
num_files = metadata.shape[0]


technique_list = ['noisy', 'wiener', 'bayes', 'binary']
num_techniques = len(technique_list)

# Metadata definitions
results_metadata = pd.DataFrame(index=[],
    columns=['filename', 'speech_name', 'noise_name', 'realization', 
    'SNR', 'technique', 'pesq', 'stoi', 'srmr'])
filename = []
speech_name = []
noise_name = []
realization = []
SNR = []
technique = []
pesq_score = []
stoi_score = []
srmr_score = []

# mixed case, still works
metric_function = speechmetrics.load(['pesq', 'stoi', 'srmr'], window=None, verbose=False)
for i in range(len(metric_function.metrics)):
    metric_function.metrics[i].fixed_rate = 8000

for k in trange(num_techniques):
# for k in range(num_techniques):

    speech_folder = data_folder / 'speech'
    noisy_folder = data_folder / 'speech+noise'

    if technique_list[k] is 'noisy':
        processed_folder = data_folder / 'speech+noise'
    else:
        processed_folder = data_folder / 'processed' / technique_list[k]

    for i in trange(num_files):
    # for i in range(num_files):
        
        # Get filename
        speech_filename = processed_folder / f"{metadata['speech_name'][i]}_SNOW0_inf.wav"
        processed_filename = processed_folder / f"{metadata['filename'][i]}.wav"

        scores = metric_function(str(processed_filename), str(speech_filename))

        filename.append(metadata['filename'][i])
        speech_name.append(metadata['speech_name'][i])
        noise_name.append(metadata['noise_name'][i])
        realization.append(metadata['realization'][i])
        SNR.append(metadata['SNR'][i])
        technique.append(technique_list[k])
        pesq_score.append(scores['pesq'])
        stoi_score.append(scores['stoi'])
        srmr_score.append(scores['srmr'])

results_metadata['filename'] = filename
results_metadata['speech_name'] = speech_name
results_metadata['noise_name'] = noise_name
results_metadata['realization'] = realization
results_metadata['SNR'] = SNR
results_metadata['technique'] = technique
results_metadata['pesq'] = pesq_score
results_metadata['stoi'] = stoi_score
results_metadata['srmr'] = srmr_score

print(results_metadata)

results_metadata.to_csv('speechmetrics_results.csv', index=False)








                


