import pathlib
import soundfile as sf
import librosa
import numpy as np
import numpy.random as rnd

from tqdm import trange
import tqdm

import pandas as pd

import analysis_utils as utils


data_folder = pathlib.Path('../data')

# ['filename', 'speech_name', 'noise_name', 'realization', 'SNR']
metadata = pd.read_csv(data_folder / 'metadata.csv')
num_files = metadata.shape[0]

technique_list = ['noisy', 'wiener', 'bayes', 'binary']
num_techniques = len(technique_list)

# Metadata definitions
results_metadata = pd.DataFrame(index=[],
    columns=['filename', 'speech_name', 'noise_name', 'realization', 
    'SNR', 'technique', 'llr', 'sdr'])
filename = []
speech_name = []
noise_name = []
realization = []
SNR = []
technique = []
sdr_score = []


sub_num_files = 100
file_ind = rnd.permutation(num_files)[:sub_num_files]

for k in trange(num_techniques):
# for k in range(num_techniques):

    speech_folder = data_folder / 'speech'
    noisy_folder = data_folder / 'speech+noise'

    if technique_list[k] is 'noisy':
        processed_folder = data_folder / 'speech+noise'
    else:
        processed_folder = data_folder / 'processed' / technique_list[k]

    # for i in trange(num_files):
    for i in tqdm.tqdm(file_ind):
    # for i in range(num_files):
        
        # Get filename
        speech_filename = processed_folder / f"{metadata['speech_name'][i]}_SNOW0_inf.wav"
        processed_filename = processed_folder / f"{metadata['filename'][i]}.wav"

        scores = {}
        scores['sdr'] = utils.sdr(processed_filename, speech_filename)

        print(scores)

        exit()

        filename.append(metadata['filename'][i])
        speech_name.append(metadata['speech_name'][i])
        noise_name.append(metadata['noise_name'][i])
        realization.append(metadata['realization'][i])
        SNR.append(metadata['SNR'][i])
        technique.append(technique_list[k])
        sdr_score.append(scores['sdr'])

results_metadata['filename'] = filename
results_metadata['speech_name'] = speech_name
results_metadata['noise_name'] = noise_name
results_metadata['realization'] = realization
results_metadata['SNR'] = SNR
results_metadata['technique'] = technique
results_metadata['sdr'] = sdr_score

print(results_metadata)

results_metadata.to_csv('custom_results.csv', index=False)








                


