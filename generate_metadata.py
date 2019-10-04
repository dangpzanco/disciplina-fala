import pathlib
import soundfile as sf
import librosa
import numpy as np
import numpy.random as rnd

import pandas as pd

rnd.seed(0)

def snr(x, y):
    x_energy = (x ** 2).sum()
    y_energy = (y ** 2).sum()
    snr = 10*np.log10(x_energy / y_energy)

    return snr


def fix_snr(a, in_snr, out_snr):

    if out_snr == np.inf:
        gain = 0
    else:
        gain = 10 ** (0.05 * (in_snr - out_snr))
    
    return gain * a



noisy_folder = pathlib.Path('../data/speech+noise/')
noisy_filenames = sorted(noisy_folder.glob('*.wav'))


# output_filename = f'{speech_filenames[i].stem}_{noise_filenames[j].stem.split()[0]}{k}_{snr_values[l]:.0f}.wav'
metadata = pd.DataFrame(index=[],columns=['filename', 'speech_name', 'noise_name', 'realization', 'SNR'])
num_files = len(noisy_filenames)

# Column lists
filename = [None] * num_files
speech_name = [None] * num_files
noise_name = [None] * num_files
realization = [None] * num_files
SNR = [None] * num_files

for i in range(num_files):
    
    splitted_name = noisy_filenames[i].stem.split('_')

    print(splitted_name)

    filename[i] = noisy_filenames[i].stem
    speech_name[i] = splitted_name[0]
    noise_name[i] = splitted_name[1][:-1]
    realization[i] = splitted_name[1][-1]
    SNR[i] = float(splitted_name[-1])


print(type(SNR[-1]))

metadata['filename'] = filename
metadata['speech_name'] = speech_name
metadata['noise_name'] = noise_name
metadata['realization'] = realization
metadata['SNR'] = SNR


# metadata = metadata.sort_values(['speech_name', 'noise_name', 'realization', 'SNR'], ascending=True)
print(metadata)

metadata.to_csv('metadata.csv', index=False)


