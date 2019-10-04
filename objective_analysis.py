import pathlib
import soundfile as sf
import librosa
import numpy as np
import numpy.random as rnd

from tqdm import trange

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



win_length_seconds = 30e-3
hop_length_seconds = win_length_seconds / 2

sr = 11025
win_length_samples = int(win_length_seconds * sr)
hop_length_samples = int(hop_length_seconds * sr)

stft_params = dict(n_fft=2048, 
    hop_length=hop_length_samples, 
    win_length=win_length_samples)

# ['filename', 'speech_name', 'noise_name', 'realization', 'SNR']
metadata = pd.read_csv('metadata.csv')
num_files = metadata.shape[0]


technique_list = ['noisy', 'wiener', 'bayes', 'binary']
num_techniques = len(technique_list)


results_dict = dict(noisy=np.empty(num_files), 
    wiener=np.empty(num_files), 
    bayes=np.empty(num_files), 
    binary=np.empty(num_files))

for k in trange(num_techniques):

    speech_folder = pathlib.Path('../data/speech/')
    noisy_folder = pathlib.Path('../data/speech+noise/')

    if technique_list[k] is 'noisy':
        processed_folder = pathlib.Path('../data/speech+noise/')
    else:
        processed_folder = pathlib.Path('../data/processed/') / technique_list[k]

    for i in trange(num_files):
        
        # Get filename
        speech_filename = speech_folder / f"{metadata['speech_name'][i]}.WAV"
        processed_filename = processed_folder / f"{metadata['filename'][i]}.wav"

        # Open files
        signal, sr = librosa.load(speech_filename, sr=None)
        processed_signal, _ = librosa.load(processed_filename, sr=None)

        # Magic number (fix amplitude mismatch)
        signal /= 5

        # Fix the size mismatch
        clean_size = signal.size
        processed_size = processed_signal.size

        # Cut extra samples
        signal = signal[:processed_size]

        # Magnitude Spectrum
        signal_spec = np.abs(librosa.core.stft(signal, **stft_params))
        processed_spec = np.abs(librosa.core.stft(processed_signal, **stft_params))

        # Mean Squared Error of the magnitude spectrum for each frame
        mse_processed = ((signal_spec - processed_spec) ** 2).mean(axis=0)

        results_dict[technique_list[k]][i] = mse_processed.mean()


outfile = '../data/objective_analysis/objective_results.npz'

np.savez(outfile, **results_dict)






                


