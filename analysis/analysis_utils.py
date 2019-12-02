import pathlib
import soundfile as sf
import librosa
import numpy as np
import numpy.random as rnd
import scipy.linalg as linalg

from tqdm import trange

import pandas as pd

# import pysiib

# win_length_seconds = 30e-3
# hop_length_seconds = win_length_seconds / 2

# sr = 8000
# win_length_samples = int(win_length_seconds * sr)
# hop_length_samples = int(hop_length_seconds * sr)

# stft_params = dict(n_fft=2048, 
#     hop_length=hop_length_samples, 
#     win_length=win_length_samples)

def open_audios(processed_filename, speech_filename):
    # Open files
    signal, sr = librosa.load(speech_filename, sr=None)
    processed_signal, _ = librosa.load(processed_filename, sr=None)

    # Fix the size mismatch
    clean_size = signal.size
    processed_size = processed_signal.size

    # Cut extra samples
    signal = signal[:processed_size]

    return processed_signal, signal, sr


# def siib(processed_filename, speech_filename):
#     processed_signal, signal, sr = open_audios(processed_filename, speech_filename)

#     # minimum_duration = 20.0
#     minimum_duration = 30.0
#     signal_duration = signal.size / sr
#     num_repeats = int(minimum_duration / signal_duration) + 1

#     processed_signal = np.hstack([processed_signal] * num_repeats)
#     signal = np.hstack([signal] * num_repeats)

#     siib_metric = pysiib.SIIB(signal, processed_signal, sr, gauss=True)

#     return siib_metric

def _snr(x, y):
    x_energy = (x ** 2).sum()
    y_energy = (y ** 2).sum()
    snr = 10*np.log10(x_energy / y_energy)

    return snr

def snr_gain(processed_filename, speech_filename, SNR):
    
    processed_signal, signal, sr = open_audios(processed_filename, speech_filename)

    # Compute SNR of the processed signal
    noise_est = processed_signal - signal

    snr_gain_metric = _snr(signal, noise_est) - SNR

    return snr_gain_metric




def llr(processed_filename, speech_filename):

    processed_signal, signal, sr = open_audios(processed_filename, speech_filename)

    # Constants
    lpc_order = 10
    alpha = 0.95
    frame_length = int(np.round(30e-3 * sr))
    hop_length = int(np.floor(frame_length / 4))

    # ----------------------------------------------------------
    # (1) Get the Frames for the test and reference speech. 
    #     Multiply by Hanning Window.
    # ----------------------------------------------------------

    processed_frames = librosa.util.frame(processed_signal.ravel(), 
        frame_length=frame_length, hop_length=hop_length)

    clean_frames = librosa.util.frame(signal.ravel(), 
        frame_length=frame_length, hop_length=hop_length)

    win = librosa.filters.get_window('hanning', frame_length)
    processed_frames *= win.reshape(-1,1)
    clean_frames *= win.reshape(-1,1)

    num_frames = clean_frames.shape[1]
    distortion = np.empty(num_frames)
    for k in range(num_frames):

        # ----------------------------------------------------------
        # (2) Get the autocorrelation lags and LPC parameters used
        #     to compute the LLR measure.
        # ----------------------------------------------------------

        clean_audio = clean_frames[:,k]
        processed_audio = processed_frames[:,k]


        lpc_clean = librosa.lpc(clean_audio, lpc_order)
        lpc_processed = librosa.lpc(processed_audio, lpc_order)

        R_clean = np.empty(lpc_order+1)
        for k in range(lpc_order+1):
            R_clean[k] = (clean_audio[:frame_length-k] * 
                          clean_audio[k:frame_length]).sum()
        R_clean = linalg.toeplitz(R_clean)

        # ----------------------------------------------------------
        # (3) Compute the LLR measure
        # ----------------------------------------------------------

        numerator   = lpc_processed.T @ R_clean @ lpc_processed
        denominator = lpc_clean.T @ R_clean @ lpc_clean

        distortion[k] = np.minimum(2, np.log(numerator / denominator))

    distortion_length = int(np.round(num_frames * alpha))
    llr = np.sort(distortion)[:distortion_length]
    llr_metric = llr.mean()

    return llr_metric



