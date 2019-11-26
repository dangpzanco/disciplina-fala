import pathlib
import soundfile as sf
import librosa
import numpy as np
import numpy.random as rnd

import pandas as pd

import tqdm

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



speech_folder = pathlib.Path('data/speech/')
noise_folder = pathlib.Path('data/noise/')
output_folder = pathlib.Path('data/speech+noise/')


speech_filenames = sorted(speech_folder.glob('*.WAV'))
noise_filenames = sorted(noise_folder.glob('*.flac'))


num_speech = 30
num_noise = 3

num_realizations = 5
num_snr = 8

snr_values = np.array([-20,-10,-5,0,5,10,20,np.inf])

speech_samplerate = 8000

# Metadata definitions
metadata = pd.DataFrame(index=[],columns=['filename', 'speech_name', 'noise_name', 'realization', 'SNR'])
filename = []
speech_name = []
noise_name = []
realization = []
SNR = []

for i in range(num_speech):
    signal, speech_samplerate = librosa.load(speech_filenames[i], sr=speech_samplerate)

    energy_signal = (signal ** 2).sum()
    num_samples = signal.size

    for j in range(num_noise):
        noise, _ = librosa.load(noise_filenames[j], sr=speech_samplerate)

        for k in range(num_realizations):

            ind = rnd.randint(0,noise.size-num_samples+1)
            real_noise = noise[ind:ind+num_samples]

            energy_noise = (real_noise ** 2).sum()
            real_snr = 10*np.log10(energy_signal / energy_noise)

            for l in range(num_snr):
                out_noise = fix_snr(real_noise, real_snr, snr_values[l])

                output_filename = f'{speech_filenames[i].stem}_{noise_filenames[j].stem.split()[0]}{k}_{snr_values[l]:.0f}.wav'
                output_path = output_folder / output_filename
                output_path.parent.mkdir(parents=True, exist_ok=True)

                print(output_filename)

                out_signal = signal + out_noise

                # Old clipping avoider
                # max_val = np.maximum(-out_signal.min(), out_signal.max())
                # out_signal *= (1/max_val) * (32767/32768)

                # temp_val = np.maximum(-out_signal.min(), out_signal.max())
                # if max_val < temp_val:
                #     max_val = temp_val
                # print(max_val)

                # Avoid clipping (magic number: 5, max_val is 4.97)
                out_signal /= 5

                sf.write(output_path, out_signal, speech_samplerate)

                # Set metadata
                filename.append(output_path.stem)
                speech_name.append(speech_filenames[i].stem)
                noise_name.append(noise_filenames[j].stem.split()[0])
                realization.append(k)
                SNR.append(snr_values[l])

metadata['filename'] = filename
metadata['speech_name'] = speech_name
metadata['noise_name'] = noise_name
metadata['realization'] = realization
metadata['SNR'] = SNR

print(metadata)

metadata.to_csv('data/metadata.csv', index=False)

