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


win_length_seconds = 30e-3
hop_length_seconds = win_length_seconds / 2

sr = 11025
win_length_samples = int(win_length_seconds * sr)
hop_length_samples = int(hop_length_seconds * sr)

stft_params = dict(n_fft=2048, 
    hop_length=hop_length_samples, 
    win_length=win_length_samples)


metadata = pd.read_csv('metadata.csv')

technique_list = ['noisy', 'wiener', 'bayes', 'binary']
num_techniques = len(technique_list)

print(metadata)

# example file
snr_value = 0
noise_name = 'WASHING'
speech_name = 'F200501'
realization = 0
audio_name = f'{speech_name}_{noise_name}{realization}_{snr_value:.0f}.wav'

print(audio_name)

spec_list = []
for i in range(num_techniques):

    if technique_list[i] is 'noisy':
        processed_folder = pathlib.Path('data/speech+noise/')
    else:
        processed_folder = pathlib.Path('data/processed/') / technique_list[i]

    audio_path = processed_folder / audio_name

    signal, sr = librosa.load(audio_path, sr=None)
    signal *= 5

    signal_spec = np.abs(librosa.core.stft(signal, **stft_params))
    signal_spec = 20*np.log10(signal_spec + 1e-3)

    spec_list.append(signal_spec)


fig, ax = plt.subplots(4,1, figsize=(10,7), sharex=True)
ax = ax.ravel()

title_list = ['Sinal Contaminado', 'Máscara de Wiener', 'Estimador Bayesiano', 'Máscara Binária']

for i in range(num_techniques):
    librosa.display.specshow(spec_list[i], x_axis='time', y_axis='linear', sr=sr, 
        hop_length=stft_params['hop_length'], ax=ax[i], cmap='inferno')

    ax[i].set_title(title_list[i])
    ax[i].set_xlabel('')
    ax[i].set_ylabel('Frequência [Hz]')
    ax[i].ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset=True, useMathText=True)

ax[-1].set_xlabel('Tempo [s]')
plt.tight_layout(0.01)

# plt.savefig('images/spectrogram_example.pdf', format='pdf')
# plt.savefig('images/spectrogram_example.png', format='png')

plt.savefig('images/spectrogram_example_slides.pdf', format='pdf')
plt.savefig('images/spectrogram_example_slides.png', format='png')


# plt.show()
