import pathlib
import soundfile as sf
import librosa
import numpy as np
import numpy.random as rnd
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import trange



metadata = pd.read_csv('metadata.csv')


results = np.load('../data/objective_analysis/objective_results.npz')
methods = dict(results).keys()
results = pd.DataFrame(dict(results))

# print(results)
print(metadata)

snr_values = np.array([-20,-10,-5,0,5,10,20,np.inf])




fig, ax = plt.subplots(2,4,figsize=(10,7))
ax = ax.ravel()

for i in range(snr_values.size):

    snr = snr_values[i]
    ind = metadata['SNR'] == snr

    ax[i].boxplot(1000 * results.values[ind,:])
    ax[i].set_xticklabels(methods)
    ax[i].set_title(f'{snr:.0f} dB')

plt.show()


