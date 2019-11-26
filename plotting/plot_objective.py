import pathlib
import soundfile as sf
import librosa
import numpy as np
import numpy.random as rnd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import trange



metadata = pd.read_csv('metadata.csv')


results = np.load('data/objective_analysis/objective_results.npz')
methods = list(dict(results).keys())
results = pd.DataFrame(dict(results))

results = np.log(results)

print(results)
print(metadata)

snr_values = np.array([-20,-10,-5,0,5,10,20,np.inf])




# fig, ax = plt.subplots(2,4,figsize=(10,7))
# ax = ax.ravel()

# for i in range(snr_values.size):

#     snr = snr_values[i]
#     ind = metadata['SNR'] == snr

#     ax[i].boxplot(1000 * results.values[ind,:])
#     ax[i].set_xticklabels(methods)
#     ax[i].set_title(f'{snr:.0f} dB')




# data = pd.DataFrame(index=[],columns=['filename', 'speech_name', 'noise_name', 'realization', 'SNR', 'method', 'MSE'])


print(methods)

num_methods = len(methods)

data = pd.concat([metadata] * num_methods).reset_index(drop=True)



method_column = [[methods[i]] * metadata.shape[0] for i in range(num_methods)]
method_column = np.array(method_column).ravel()

data['method'] = method_column
data['logMSE'] = np.nan

for tech in methods:
    data['logMSE'][method_column == tech] = results[tech].values



print(data)

# df1['e'] = pd.Series(np.random.randn(sLength), index=df1.index)


# sns.set(style="ticks")

fig, ax = plt.subplots(figsize=(10,7))
sns.catplot(x='SNR', y='logMSE', hue='method', data=data, kind='box', ax=ax)

fig.set_tight_layout(0.1)
fig.savefig('images/objective_analysis.pdf', format='pdf')
fig.savefig('images/objective_analysis.png', format='png')
plt.show()





