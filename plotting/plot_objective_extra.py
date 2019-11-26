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


results = np.load('data/objective_analysis/objective_results_extra.npz')
results = pd.DataFrame(dict(results))
del results['noisy']
methods = list(dict(results).keys())

print(results)
print(metadata)

snr_values = np.array([-20,-10,-5,0,5,10,20,np.inf])



print(methods)

num_methods = len(methods)

data = pd.concat([metadata] * num_methods).reset_index(drop=True)



method_column = [[methods[i]] * metadata.shape[0] for i in range(num_methods)]
method_column = np.array(method_column).ravel()

data['method'] = method_column
data['SNR increment [dB]'] = np.nan

for tech in methods:
    data['SNR increment [dB]'][method_column == tech] = results[tech].values


print(data)

# df1['e'] = pd.Series(np.random.randn(sLength), index=df1.index)

data = data.drop(np.arange(data.shape[0])[data['SNR'] == np.inf]).reset_index(drop=True)

# sns.set(style="ticks")

fig, ax = plt.subplots(figsize=(10,7))

sns.catplot(x='SNR', y='SNR increment [dB]', hue='method', data=data, kind='box', ax=ax)

fig.set_tight_layout(0.1)
fig.savefig('images/objective_snr_increment.pdf', format='pdf')
fig.savefig('images/objective_snr_increment.png', format='png')
plt.show()





