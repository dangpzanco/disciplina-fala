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


# results = np.load('data/objective_analysis/objective_results_extra.npz')
# results = pd.DataFrame(dict(results))
# del results['noisy']
# methods = list(dict(results).keys())


results_folder = pathlib.Path('data/exp_data/results')
subject_list = [item.stem for item in results_folder.glob('*')]

print(subject_list)

technique_list = ['noisy', 'wiener', 'bayes', 'binary']
num_techniques = len(technique_list)

subject_results = []
for i, sub in enumerate(subject_list):
    folder = results_folder / sub

    results_list = []
    for k in range(num_techniques):
        df = pd.read_csv(folder / f'exp_metadata{technique_list[k]}.csv')
        results_list.append(df)

    subject_results.append(pd.concat(results_list).reset_index(drop=True))

    if i == 0:
        method_column = [[technique_list[j]] * df.shape[0] for j in range(num_techniques)]
        method_column = np.array(method_column).ravel()


    print(method_column.shape, subject_results[i].shape)

    subject_results[i]['method'] = method_column
    subject_results[i]['subject'] = sub

    subject_results[i]['SNR'] = np.nan
    for k in range(subject_results[i].shape[0]):
        filename = subject_results[i]['filename'][k]
        subject_results[i]['SNR'][k] = float(filename.split('_')[-1])

    print(subject_results)

data = pd.concat(subject_results).reset_index(drop=True)

data['quality'][data['quality'] == 0] = 1

# exit(0)


# print(results)
# print(metadata)

# snr_values = np.array([-20,-10,-5,0,5,10,20,np.inf])



# print(methods)

# num_methods = len(methods)

# data = pd.concat([metadata] * num_methods).reset_index(drop=True)



# method_column = [[methods[i]] * metadata.shape[0] for i in range(num_methods)]
# method_column = np.array(method_column).ravel()

# data['method'] = method_column
# data['SNR increment [dB]'] = np.nan

# for tech in methods:
#     data['SNR increment [dB]'][method_column == tech] = results[tech].values


# print(data)

# # df1['e'] = pd.Series(np.random.randn(sLength), index=df1.index)

# data = data.drop(np.arange(data.shape[0])[data['SNR'] == np.inf]).reset_index(drop=True)

# # sns.set(style="ticks")

fig, ax = plt.subplots(figsize=(10,7))

sns.catplot(x='SNR', y='quality', hue='method', data=data, kind='box', ax=ax)

fig.set_tight_layout(0.1)
fig.savefig('images/subjective_quality.pdf', format='pdf')
fig.savefig('images/subjective_quality.png', format='png')
plt.show()





