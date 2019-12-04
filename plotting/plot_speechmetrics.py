import pathlib
import soundfile as sf
import librosa
import numpy as np
import numpy.random as rnd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import trange



def format_figure(fig, ax):

    figsize = np.array([16, 10])/2.54
    
    par_ticks = {'fontname': 'Arial',
                 'weight': 'normal',
                 'fontsize' : '10'}
    
    par_labels = {'family': 'Times New Roman',
                  'weight' : 'normal',
                  'fontsize' : '12'}
    
    # Adjust the axis inside the figure
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.14, top=0.975, wspace=0, hspace=0)
    
    #fig.subplots_adjust(left=0.116, right=0.965, bottom=0.17, top=0.98, wspace=0, hspace=0)
    # Usar para 9cm por 4,5cm
    
    # Adjust figure size
    fig.set_size_inches(figsize)

    # legend_title = ax.get_legend().get_title().get_text().capitalize()
    # legend_text = [text.get_text().capitalize() for text in ax.get_legend().get_texts()]
    # print(legend_title, legend_text)
    # ax.legend(legend_text, title=legend_title)
    
    # Adjust the font of x and y labels
    ax.set_xlabel(ax.get_xlabel(), **par_labels)
    ax.set_ylabel(ax.get_ylabel(), **par_labels)
   
    # Set the font name for axis tick labels
    for tick in ax.get_xticklabels():
        tick.set_fontname(par_ticks['fontname'])
        tick.set_fontsize(par_ticks['fontsize'])
        tick.set_weight(par_ticks['weight'])
        
    for tick in ax.get_yticklabels():
        tick.set_fontname(par_ticks['fontname'])
        tick.set_fontsize(par_ticks['fontsize'])
        tick.set_weight(par_ticks['weight'])

    # fig.set_tight_layout(0.01)





# results = pd.read_csv('../analysis/speechmetrics_results.csv')
results = pd.read_csv('../analysis/matlab_results.csv')

results_custom = pd.read_csv('../analysis/custom_results.csv')
results['snrg'] = results_custom['snrg']

# Drop SNR = Inf dB
drop_index = (results['SNR'] == np.inf).values
drop_index = np.where(drop_index)[0]
results = results.drop(drop_index)

# Normalize LLR
results['llr'] = (2 - results['llr'])/2

metric = 'pesq'
# metric = 'stoi'
# metric = 'srmr'

metric_list = ['pesq', 'stoi', 'srmr', 'llr', 'csii', 'snrg']

kind = 'violin'
kind = 'box'
# kind = 'bar'

# hue_order = ['noisy', 'wiener', 'bayes', 'binary']
hue_order = ['noisy', 'binary', 'wiener', 'bayes']


for i, metric in enumerate(metric_list):
    fig, ax = plt.subplots(figsize=(10,7))
    
    snsfig = sns.catplot(x='SNR', y=metric, hue='technique', ax=ax,
        hue_order=hue_order, data=results, kind=kind)
    plt.close(snsfig.fig)

    ax.set_ylabel(metric.upper())
    ax.set_xlabel('SNR [dB]')
    format_figure(fig, ax)

    fig.savefig(f'../images/speechmetrics_{metric}.pdf', format='pdf', transparent=True)
    fig.savefig(f'../images/speechmetrics_{metric}.png', format='png', transparent=True)

plt.show()





