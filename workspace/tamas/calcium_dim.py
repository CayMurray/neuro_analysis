import sys
sys.path.append('/workspaces/neuro_analysis')

import os
import random
random.seed(10)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex 

from src.data_handling import ReduceDims
from src.visualization import BaseVisualizer

df = pd.read_csv('data/tamas/non-FS_raw.csv')
contexts = [i.split('.')[0] if '.' in i else i for i in df.columns]
df = pd.DataFrame(df.T.values)
df['labels'] = contexts

reducer = ReduceDims()
visualizer = BaseVisualizer(fontsize=30,figsize=(10,6))
data = df.drop(['labels'],axis=1).values
labels = df['labels']
scaled_data = (data - data.mean())/(data.std())

reduced = reducer.get_components(scaled_data)
reduced['labels'] = labels
reduced = reduced[reduced['labels'].isin(['HC_PRE','US_PRE','HC_POST'])]

colours = [to_hex([random.random() for _ in range(3)]) for _ in range(3)]

for contexts in [[''],['HC_PRE'],['HC_PRE','US_PRE'],['HC_PRE','US_PRE','HC_POST']]:
    copy = reduced.copy()
    bool_vector = copy['labels'].isin(contexts)
    flipped_bool = np.logical_not(bool_vector)
    copy.loc[flipped_bool,['PCA_1','PCA_2']] = -500
    fig,ax = plt.subplots(figsize=(15,10))
    sns.scatterplot(ax=ax,data=copy,x='PCA_1',y='PCA_2',hue='labels')
    ax.set_xlim((-6,8))
    ax.set_ylim((-8,8))
    ax.set_title('Principal Components of FS Mice',fontsize=30,pad=20)
    ax.set_xlabel('PCA_1',fontsize=20,labelpad=20)
    ax.set_ylabel('PCA_2',fontsize=20,labelpad=20)
    plt.show()
    fig.savefig(f'{contexts}.png',dpi=300)

for i in os.listdir():
    if i.endswith('png'):
        os.remove(i)
        pass
