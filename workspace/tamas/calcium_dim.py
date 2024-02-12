import sys
sys.path.append('c:/Users/cayde/OneDrive/Desktop/neuro_analysis')

import numpy as np
import pandas as pd

from src.data_handling import ReduceDims
from src.visualization import BaseVisualizer

df = pd.read_csv('data/tamas/FS_raw.csv')
contexts = [i.split('.')[0] if '.' in i else i for i in df.columns]
df = pd.DataFrame(df.T.values)
df['labels'] = contexts

reducer = ReduceDims()
visualizer = BaseVisualizer(fontsize=30,figsize=(10,6))
data = df.drop(['labels'],axis=1).values
labels = df['labels']
scaled_data = (data - data.mean())/(data.std())

components = reducer.get_components(scaled_data)
reduced = pd.DataFrame(data=components,columns=['PC1','PC2'])
reduced['labels'] = labels
reduced = reduced[reduced['labels'].isin(['HC_PRE','US_PRE','FS','HC_POST'])]
colour = len(set(reduced['labels']))

visualizer.scatter_plot('Principal Components from Footshock Mice',reduced,'PC1','PC2','labels',colour)
