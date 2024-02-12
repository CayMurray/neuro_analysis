## MODULES ##

import sys 
sys.path.append('c:/Users/cayde/OneDrive/Desktop/neuro_analysis')

import pandas as pd

from src.base import LoadData,ReduceDims,Visualize


## LOAD AND REDUCE DATA ##

path = 'data/canbind'
loader = LoadData()
reducer = ReduceDims(method='UMAP')
visualizer = Visualize()

'''

data = loader.mat(f'{path}/barcodes.mat')['barcodes']
labels = [[i]*10 for i in range(1,11)]
labels = [i for l in labels for i in l]

reduced_data = pd.DataFrame(reducer.get_components(data),columns=['UMAP_1','UMAP_2'])
reduced_data['labels'] = labels

visualizer.scatter_plot(reduced_data,'UMAP_1','UMAP_2','labels','Midnight UMAP')

'''