import sys
sys.path.append('c:/Users/cayde/OneDrive/Desktop/neuro_analysis')

import os 
import pandas as pd

from src.data_handling import LoadData,ReduceDims
from src.visualization import BaseVisualizer

path = 'data/canbind'

loader = LoadData()
reducer = ReduceDims(method='UMAP')
visualizer = BaseVisualizer()

for i in range(1,3):
    file_path = f'{path}/week_{str(i)}.mat'
    data = loader.mat(file_path)['barcodes']
    reduced_components = reducer.get_components(data)
    df = pd.DataFrame(data=reduced_components,columns=['UMAP1','UMAP2'])
    visualizer.scatter_plot(f'week_{str(i)}',df,'UMAP1','UMAP2',None,1)