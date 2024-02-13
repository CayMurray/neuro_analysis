## MODULES ##

import sys
sys.path.append('/workspaces/neuro_analysis')
import re

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.data_handling import ReduceDims
from src.visualization import BaseVisualizer


## GET COMPONENTS ##

path = 'data/sabin/Full-MOFFT-Pre.xlsx'
reducer = ReduceDims(method='PCA')
visualizer = BaseVisualizer()
include = ['C5','C6','C11','C12']
include_pattern = '|'.join(re.escape(substring) for substring in include)
groups = ['FD','Sated']

df_list = []
sheet_names = pd.ExcelFile(path).sheet_names
sheet_names = [i for i in sheet_names if re.search(include_pattern,i)]

for (val,experiment) in enumerate(sheet_names):
    df = pd.read_excel(path,sheet_name=experiment)
    df_list.append(df)

    if val == 0:
        master_columns = df.columns[1:]
    
master_df = pd.concat(df_list,axis=0)
include = '|'.join(groups)
filtered_df = master_df[master_df['ID'].str.contains(include,na=False)]
filtered_df.loc[:,'ID'] = filtered_df['ID'].apply(lambda x: groups[0] if groups[0] in x else groups[1])
data = filtered_df.drop(['ID'],axis=1).values
data = (data-data.mean())/(data.std())

components = reducer.get_components(data)
components['labels'] = filtered_df['ID'].values
num_of_labels = len(set(components['labels']))


## VISUALIZE AND COMPARE DISTRIBUTIONS ##

#visualizer.scatter_plot(f'{groups[0]},{groups[1]}',components,'PCA_1','PCA_2','labels',num_of_labels)

intra_differences = []
inter_differences = []
for label in set(components['labels']):
    intra_data = components[components['labels'].str.contains(label)].drop(['labels'],axis=1)
    inter_data = components[~components['labels'].str.contains(label)].drop(['labels'],axis=1)

    for sample_1 in range(len(intra_data)):
        point_1 = intra_data.iloc[sample_1,:].values
        
        for sample_2 in range(len(intra_data)):
            if sample_1 != sample_2:
                point_2 = intra_data.iloc[sample_2,:].values
                intra_diff = point_2 - point_1
                intra_differences.append(np.sqrt(np.sum(intra_diff**2)))

        for sample_3 in range(len(inter_data)):
            point_3 = inter_data.iloc[sample_3,:].values
            inter_diff = point_3 - point_1
            inter_differences.append(np.sqrt(np.sum(inter_diff**2)))

kstat,kpvalue = ks_2samp(intra_differences,inter_differences)
print(f'kstat: {kstat}', f'p-value: {kpvalue}')

