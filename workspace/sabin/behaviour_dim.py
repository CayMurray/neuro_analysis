## MODULES ##

import sys
sys.path.append('c:/Users/cayde/OneDrive/Desktop/neuro_analysis')
import re

import pandas as pd

from src.data_handling import ReduceDims
from src.visualization import BaseVisualizer


## GET COMPONENTS ##

path = 'data/sabin/Full-MOFFT-Pre.xlsx'
reducer = ReduceDims(method='PCA')
visualizer = BaseVisualizer()
groups = ['Male','Female']
exclude = ['C7','C15','C19','C20']
exclude_pattern = '|'.join(re.escape(substring) for substring in exclude)

df_list = []
sheet_names = pd.ExcelFile(path).sheet_names
sheet_names = [i for i in sheet_names if not re.search(exclude_pattern,i)]

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
visualizer.scatter_plot(f'{groups[0],groups[1]}',components,'PCA_1','PCA_2','labels',2)

    

