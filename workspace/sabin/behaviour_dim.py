## MODULES ##

import sys
sys.path.append('/workspaces/neuro_analysis')

import random
random.seed(42)

import os
import re

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from src.data_handling import ReduceDims
from src.visualization import BaseVisualizer


## FUNCTIONS ##

def extract_groups(id,groups,groups_dict):
    parts = id.split('_')
    return '_'.join(parts[groups_dict[group]] for group in groups)


def plot_loadings(title,loadings,variance_ratio):
    num_of_components = loadings.shape[1]
    fig, axs = plt.subplots(nrows=1, ncols=num_of_components, figsize=(15, 5), sharey=False)

    for i, pc in enumerate(loadings.columns):
        temp_df = loadings[[pc]].copy()
        temp_df['Feature'] = temp_df.index
        temp_df['Sign'] = ['Positive' if x >= 0 else 'Negative' for x in temp_df[pc]]

        sns.barplot(ax=axs[i],data=temp_df,x=pc,y='Feature',hue='Sign',legend=False)
        axs[i].set_title(f'{pc} ({round(variance_ratio[i]*100,2)} % of variance)')
        axs[i].set_xlabel('Loadings',fontsize=15,labelpad=20)
        axs[i].set_xlim(-1, 1)  
        axs[i].set_ylabel('Original Features',fontsize=15,labelpad=30)

        if i > 0:
            axs[i].set_ylabel('')
            axs[i].set_yticklabels([])
            axs[i].tick_params(left=False)

    plt.suptitle(title, fontsize=30, y=0.97)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def compare_distributions(components):
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
    #print(f'kstat: {kstat}', f'p-value: {kpvalue}')

    return intra_differences,inter_differences,kstat,kpvalue


## PREPARE DATA ##

path = 'data/sabin/Full-MOFFT-Pre.xlsx'
experiments = ['C5','C6','C11','C12']
groups = ['sex','robot'] # rat, food, sex, robot

groups_dict = {'rat':0,'food':1 ,'sex':2,'robot':3,'day':4}
reducer = ReduceDims(method='PCA',n_components=2)
visualizer = BaseVisualizer()

experiments= '|'.join(re.escape(substring) for substring in experiments)
sheet_names = pd.ExcelFile(path).sheet_names
sheet_names = [i for i in sheet_names if re.search(experiments,i)]

df_list = []
for (val,experiment) in enumerate(sheet_names):
    df = pd.read_excel(path,sheet_name=experiment)
    df_list.append(df)

    if val == 0:
        master_columns = df.columns[1:]

master_df = pd.concat(df_list,axis=0)
master_df.loc[:,'labels'] = master_df['labels'].apply(lambda x: extract_groups(x,groups,groups_dict))
data = master_df.drop(['labels'],axis=1).values
data = (data-data.mean())/(data.std())

components = reducer.get_components(data)
components['labels'] = master_df['labels'].values

loadings,variance_ratio  = reducer.get_loadings()
loadings.index = master_df.columns[1:]


## VISUALIZE ##

visualizer.scatter_plot("PCA Space of Rats with Sex and Robot as Variables",components,'labels')
#plot_loadings(f'PCA Loadings for Rats',loadings,variance_ratio)

'''

## DISTRIBUTIONS ##

#intra,inter,kstat,kpvalue = compare_distributions(master_df)
#visualizer.bar_chart('Intra vs Inter Distances with Sex and Robot as Variables',intra=intra,inter=inter,ks=(kstat,kpvalue))

## MACHINE LEARNING ##

model = RandomForestClassifier(n_estimators=100,random_state=42)

X = master_df.drop(['labels'],axis=1)
y = master_df['labels']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,shuffle=True,random_state=42)

model.fit(X_train,y_train)
predictions = model.predict(X_test)
cm = confusion_matrix(predictions,y_test)
unique_labels = set(y)

#visualizer.heat_map('Confusion Matrix from Random Forest Classifier',data=cm,xticks=unique_labels,yticks=unique_labels)
'''

for i in os.listdir():
    if i.endswith('png'):
        os.remove(i)
        pass
