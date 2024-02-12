## MODULES ##

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

from scipy.stats import ks_2samp

np.random.seed(42)

## FUNCTIONS ##


def calculate_angle(v1,v2):
    v1_norm = v1/np.linalg.norm(v1)
    v2_norm = v2/np.linalg.norm(v2)
    dot_product = np.dot(v1_norm,v2_norm)
    angle_rad = np.arccos(dot_product)

    return np.degrees(angle_rad)

path = 'data/tamas'
df = pd.read_csv(f'{path}/FS_raw.csv')
df.columns = [i.split('.')[0] for i in df.columns]
contexts = ['FS','HC_POST']

POST_data = df['HC_POST'].to_numpy().T
FS_data = df['FS'].to_numpy().T
FS_shuffle = np.copy(FS_data)
np.apply_along_axis(np.random.shuffle,1,FS_shuffle)

base_angles = []
for i in range(len(FS_shuffle)):
    for j in range(i+1,len(FS_shuffle)):
        base_angles.append(calculate_angle(FS_shuffle[i],FS_shuffle[j]))

compare_angles = []
for i in range(len(POST_data)):
    for j in range(len(FS_data)):
        compare_angles.append(calculate_angle(POST_data[i],FS_data[j]))

fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(15,10))
axes = axes.flatten()
for (ax,l) in zip(axes,[base_angles,compare_angles]):
    sns.histplot(ax=ax,data=l)
    
#plt.tight_layout()
#plt.show()

fig,ax = plt.subplots(figsize=(15,10))
sns.histplot(ax=ax,data=base_angles,color='blue',label='Baseline',stat='density')
sns.histplot(ax=ax,data=compare_angles,color='red',label='FS-HC_POST',stat='density')
ax.set_xlabel('Angles',fontsize=15,labelpad=25)
ax.set_ylabel('Density',fontsize=15,labelpad=25)

mean = np.mean(base_angles)
std = np.std(base_angles)
ax.axvline(x=mean,color='red')
ax.text(mean,ax.get_ylim()[1]*1.02,'Baseline Mean',horizontalalignment='center')
for i in range(1,3):
    plus = mean+(i*std)
    minus = mean-(i*std)
    ax.axvline(x=plus,color='red')
    ax.text(plus,ax.get_ylim()[1]*1.02,f'{i} STD',horizontalalignment='center')
    ax.axvline(x=minus,color='red')
    ax.text(minus,ax.get_ylim()[1]*1.02,f'{i} STD',horizontalalignment='center')

plt.legend()
plt.show()

k_stat,k_pvalue = ks_2samp(base_angles,compare_angles)
print(f'K Statistic: {k_stat}')
print(f'p-value: {k_pvalue}')

