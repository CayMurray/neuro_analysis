## IMPORTS ##

import sys
sys.path.append('/workspaces/neuro_analysis')

import pandas as pd
from scipy.stats import shapiro,levene, ttest_ind

import matplotlib.pyplot as plt
import seaborn as sns

time_path = 'data/sabin/Full-MOFFT-Time.xlsx'
gen_path = 'data/sabin/Full-MOFFT-Gen.xlsx'

male_path = 'C6_Robot-MOFFT_Males_Beh'
female_path = 'C12_Robot-MOFFT_Females_Beh'


## FUNCTIONS ##

def shapiro_wilk(group_1,group_2):
    p_list = []
    _,p_1 = shapiro(group_1)
    _,p_2 = shapiro(group_2)
    _,p_3 = levene(group_1,group_2)

    return all(i > 0.05 for i in [p_1,p_2,p_3])

def plot_barchart(df,y_label,p_value):
    fig,ax = plt.subplots(figsize=(15,10))
    sns.barplot(ax=ax,data=df,x='labels',y=y_label,hue='labels',palette='pastel',errorbar='sd')
    ax.set_xlabel('Sex',fontsize=20,labelpad=20)
    ax.set_ylabel(y_label,fontsize=20,labelpad=40)
    ax.text(0.01, 0.99, f'p-value: {p_value}', transform=ax.transAxes, verticalalignment='top', fontsize=15, color='black')
    plt.show()

def prepare_data(path,sheet_male,sheet_female,drop_list,gen=False):
    male = pd.read_excel(path,sheet_name=sheet_male).drop(drop_list,axis=1)
    female = pd.read_excel(path,sheet_name=sheet_female).drop(drop_list,axis=1)

    print(male.head())

    if gen:
        new_columns = [i for i in male.columns if i.split('_')[0]!=i.split('_')[1] and i.split('_')[1]!='N HomeBox']
        male= male[new_columns].sum(axis=1)
        female= female[new_columns].sum(axis=1)
        y_axis = 'Total Transition Rates'    

    else:
        male = male.sum(axis=1)
        female = female.sum(axis=1)
        y_axis = 'Total Holding Times'

    df = pd.DataFrame({y_axis:pd.concat([male,female],axis=0), 'labels': ['Male']*len(male) + ['Female']*len(female)})

    norm = shapiro_wilk(male,female)
    if norm:
        _,p_value = ttest_ind(male,female)

    plot_barchart(df,y_axis,p_value)


## GET STATS ##
    
#prepare_data(time_path,male_path,female_path,['labels','N HomeBox'])
prepare_data(gen_path,male_path,female_path,drop_list=['labels'],gen=True)
