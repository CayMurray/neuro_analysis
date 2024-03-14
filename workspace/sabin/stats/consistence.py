## IMPORTS ##

import sys
sys.path.append('/workspaces/neuro_analysis')

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import shapiro,friedmanchisquare
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp


label_dict = {'ID':0,'sex':1,'day':2}
path = 'data/sabin/Full-MOFFT-Time.xlsx'
sheet_name = 'C19_RM_Males_Beh'

## FUNCTIONS ##

def convert_labels(x):
    indv_labels = x.split('_')
    new_label = '_'.join(indv_labels[i] for i in [0,2,4])

    return new_label

def prepare_data(path,sheet_name,type='holding'):
    new_dict = {'ID':[],'sex':[],'day':[],'metric':[]}
    data = pd.read_excel(path,sheet_name=sheet_name)
    df = data[data['labels'].str.contains('_NoR_')]
    df.loc[:,'labels'] = df['labels'].apply(lambda x: convert_labels(x))
    labels = df['labels']
    df = df.drop(['labels'],axis=1)

    if type == 'holding':
        df = df.drop(['N HomeBox'],axis=1)

    elif type == 'gen':
        df = df[[i for i in df.columns if i.split('_')[0]!=i.split('_')[1] and i.split('_')[1] !='N HomeBox']]

    for (row_num,label) in enumerate(labels):
        parts = label.split('_')

        for key in new_dict.keys():
            if key != 'metric':
                new_dict[key].append(parts[label_dict[key]])

        aggregate = df.sum(axis=1).iloc[row_num]
        new_dict['metric'].append(aggregate)

    return pd.DataFrame(new_dict)

def compare_days(df,chart='line'):
    p_list = []
    for day in ['day1','day2','day3']:
        day_df = df[df['day'].str.contains(day)]
        _,p_value = shapiro(day_df['metric'])
        p_list.append(p_value)

    if all(i>0.05 for i in p_list):
        rm_anova = AnovaRM(data=df, depvar='metric', subject='ID', within=['day'])
        results = rm_anova.fit()
        tukey_results = pairwise_tukeyhsd(endog=df['metric'],groups=df['day'])
        print(tukey_results)

    else:
        pivoted_df = df.pivot(index='ID', columns='day', values='metric')
        stat,p = friedmanchisquare(pivoted_df['day1'],pivoted_df['day2'],pivoted_df['day3'])
        nemenyi_results = sp.posthoc_nemenyi_friedman(pivoted_df)
        print(nemenyi_results)

    fig,ax = plt.subplots(figsize=(15,10))

    if chart == 'line':
        sns.lineplot(ax=ax,data=df,x='day',y='metric',hue='ID',palette='tab20',legend='full',alpha=0.7)
        ax.legend(title='Rat ID', bbox_to_anchor=(1.05, 1), loc='upper left')

    elif chart == 'box':
        sns.boxplot(ax=ax,data=df,x='day',y='metric',hue='day',palette='tab20')

    #ax.set_title('Holding Time Across Days for Each Rat',fontsize=40)
    ax.set_xlabel('Day',fontsize=20,labelpad=20)
    ax.set_ylabel('Metric',fontsize=20,labelpad=40)
    plt.tight_layout()
    plt.show()

## PREPARE DATA ##
    
final_df = prepare_data(path,sheet_name,type='holding')


## STATS ##

compare_days(final_df,chart='line')