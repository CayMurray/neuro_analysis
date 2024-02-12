## MODULES ##

import sys
sys.path.append('c:/Users/cayde/OneDrive/Desktop/neuro_analysis')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


## PREPARE DATA ##

master_dict = {}
experiments_list = []
sheet_names = pd.ExcelFile('data/sabin/complete.xlsx').sheet_names
experiments = [i for i in sheet_names if 'Beh' in i and 'RM' not in i]

for experiment in experiments:
    master_dict[experiment] = {'Sated':{},'FD':{}}
    df = pd.read_excel('data/sabin/complete.xlsx',sheet_name=experiment)
    df = df[[i for (val,i) in enumerate(df.columns) if val%6==0 or val%6==1]]
    
    new_columns = []
    for (idx,col) in enumerate(df.columns):
        label = col.split()[0] if ' ' in col else df.columns[idx+1].split()[0]
        new_columns.append(f'{idx//2}_behaviour_{label}' if idx%2==0 else f'{idx//2}_timestamp_{label}')

    df.columns = new_columns
    df = df.iloc[1:,:]

    behaviours_df = df[[i for i in df.columns if 'behaviour' in i]]
    non_nan_values = behaviours_df.fillna(value='VIDEO_END').values.flatten()
    master_states = set([value for value in non_nan_values if value!='VIDEO_END'])

    for i in range(len(df.columns)//2):
        indv_columns = [j for j in df.columns if j.split('_')[0]==str(i)]
        label = indv_columns[0].split('_')[2]
        master_dict[experiment][label][i] = {state:[0] for state in master_states}
        indv_data = df[indv_columns].dropna().iloc[:-1]
        total_time = indv_data[f'{i}_timestamp_{label}'].iloc[-1]

        for t in range(1,len(indv_data[f'{i}_timestamp_{label}'])):
            prev_state = indv_data[f'{i}_behaviour_{label}'].iloc[t-1]
            curr_state = indv_data[f'{i}_behaviour_{label}'].iloc[t]
            holding = indv_data[f'{i}_timestamp_{label}'].iloc[t] - indv_data[f'{i}_timestamp_{label}'].iloc[t-1]

            if prev_state != curr_state:
                if master_dict[experiment][label][i][prev_state][0] == 0:
                    master_dict[experiment][label][i][prev_state].pop(0)

                master_dict[experiment][label][i][prev_state].append(holding)

        for state in master_states:
            total_holding = np.sum(master_dict[experiment][label][i][state])
            master_dict[experiment][label][i][state] = total_holding/total_time
            
    experiment_dict = {'ID':[], **{state:[] for state in master_states}}
    for label in master_dict[experiment]:
        for (id,states_dict) in master_dict[experiment][label].items():
            experiment_dict['ID'].append(f'{id}_{label}')
            for state in states_dict:
                experiment_dict[state].append(states_dict[state])

    experiments_list.append((experiment,pd.DataFrame(experiment_dict)))

with pd.ExcelWriter('data/sabin/complete_pre.xlsx') as writer:
    for sheet_name,df in experiments_list:
        df.to_excel(writer,sheet_name=sheet_name,index=False)
