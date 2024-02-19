## MODULES ##

import sys
sys.path.append('c:/Users/cayde/OneDrive/Desktop/neuro_analysis')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


## PREPARE DATA ##

master_dict = {}
abbrev_dict = {'Males':'Male','Females':'Female','No-Robot':'NoR','Robot':'R'}
experiments_list = []
sheet_names = pd.ExcelFile('data/sabin/Full-MOFFT-Data.xlsx').sheet_names
experiments = sorted([i for i in sheet_names if 'Beh' in i and 'Tri' not in i],key=lambda x: int(x.split('_')[0][1:]))

master_states = []
for experiment in experiments:
    df = pd.read_excel('data/sabin/Full-MOFFT-Data.xlsx',sheet_name=experiment)
    df = df[[i for i in df.columns if 'rat' in i]]
    for i in df.columns:
        values = pd.unique(df[(df[i]!='VIDEO_END') & (df[i]!='Behaviour')].dropna()[i])
        master_states.append(values)

master_states = set([i for l in master_states for i in l])

for experiment in experiments:
    master_dict[experiment] = {}
    sex = abbrev_dict[experiment.split('_')[-2]]
    df = pd.read_excel('data/sabin/Full-MOFFT-Data.xlsx',sheet_name=experiment)
    df = df[[i for (val,i) in enumerate(df.columns) if val%6==0 or val%6==1]]

    new_columns = []
    for (idx,col) in enumerate(df.columns):
        if 'RM' not in experiment:
            rat_num = idx//2
            robot = abbrev_dict['No-Robot' if 'No' in experiment else 'Robot']
            label = f"{col.split()[0] if ' ' in col else df.columns[idx+1].split()[0]}_{sex}_{robot}_day0"
            label = f'{rat_num}_{label}'
        
        elif 'RM' in experiment and 'MOFT' not in experiment:
            rat_num = int(col.split('rat')[-1])-1 if 'rat' in col else int(df.columns[idx-1].split('rat')[-1])-1
            robot = col.split()[1] if ' ' in col else df.columns[idx+1].split()[1]
            day = col.split('-')[0] if 'day' in col else df.columns[idx-1].split('-')[0]
            label = f'{rat_num}_FD_{sex}_{robot}_{day}'
            label = f"{label.split('.')[0]}"

            if day not in label.split('_')[-1]:
                label = f'{label}_{day}'

        else:
             rat_num = int(col.split('rat')[-1])-1 if 'rat' in col else int(df.columns[idx-1].split('rat')[-1])-1
             robot = col.split()[0] if ' ' in col else df.columns[idx+1].split()[0]
             day = col.split('-')[0] if 'day' in col else df.columns[idx-1].split('-')[0]

             if day == 'day1':
                 robot = 'NoR'

             label = f'{rat_num}_NM_{sex}_{robot}_{day}'

        new_columns.append(f'{idx//2}_behaviour_{label}' if idx%2==0 else f'{idx//2}_timestamp_{label}')

    df.columns = new_columns
    df = df.iloc[1:,:]

    #behaviours_df = df[[i for i in df.columns if 'behaviour' in i]]
    #non_nan_values = behaviours_df.fillna(value='VIDEO_END').values.flatten()
    #master_states = set([value for value in non_nan_values if value!='VIDEO_END'])

    for i in range(len(df.columns)//2):
        indv_columns = [j for j in df.columns if j.split('_')[0]==str(i)]
        label = indv_columns[0].split('_behaviour_')[-1]
        master_dict[experiment][label] = {state:[0] for state in master_states}
        indv_data = df[indv_columns].dropna().iloc[:-1]
        total_time = indv_data[f'{i}_timestamp_{label}'].iloc[-1]

        for t in range(1,len(indv_data[f'{i}_timestamp_{label}'])):
            prev_state = indv_data[f'{i}_behaviour_{label}'].iloc[t-1]
            curr_state = indv_data[f'{i}_behaviour_{label}'].iloc[t]
            holding = indv_data[f'{i}_timestamp_{label}'].iloc[t] - indv_data[f'{i}_timestamp_{label}'].iloc[t-1]

            if prev_state != curr_state:
                if master_dict[experiment][label][prev_state][0] == 0:
                    master_dict[experiment][label][prev_state].pop(0)

                master_dict[experiment][label][prev_state].append(holding)

        for state in master_states:
            total_holding = np.sum(master_dict[experiment][label][state])
            master_dict[experiment][label][state] = total_holding/total_time
    
    experiment_dict = {'labels':[], **{state:[] for state in master_states}}
    for rat_id,steady_state_dict in master_dict[experiment].items():
        experiment_dict['labels'].append(rat_id)
        for (state,state_value) in master_dict[experiment][rat_id].items():
            experiment_dict[state].append(state_value)

    experiments_list.append((experiment,pd.DataFrame(experiment_dict)))

with pd.ExcelWriter('data/sabin/Full-MOFFT-Pre.xlsx') as writer:
    for sheet_name,df in experiments_list:
        df.to_excel(writer,sheet_name=sheet_name,index=False)
