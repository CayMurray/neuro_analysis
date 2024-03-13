import sys
sys.path.append('/workspaces/neuro_analysis')

import pandas as pd
import numpy as np

path = 'data/sabin/Full-MOFFT-Pre.xlsx'
sheet_names = [i for i in pd.ExcelFile(path).sheet_names if 'Tri' not in i]
unique_states = []

experiment_dict = {}
final_list = []

time_dict = {}
time_list = []

for experiment in sheet_names:
    df = pd.read_excel(path,sheet_name=experiment).dropna()
    df = df[[i for i in df.columns if 'behaviour' in i]]
    
    for i in df.columns:
        unique_states.append(pd.unique(df[df[i]!='VIDEO_END'][i]))

unique_states = set([i for l in unique_states for i in l])
state_index = {state:i for (i,state) in enumerate(unique_states)}
transition_labels = [f'{i}_{j}' for i in unique_states for j in unique_states]

for experiment in sheet_names:
    experiment_dict[experiment] = {'labels':[],**{t_label:[] for t_label in transition_labels}}
    time_dict[experiment] = {'labels':[],**{state:[] for state in unique_states}}
    raw_df = pd.read_excel(path,sheet_name=experiment)
    raw_df.columns = ['_'.join(i.split('_')[2:]) for i in raw_df.columns]
    
    for i in range(0,len(raw_df.columns),2):
        experiment_dict[experiment]['labels'].append(raw_df.columns[i])
        time_dict[experiment]['labels'].append(raw_df.columns[i])

        indv_df = raw_df[raw_df.columns[i]]
        indv_df = indv_df[indv_df.iloc[:,0]!='VIDEO_END']
        behaviour = indv_df.iloc[:,0].dropna()
        time = indv_df.iloc[:,1].dropna()
        generator_matrix = np.zeros((len(unique_states),len(unique_states)))
        state_holding = {state:0 for state in unique_states}

        for t in range(1,len(time)):
            prev_state = behaviour.iloc[t-1]
            curr_state = behaviour.iloc[t]
            holding_time = time.iloc[t] - time.iloc[t-1]
            
            state_holding[prev_state] += holding_time
            generator_matrix[state_index[prev_state],state_index[curr_state]] += 1
        
        for state in unique_states:
            time_dict[experiment][state].append(state_holding[state])

            if state_holding[state] != 0:
                generator_matrix[state_index[state],:] /= state_holding[state]
                pass

            generator_matrix[state_index[state],state_index[state]] = -np.sum(generator_matrix[state_index[state],:])
        
        generator_matrix = generator_matrix.flatten()
        for t_label,value in zip(transition_labels,generator_matrix):
            experiment_dict[experiment][t_label].append(value)

    final_df = pd.DataFrame(experiment_dict[experiment])
    final_list.append((experiment,final_df))
    time_df = pd.DataFrame(time_dict[experiment])
    time_list.append((experiment,time_df))
    
with pd.ExcelWriter('data/sabin/Full-MOFFT-Gen.xlsx') as writer:
    for (sheet_name,df) in final_list:
        df.to_excel(writer,sheet_name=sheet_name,index=False)

with pd.ExcelWriter('data/sabin/Full-MOFFT-Time.xlsx') as writer:
    for (sheet_name,df) in time_list:
        df.to_excel(writer,sheet_name=sheet_name,index=False)
        