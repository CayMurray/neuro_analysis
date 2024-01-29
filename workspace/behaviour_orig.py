## MODULES ##

import sys
sys.path.append('c:/Users/cayde/OneDrive/Desktop/neuro_analysis')

import numpy as np
import pandas as pd


## PREPARE DATA ##

master_dict = {}

for experiment in ['Sated','FD']:
    master_dict[experiment] = {}
    df = pd.read_excel('data/sabin/Sabin.xlsx',sheet_name=f'{experiment}_Beh')
    df = df[[i for (val,i) in enumerate(df.columns) if val %3 != 2]]
    df.columns = [f'{i//2}_{"behaviour" if i%2 == 0 else "timestamp"}' for i in range(0,len(df.columns))]
    df = df.iloc[1:500]
    
    for i in range(len(df.columns)//2):
        master_dict[experiment][i] = {}
        indv_data = df[[f'{i}_behaviour',f'{i}_timestamp']].dropna()[:-1]

        unique_states = set(indv_data[f'{i}_behaviour'])
        n_states = len(unique_states)
        state_index = {state:i for (i,state) in enumerate(unique_states)}
        
        rate_matrix = np.zeros((n_states,n_states))
        holding_time = np.zeros(n_states)

        for t in range(1, len(indv_data)):
            prev_state = indv_data[f'{i}_behaviour'].iloc[t-1]
            curr_state = indv_data[f'{i}_behaviour'].iloc[t]
            time_spent = indv_data[f'{i}_timestamp'].iloc[t] - indv_data[f'{i}_timestamp'].iloc[t-1]

            holding_time[state_index[prev_state]] += time_spent
            if prev_state != curr_state:
                rate_matrix[state_index[prev_state], state_index[curr_state]] += 1

        for n in range(n_states):
            if holding_time[n] != 0:
                rate_matrix[n, :] /= holding_time[n]

            rate_matrix[n, n] = -np.sum(rate_matrix[n, :])

        eig_values,eig_vectors = np.linalg.eig(rate_matrix.T)
        real_values,real_vectors = eig_values.real,eig_vectors.real
        index = np.argmin(np.abs(real_values))
        steady = real_vectors[:,index]
        steady /= np.sum(steady)
