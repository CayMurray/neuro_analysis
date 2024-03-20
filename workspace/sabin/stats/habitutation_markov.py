import sys
sys.path.append('/workspaces/neuro_analysis')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.linalg import expm

np.random.seed(42)

time_path = 'data/sabin/Full-MOFFT-Time.xlsx'
gen_path = 'data/sabin/Full-MOFFT-Gen.xlsx'
pairs = [('C5_No-Robot-MOFFT_Males_Beh','C6_Robot-MOFFT_Males_Beh'),
         ('C11_No-Robot-MOFFT_Females_Beh','C12_Robot-MOFFT_Females_Beh')]

for (no_r,r) in pairs:
    sex = no_r.split('_')[-2]
    array_dict = {'no_r':[],'r':[]}
    df_no = pd.read_excel(time_path,sheet_name=no_r).drop(['labels'],axis=1)
    df_r = pd.read_excel(gen_path,sheet_name=r).drop(['labels'],axis=1)

    for (label,df) in [('no_r',df_no),('r',df_r)]:
        for row in range(len(df)):
            indv_row = df.iloc[row]
            indv_array = indv_row.to_numpy().reshape((12,12)) if label == 'r' else indv_row
            array_dict[label].append(indv_array)

    no_robot_steady = np.mean(np.stack(array_dict['no_r']),axis=0)
    steady_norm = no_robot_steady/np.sum(no_robot_steady)
    robot_Q = np.sum(np.stack(array_dict['r']), axis=0) / len(array_dict['r'])
    pO = np.random.rand(12)
    pO = pO / np.sum(pO)

    distances = []
    time = []
    timesteps = np.arange(1,100000,100)
    for t in timesteps:
        pt = np.dot(pO,expm(robot_Q*t))
        distance = 0.5*(np.sum(np.abs(pt-pO)))
        distances.append(distance)
        time.append(t)

    distance_df = pd.DataFrame({'Distance':distances,'Time':time})
    fig,ax = plt.subplots(figsize=(15,10))
    sns.lineplot(ax=ax,data=distance_df,x='Time',y='Distance')
    ax.set_title(sex)
    #ax.set_ylim((0.3,0.5))
    plt.show()