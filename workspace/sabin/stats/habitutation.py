import sys
sys.path.append('/workspaces/neuro_analysis')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.linalg import eig, expm


path = 'data/sabin/Full-MOFFT-Gen.xlsx'
pairs = [('C5_No-Robot-MOFFT_Males_Beh','C6_Robot-MOFFT_Males_Beh'),
         ('C11_No-Robot-MOFFT_Females_Beh','C12_Robot-MOFFT_Females_Beh')]

for (no_r,r) in pairs:
    generator_dict = {'no_r':[],'r':[]}
    df_no = pd.read_excel(path,sheet_name=no_r).drop(['labels'],axis=1)
    df_r = pd.read_excel(path,sheet_name=r).drop(['labels'],axis=1)

    for (label,df) in [('no_r',df_no),('r',df_r)]:
        for row in range(len(df)):
            indv_row = df.iloc[row]
            indv_Q = indv_row.to_numpy().reshape((12,12))
            generator_dict[label].append(indv_Q)

    no_robot_Q = np.sum(np.stack(generator_dict['no_r']), axis=0) / len(generator_dict['no_r'])
    eigenvalues,eigenvectors = eig(no_robot_Q.T)
    index = np.argmin(np.abs(eigenvalues))
    ss_vector = np.real(eigenvectors[:,index])
    norm_vector = ss_vector / np.sum(ss_vector)

    robot_Q = np.sum(np.stack(generator_dict['r']), axis=0) / len(generator_dict['r'])
    pO = np.ones(12)/12

    distances = []
    time = []
    timesteps = np.arange(1,10000,100)
    for t in timesteps:
        pt = np.dot(pO,expm(no_robot_Q*t))
        distance = np.linalg.norm(pt-norm_vector)
        distances.append(distance)
        time.append(t)

    distance_df = pd.DataFrame({'Distance':distances,'Time':time})
    fig,ax = plt.subplots(figsize=(15,10))
    sns.lineplot(ax=ax,data=distance_df,x='Time',y='Distance')
    plt.show()