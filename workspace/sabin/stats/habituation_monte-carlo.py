## MODULES ##

import sys
sys.path.append('/workspaces/neuro_analysis')

from collections import Counter

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.linalg import expm

np.random.seed(42)


## FUNCTIONS ##

def sort_generator(col_name):
    parts = col_name.split('_')
    return parts
    
def reorder_sheets(path,sheet_name='C5_No-Robot-MOFFT_Males_Beh',func=None):
    df = pd.read_excel(path,sheet_name=sheet_name)
    labels = df[['labels']]
    values = df.drop(['labels'],axis=1)
    
    reordered_columns = sorted(values.columns,key=func)
    sorted_df = values[reordered_columns]
    sorted_df.insert(0,'labels',labels['labels'])

    return sorted_df


def coin_flip():
    num_of_samples = 100
    transition_matrix = np.array([[0.53, 0.47],
                                [0.6, 0.4]])

    alpha = [1,1]
    samples = np.random.dirichlet(alpha,num_of_samples)

    eigvals,eigvectors = np.linalg.eig(transition_matrix.T)

    for (val,sample) in enumerate(samples):
        pt = sample
        for t in range(1000):
            pt = np.dot(pt,transition_matrix)

        if val%10 == 0:
            print(f'Initial: {sample}, Theoretical: {eigvectors[:,0]/np.sum(eigvectors[:,0])}, Experimental: {pt}')

def markets():
    samples = np.random.dirichlet([1]*3,100)

    Q = np.array([[-.025,0.02,0.005],
                [0.3,-0.5,0.2],
                [0.02,0.4,-0.42]])

    eigvals,eigvectors = np.linalg.eig(Q.T)
    steady_state = eigvectors[:,np.argmin(np.abs(eigvals))]
    steady_state = steady_state / np.sum(steady_state)

    for sample in samples:
        distances = []

        pt = np.dot(sample,expm(Q*1000))
        print(f'Theoretical:{steady_state}, Experimental:{pt}')


def rodents():
    time_df = reorder_sheets('data/sabin/Full-MOFFT-Time.xlsx').drop(['labels'],axis=1)
    gen_df = reorder_sheets('data/sabin/Full-MOFFT-Gen.xlsx',func=sort_generator).drop(['labels'],axis=1)
    samples = np.random.dirichlet([1]*12,100)

    for rat in range(time_df.shape[0]):
        time = time_df.iloc[rat,:].values
        Q = gen_df.iloc[rat,:].values.reshape((12,12))
        eigvals,eigvectors = np.linalg.eig(Q.T)
        rounded_vals = np.real(np.round(eigvals,decimals=5))

        eig_dict = {}
        multi = Counter()
        for (index,i) in enumerate(rounded_vals):
            multi[i] +=1
            if np.abs(i) == 0.0:
                eig_dict[index] = np.real(eigvectors[:,index]/np.sum(eigvectors[:,index]))

        for sample in samples:
            pt = np.dot(sample,expm(Q*1000))

        for (key,vector) in eig_dict.items():
            distance = 0.5*(np.sum(np.abs(pt-vector)))


## MONTE-CARLO ##

rodents()

    
            
    



