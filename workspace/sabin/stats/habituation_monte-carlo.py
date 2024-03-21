## MODULES ##

import sys
sys.path.append('/workspaces/neuro_analysis')

from collections import Counter

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.linalg import expm
from scipy.spatial.distance import jensenshannon
from scipy.stats import shapiro, ttest_ind, mannwhitneyu

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


def rodents(sheet_name = 'C19_RM_Males_Beh'):
    #labels = reorder_sheets('data/sabin/Full-MOFFT-Time.xlsx',sheet_name=sheet_name)['labels']
    time_df = reorder_sheets('data/sabin/Full-MOFFT-Time.xlsx',sheet_name=sheet_name).drop(['labels'],axis=1)
    gen_df = reorder_sheets('data/sabin/Full-MOFFT-Gen.xlsx',sheet_name=sheet_name,func=sort_generator).drop(['labels'],axis=1)
    samples = np.random.dirichlet([1]*12,100)

    multi_list = []
    prob_list = []
    for rat in range(time_df.shape[0]):
        Q = gen_df.iloc[rat,:].values.reshape((12,12))
        eigvals,eigvectors = np.linalg.eig(Q.T)
        rounded_vals = np.real(np.round(eigvals,decimals=5))

        eig_dict = {}
        multi = Counter()
        for (index,i) in enumerate(rounded_vals):
            multi[i] +=1
            if np.abs(i) == 0.0:
                eig_dict[index] = np.real(eigvectors[:,index]/np.sum(eigvectors[:,index]))

        multi = dict(sorted(multi.items(),key=lambda x: np.abs(x[0])))
        multi_list.append([i for i in multi.values()][0])

        for sample in samples:
            pt = np.dot(sample,expm(Q*1000))
            prob_list.append(pt)

    '''
    df = pd.DataFrame(multi_list, columns=['Multiplicities'])
    fig,ax = plt.subplots()
    sns.histplot(ax=ax, data=df, x='Multiplicities', discrete=True, shrink=0.8)
    ax.set_title('Histogram of Multiplicities',fontsize=30,pad=20)
    ax.set_xlabel('Multiplicities of Eigenvalues',fontsize=30,labelpad=20)
    ax.set_ylabel('Count',fontsize=30,labelpad=20)
    ax.set_xticks(range(min(multi_list), max(multi_list) + 1))

    plt.show()
    '''

    return prob_list

    
## MONTE-CARLO ##

main_dict = {}

for (df1,df2) in [('C5_No-Robot-MOFFT_Males_Beh','C6_Robot-MOFFT_Males_Beh'),('C11_No-Robot-MOFFT_Females_Beh','C12_Robot-MOFFT_Females_Beh')]:
    js_diverge = []
    d1 = rodents(df1)
    d2 = rodents(df2)
    
    for vec1 in d1:
        for vec2 in d2:
            js_diverge.append(jensenshannon(vec1,vec2,base=2)**2)

    main_dict[df1.split('_')[-2]] = js_diverge

p_values = []
for (key,value) in main_dict.items():
    _,p = shapiro(value)
    p_values.append(p)

if any(i < 0.05 for i in p_values):
    u,p = mannwhitneyu(main_dict['Males'],main_dict['Females'],alternative='two-sided')
    print(u,p)

print(f'Males: {np.median(main_dict["Males"])}')
print(f'Females: {np.median(main_dict["Females"])}')

fig,ax = plt.subplots()
for (sex,value) in main_dict.items():
    sns.histplot(ax=ax,data=value,label=sex)

ax.set_title('JS Divergence Values from Single-day Experiments',fontsize=30,pad=20)
ax.set_xlabel('JS Divergence',fontsize=30,labelpad=20)
ax.set_ylabel('Counts',fontsize=30,labelpad=20)

plt.legend()
plt.show()


    
            
    



