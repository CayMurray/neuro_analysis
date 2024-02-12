import random
random.seed(42)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


path = 'data/tamas'
df = pd.read_csv(f'{path}/FS_raw.csv')
df.columns = [i.split('.')[0] for i in df.columns]
contexts = ['US+3', 'HC_POST+1', 'US+2', 'HC_PRE+2', 'HC_PRE', 'HC_POST']

for context in contexts:
    arr = df[[context]].T.to_numpy()
    block_size = 10
    split_index = len(arr)//2
    split_index = (split_index // block_size) * block_size

    labels = ['1st_half' if val < split_index else '2nd_half' for val in range(0,len(arr))]
    df_filtered = pd.DataFrame(arr)
    df_filtered['labels'] = labels

    first_half = df_filtered[df_filtered['labels']=='1st_half']
    first_half = [first_half.iloc[i:i+block_size] for i in range(0,len(first_half),block_size)]
    second_half = df_filtered[df_filtered['labels']=='2nd_half']
    second_half  = [second_half.iloc[i:i+block_size] for i in range(0,len(second_half),block_size)][:-1]
    random.shuffle(first_half),random.shuffle(second_half)
    train = pd.concat(first_half[:50] + second_half[:50],axis=0)
    test = pd.concat(first_half[50:] + second_half[50:],axis=0)
    
    X_train = train.drop(['labels'],axis=1)
    y_train = train['labels']
    X_test = test.drop(['labels'],axis=1)
    y_test = test['labels']

    rf = RandomForestClassifier(n_estimators=100,random_state=42)
    rf.fit(X_train,y_train)
    predicted = rf.predict(X_test)
    cm = confusion_matrix(predicted,y_test)
    unique_labels = sorted(set(y_train))

    fig,ax = plt.subplots(figsize=(15,10))
    sns.heatmap(ax=ax, data=cm, fmt='g',annot=True, cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
    ax.set_xlabel('Predicted',fontsize=20,labelpad=20)
    ax.set_ylabel('True',fontsize=20,labelpad=20)
    ax.set_title(f'{context} - 1st half & 2nd half',fontsize=30,pad=20)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)

    plt.savefig(f'../{context}.png')