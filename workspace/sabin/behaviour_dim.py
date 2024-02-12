## MODULES ##

import sys
sys.path.append('c:/Users/cayde/OneDrive/Desktop/neuro_analysis')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from src.data_handling import ReduceDims
from src.visualization import BaseVisualizer


## GET COMPONENTS ##

reducer = ReduceDims(method='PCA')
visualizer = BaseVisualizer()
sheet_names = pd.ExcelFile('data/sabin/complete_pre.xlsx').sheet_names

for experiment in sheet_names:
    df = pd.read_excel('data/sabin/complete_pre.xlsx',sheet_name=experiment)
    data = df.drop(['ID'],axis=1).values
    labels = [i.split('_')[1] for i in df['ID']]

    scaled_data = (data-data.mean())/(data.std())
    components = pd.DataFrame(reducer.get_components(scaled_data),columns=['PC1','PC2'])
    components['labels'] = labels
    visualizer.scatter_plot(experiment,components,'PC1','PC2','labels',len(set(labels)))

    classifier = RandomForestClassifier(n_estimators=100,random_state=42)
    X_train,X_test,y_train,y_test = train_test_split(data,components['labels'],test_size=0.3,shuffle=True,random_state=42)

    classifier.fit(X_train,y_train)
    predictions = classifier.predict(X_test)
    cm = confusion_matrix(predictions,y_test)

    unique_labels = set(components['labels'])
    fig,ax = plt.subplots(figsize=(15,10))
    sns.heatmap(ax=ax,data=cm,annot=True,xticklabels=unique_labels,yticklabels=unique_labels)
    ax.set_title(experiment)
    plt.show()
