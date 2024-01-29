## MODULES ##

from abc import ABC 

from math import gcd
from functools import reduce 
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.io import loadmat 
from sklearn.decomposition import PCA 
from umap import UMAP


## DATA PREP ##

class LoadData:
    @staticmethod 
    def csv(csv_file,header=0):
        return pd.read_csv(csv_file,header=header)

    @staticmethod
    def mat(mat_file):
        return loadmat(mat_file)

    @staticmethod
    def nii(nii_file):
        return nib.load(nii_file).get_fdata()
    
    
## DIMENSIONALITY REDUCTION ##

class ReduceDims:
    def __init__(self,method='PCA',n_components=2):
        self.n_components = n_components
        self.algorithm = self._initialize_algorithm(method)

    def _initialize_algorithm(self,method):
        if method == 'PCA':
            return PCA(n_components=self.n_components)
        elif method == 'UMAP':
            return UMAP(n_components=self.n_components,random_state=42)
        
    def get_components(self,data):
        return self.algorithm.fit_transform(data)


## MARKOV CHAINS ##
    
class MarkovAnalyze:
    def __init__(self):
        self.dict = {}

    def test_ergodicity(self):
        for (id,input) in self.dict.items():
            data = input['t']
            G = nx.DiGraph()

            for i in data.index:
                G.add_node(i)

                for j in data.columns:
                    w = data.at[i,j]

                    if w > 0:
                        transformed_w = -np.log(w)
                        G.add_edge(i,j,weight=transformed_w)

            irreducibility = nx.is_strongly_connected(G)
            aperiodicity = nx.is_aperiodic(G)

            self.dict[id]['irreducible'] = irreducibility
            self.dict[id]['aperiodic'] = aperiodicity
                

## VISUALIZATION ##
    
class Visualize:
    def __init__(self,figsize=(15,5),fontsize=20,labelpad=20):
        self.figsize = figsize
        self.fontsize = fontsize
        self.labelpad = labelpad

    def _plot(self,ax,X,Y,title):
        ax.set_xlabel(X,fontsize=self.fontsize,labelpad=self.labelpad)
        ax.set_ylabel(Y,fontsize=self.fontsize,labelpad=self.labelpad)
        ax.set_title(title,fontsize=self.fontsize)
        plt.show()

    def heat_map(self,data,X,Y,hue,title):
        array = data.drop([hue],axis=1).to_numpy()
        vmin = array.mean() - array.std()
        vmax = array.mean() + array.std()

        fig,ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(ax=ax,data=data,hue=hue,cmap='viridis',vmin=vmin,vmax=vmax,mask=None)
        self._plot(ax,X,Y,title)

    def scatter_plot(self,data,X,Y,hue,title):
        fig,ax = plt.subplots(figsize=self.figsize)
        sns.scatterplot(ax=ax,data=data,x=X,y=Y,hue=hue,palette='tab10')
        self._plot(ax,X,Y,title)
