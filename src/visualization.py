## MODULES ##

import random 
random.seed(42)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex


## BASIC PLOTS ##
    
class BaseVisualizer:
    def __init__(self,figsize=(15,10),fontsize=20,labelpad=20):
        self.figsize = figsize
        self.fontsize = fontsize
        self.labelpad = labelpad

    @staticmethod
    def _rgb2hex(n_colours):
        return [to_hex([random.random() for _ in range(3)]) for _ in range(n_colours)]
    
    @staticmethod
    def _max_min(data,hue):
        array = data.drop([hue],axis=1).to_numpy()
        vmin = array.mean() - array.std()
        vmax = array.mean() + array.std()
        return vmin,vmax

    def _plot(self,ax,X,Y,title,fig):
        ax.set_xlabel(X,fontsize=self.fontsize,labelpad=self.labelpad)
        ax.set_ylabel(Y,fontsize=self.fontsize,labelpad=self.labelpad)
        ax.set_title(title,fontsize=self.fontsize,pad=self.labelpad)
        plt.show()
        fig.savefig(f'{title}.png',dpi=300)

    def heat_map(self,title,data,hue,normalize=True,xticks=None,yticks=None,annot=False):
        vmin,vmax = self._max_min(data,hue) if normalize else (None,None)
        fig,ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(ax=ax,data=data,cmap='viridis',xticklabels=xticks,yticklabels=yticks,vmin=vmin,vmax=vmax,annot=annot)
        self._plot(ax,'Predicted','True',title,fig)

    def scatter_plot(self,title,data,hue=None):
        n_colours = len(set(data[hue]))
        fig,ax = plt.subplots(figsize=self.figsize)
        sns.scatterplot(ax=ax,data=data,x=data.columns[0],y=data.columns[1],hue=hue,palette=self._rgb2hex(n_colours))
        self._plot(ax,data.columns[0],data.columns[1],title,fig)

    def bar_chart(self,title,ks=None,**kwargs):
        fig,ax = plt.subplots(figsize=(15,10))
        colour_list = self._rgb2hex(len(kwargs))

        for (i,(label,dist)) in enumerate(kwargs.items()):
            sns.histplot(ax=ax,data=dist,color=colour_list[i],stat='density',label=label)

        ax.set_title(title,fontsize=self.fontsize,pad=self.labelpad)
        ax.set_ylabel('Density',fontsize=self.fontsize,labelpad=self.labelpad)
        ax.set_xlabel('Bins',fontsize=self.fontsize,labelpad=self.labelpad)

        if ks is not None:
            ax.text(0.01, 0.99, f'KS stat: {ks[0]}, p-value: {ks[1]}', transform=ax.transAxes, verticalalignment='top', fontsize=15, color='black')

        plt.legend()
        plt.show()

        fig.savefig(f'{title}.png',dpi=300)

        


