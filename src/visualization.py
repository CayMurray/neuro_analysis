## MODULES ##

import random 
random.seed(10)

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

        self.colours = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', 
           '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', 
           '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', 
           '#000075', '#808080', '#ffffff', '#000000', '#DAA520', '#CD5C5C', 
           '#ADFF2F']
        
        self.shapes = ['o','s','D','X','P']

        self.colours_dict = {}
        self.shapes_dict = {}

    @staticmethod
    def _rgb2hex(n_colours):
        unique_colours = set()

        while len(unique_colours) < n_colours:
            new_colour = to_hex([random.random() for _ in range(3)])
            unique_colours.add(new_colour)

        return unique_colours
    
    @staticmethod
    def _max_min(data,hue):
        array = data.drop([hue],axis=1).to_numpy()
        vmin = array.mean() - array.std()
        vmax = array.mean() + array.std()
        return vmin,vmax
    
    def _update_dicts_for_shapes(self,labels,distinct_shapes=False):
        unique_labels = sorted(set(labels))

        for (i,label) in enumerate(unique_labels):
            self.colours_dict[label] = self.colours[i]
            self.shapes_dict[label] = self.shapes[i] if distinct_shapes else 'o'

    def _plot(self,ax,X,Y,title,fig):
        ax.set_xlabel(X,fontsize=self.fontsize,labelpad=self.labelpad)
        ax.set_ylabel(Y,fontsize=self.fontsize,labelpad=self.labelpad)
        ax.set_title(title,fontsize=self.fontsize,pad=self.labelpad)
        plt.show()
        fig.savefig(f'{title}.png',transparent=True,dpi=300)

    def heat_map(self,title,data,hue='labels',normalize=False,xticks=None,yticks=None,annot=True):
        vmin,vmax = self._max_min(data,hue) if normalize else (None,None)
        fig,ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(ax=ax,data=data,cmap='viridis',xticklabels=xticks,yticklabels=yticks,vmin=vmin,vmax=vmax,annot=annot,annot_kws={"size": 18})
        self._plot(ax,'Predicted','True',title,fig)

    def scatter_plot(self,title,data,hue=None):
        fig,ax = plt.subplots(figsize=self.figsize)
        n_colours = len(set(data[hue]))
        self._update_dicts_for_shapes(data[hue])
        sns.scatterplot(ax=ax,data=data,x=data.columns[0],y=data.columns[1],hue=hue,style=hue,palette=self.colours_dict,markers=self.shapes_dict)
        self._plot(ax,data.columns[0],data.columns[1],title,fig)

    def bar_chart(self,title,ks=None,**kwargs):
        fig,ax = plt.subplots(figsize=(15,10))
        colour_list = [to_hex([random.random() for _ in range(3)]) for _ in range(len(kwargs))]

        for (i,(label,dist)) in enumerate(kwargs.items()):
            sns.histplot(ax=ax,data=dist,color=colour_list[i],stat='density',label=label,alpha=1.0)

        ax.set_title(title,fontsize=self.fontsize,pad=self.labelpad)
        ax.set_ylabel('Density',fontsize=self.fontsize,labelpad=self.labelpad)
        ax.set_xlabel('Bins',fontsize=self.fontsize,labelpad=self.labelpad)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16) 

        if ks is not None:
            ax.text(0.01, 0.99, f'KS stat: {ks[0]}, p-value: {ks[1]}', transform=ax.transAxes, verticalalignment='top', fontsize=15, color='black')

        plt.legend(frameon=False)
        plt.show()

        fig.savefig(f'{title}.png',transparent=True,dpi=300)

        


