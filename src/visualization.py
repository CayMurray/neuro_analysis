## MODULES ##

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

    def _plot(self,ax,X,Y,title):
        ax.set_xlabel(X,fontsize=self.fontsize,labelpad=self.labelpad)
        ax.set_ylabel(Y,fontsize=self.fontsize,labelpad=self.labelpad)
        ax.set_title(title,fontsize=self.fontsize,pad=self.labelpad)
        plt.show()

    def heat_map(self,title,data,X,Y,hue,xticks=None,yticks=None):
        array = data.drop([hue],axis=1).to_numpy()
        vmin = array.mean() - array.std()
        vmax = array.mean() + array.std()
        fig,ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(ax=ax,data=data,cmap='viridis',xticklabels=xticks,yticklabels=yticks,vmin=vmin,vmax=vmax,mask=None)
        self._plot(ax,X,Y,title)

    def scatter_plot(self,title,data,X,Y,hue=None,n_colours=1):
        fig,ax = plt.subplots(figsize=self.figsize)
        colormap = plt.cm.viridis
        color_positions = np.linspace(0, 1, n_colours)
        colour_list = [to_hex(colormap(position)) for position in color_positions]
        sns.scatterplot(ax=ax,data=data,x=X,y=Y,hue=hue,palette=colour_list)
        self._plot(ax,X,Y,title)
        fig.savefig(f'../{title}.png',dpi=300)
