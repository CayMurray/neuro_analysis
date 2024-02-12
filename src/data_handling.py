## MODULES ##

import pandas as pd
import nibabel as nib
from scipy.io import loadmat 
from sklearn.decomposition import PCA 
from umap import UMAP

## LOAD DATA ##

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
