## MODULES ##

import torch 
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms 

## TRANSFORMS ##

class NormalizeTensor:
    def __call__(self,data):
        return (data-data.mean())/data.std()
    
class DfToTensor:
    def __call__(self,data):
        return torch.tensor(data.to_numpy(),dtype=torch.float32)
    

## DATASETS ##

class BaseLoader(Dataset):
    def __init__(self,data,transforms=None):
        super().__init__()
        self.data = data
        self.transforms = transforms 
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        sample = self.data[idx]
        if self.transforms:
            sample = self.transforms(sample)

        return sample 
    
class DfLoader(BaseLoader):
    def __getitem__(self,idx):
        data = self.data.iloc[idx,:-1]
        labels = self.data.iloc[idx,-1]

        if self.transforms:
            transformed_data = self.transforms(data)

        return data,labels  

## MODELS ## 


    