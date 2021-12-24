import torch
import cv2 as cv
from torch.utils.data import Dataset, DataLoader
from config import *
class ZaloDataset(Dataset):
    
    def __init__(self, image_paths, targets, transform=None):
        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        #read the image using the path.
        img = cv.imread(self.image_paths[index], 1)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (SIZE_IMG, SIZE_IMG), interpolation = cv.INTER_AREA)
        
        if self.transform is not None:
            img = self.transform(image=img)['image']
            
        img = img.float()
        #get the label and convert it to 0 to 1
        label = torch.tensor(self.targets[index], dtype=torch.long)
        
        return (img, label)

class ZaloTestset(Dataset):
    
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        #read the image using the path.
        img = cv.imread(self.image_paths[index], 1)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (SIZE_IMG, SIZE_IMG), interpolation = cv.INTER_AREA)
        
        if self.transform is not None:
            img = self.transform(image=img)['image']
            
        img = img.float()
            
        #get the dense features.
        
        return img
