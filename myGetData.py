# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 16:18:56 2022

@author: sammy
"""

#load data
import torchvision
from torch.utils.data import Dataset
from myTransform import SimCLRTransform

class GetDataSet(Dataset):
    def __init__(self,file,file_paths):
        self.file_paths = file_paths #file names of images
        self.transforms = SimCLRTransform(image_size=256)
        self.num_samples = len(self.file_paths)
        
    def __getitem__(self,idx):
        file_path = self.file_paths[idx]
        img = torchvision.io.read_image(file_path,torchvision.io.ImageReadMode.RGB)
        img1,img2,img3 = self.transforms(img,img)
        return img1,img2,img3
        
    def __len__(self):
        return self.num_samples
    
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class GetTestDataSet(Dataset):
    def __init__(self,file_paths,folder_path,kaggle = False):
        self.file_paths = file_paths #file names of images
        self.folder_path = folder_path
        self.kaggle = kaggle
        self.transforms = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((256,256)),
                            transforms.ToTensor()
                            ])
        self.num_samples = len(self.file_paths)
        
    def __getitem__(self,idx):
        file_path = self.file_paths[idx] 
        if (self.kaggle):
            file_path1 = os.path.join(os.path.join(self.folder_path,'test'), file_path[0]+'.png')
            file_path2 = os.path.join(os.path.join(self.folder_path,'test'), file_path[1]+'.png')
        else:
            file_path1 = os.path.join(os.path.join(self.folder_path,'train'), file_path[0]+'.png')
            file_path2 = os.path.join(os.path.join(self.folder_path,'train'), file_path[1]+'.png')
        
        img1 = torchvision.io.read_image(file_path1,torchvision.io.ImageReadMode.RGB)
        img1 = self.transforms(img1)
        img2 = torchvision.io.read_image(file_path2,torchvision.io.ImageReadMode.RGB)
        img2 = self.transforms(img2)
        return img1,img2
        
    def __len__(self):
        return self.num_samples