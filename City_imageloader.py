# -*- coding: utf-8 -*-
"""
Created on Sat May 21 10:17:03 2022

@author: Jean-
"""
# Packages
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import glob
from PIL import Image  


class CityscapeDataset(object):
    '''
    load_shapes()
    load_image()
    load_mask()
    '''

    def __init__(self,root ,transform,num_imgs = np.inf):
        
        self.root = root
        self.img_paths = glob.glob(root + '/*.jpg')
        self.img_paths.sort()
        print('num_images: ',len(self.img_paths))
        self.transform = transform
        if (num_imgs < len(self.img_paths)):
            self.length = num_imgs
        else:
            self.length = len(self.img_paths)
    
    def __len__(self):
       return self.length
   
    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])
        img = T.Resize((256,256))(img) 

            
           
        return self.transform(img)

class TwoCropsTransform:

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]
