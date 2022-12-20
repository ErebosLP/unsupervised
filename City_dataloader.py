# -*- coding: utf-8 -*-
"""
Created on Sat May 21 10:17:03 2022

@author: Jean-
"""
# Packages
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
from scipy.io import loadmat
import torch
import glob

class CityscapeDataset(object):

    def __init__(self,root_img, root_mask ,  subset, transform):
        
        self.subset = subset
        self.root_img = root_img
        self.root_mask = root_mask
        self.img_paths = glob.glob(root_img + subset + '/*/*_leftImg8bit.png')
        self.img_paths.sort()
        self.mask_paths = glob.glob(root_mask + 'gtfine/' + subset + '/*/*_gtFine_instanceIds.png')
        self.mask_paths.sort()
        self.transform = transform

        transforms = []
        transforms.append(T.ToTensor())
        self.transforms_in = T.Compose(transforms)

        print('num_images: ',len(self.img_paths))
        print('num_masks: ',len(self.mask_paths))
   
    def __len__(self):
       return len(self.img_paths)
   
    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])
        img = T.Resize((244,244))(img) 
        image_mask = Image.open(self.mask_paths[index])
        image_mask = T.Resize((244,244),interpolation = T.InterpolationMode.NEAREST)(image_mask) 

        # Compute IMage id
        image_id = torch.tensor([index])
        
        # Compute masks
        obj_ids = np.unique(image_mask)
        obj_ids = obj_ids[1:]  #cut Background
        masks = []
        labels = np.zeros(np.sum(obj_ids>=1000),dtype=int)
        boxes = np.zeros((np.sum(obj_ids>=1000),4))
        j = 0
        for i in obj_ids:
            if i > 1000:
                maske = np.array(image_mask) == i
                #masks.append(maske)
                pos = np.where(maske)
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                
                if (xmax - xmin) * (ymax -ymin) > 0:
                    
                    boxes[j,:] = [xmin, ymin, xmax, ymax]
                    class_id = int(str(i)[:2])-23
                    labels[j] = class_id
                    masks.append(maske) #* class_id)
                    j = j + 1
                else:
                    labels = labels[:-1]
                    boxes = boxes[:-1]

        # Area der Bbox berchenen
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # target values to Tensor
        if len(labels) == 0:
            
            labels = torch.zeros(1, dtype=torch.int64)
            masks = torch.zeros(np.shape(image_mask), dtype=torch.uint8).unsqueeze(0)
            boxes = torch.as_tensor([0,0,1024,2048],dtype=torch.float32).unsqueeze(0)
            area = torch.as_tensor([1024*2048], dtype=torch.float32)               
        else:
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)  #load if used for training
            boxes = torch.as_tensor(boxes, dtype=torch.float32)   
            area = torch.as_tensor(area, dtype=torch.float32)
        
        # is Crowd füllen
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        # Abspeichern in Dictionarry
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] =  masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return self.transform(img,target)
      
class TwoCropsTransform:

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x, target):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k, target]
