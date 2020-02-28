import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np


class SPINEDetection(data.Dataset):

    def __init__(self, root, bboxes_df, fileame_df, image_set='training',
     transform=None, img_size = (2048,1024)):
        self.root = osp.join(root, image_set)
        self.transform = transform
        self.bboxes_df = bboxes_df
        self.fileame_df = fileame_df
        self.H, self.W = img_size

    def __getitem__(self, index):
        img_id = self.fileame_df.iloc[index,0]
        bboxes = self.bboxes_df[self.bboxes_df.image_id == img_id]

        img = cv2.imread(osp.join(self.root, img_id))
        bbox = bboxes.iloc[:,1:5].values
        H_i,W_i,_ = img.shape
        H_n = H_i
        W_n = W_i
        H_max = self.H
        W_max = self.W
        ar = 2
        ar_i = H_i/W_i
        if(H_i>H_max and W_i>W_max):
            if(ar_i>ar):
                H_n = H_max
                W_n = H_n/ar_i
            else:
                W_n = W_max
                H_n = W_n*ar_i
                
        elif(H_i>H_max):
            H_n = H_max
            W_n = H_n/ar_i
            
        elif(W_i>W_max):
            W_n = W_max
            H_n = W_n*ar_i
           
        H_n = int(H_n - H_n%128)
        W_n = int(W_n - W_n%128)
        img = cv2.resize(img,(W_n,H_n))
        bbox[:,1]*= H_n/H_i
        bbox[:,3]*= H_n/H_i
        bbox[:,0]*= W_n/W_i
        bbox[:,2]*= W_n/W_i
            
        bbox = bbox.astype(int)   
        
        labels = bboxes.iloc[:,5].values
        if self.transform is not None:
            annotation = {'image': img, 'bboxes': bbox, 'category_id': labels}
            augmentation = self.transform(**annotation)
            img = augmentation['image']
            bbox = augmentation['bboxes']
            labels = augmentation['category_id']


        return {'image': img, 'bboxes': bbox, 'category_id': labels} 


    def __len__(self):
        return len(self.fileame_df)

class SPINEDetection_test(data.Dataset):

    def __init__(self, root = 'test_images', transform = None, img_size = (1408,768)):
        self.root = root
        self.transform = transform
        self.H, self.W = img_size
        path = []
        for img in os.listdir(root):
            if img[-3:] == 'jpg':
                path.append(img)
        self.path = path

    def __getitem__(self, index):
        img_name = self.path[index]
        img = cv2.imread(osp.join(self.root, img_name))

        if self.transform is not None:
            annotation = {'image': img}
            augmentation = self.transform(**annotation)
            img = augmentation['image']
            
        return img 


    def __len__(self):
        return len(self.path)