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
     transform=None, img_size = (1408,768)):
        self.root = osp.join(root, image_set)
        self.transform = transform
        self.bboxes_df = bboxes_df
        self.fileame_df = fileame_df
        self.H, self.W = img_size

    def __getitem__(self, index):
        img_id = self.fileame_df.iloc[index,0]
        bboxes = self.bboxes_df[self.bboxes_df.image_id == img_id]

        img = cv2.imread(osp.join(self.root, img_id))
        H_i,W_i,_ = img.shape
        H = 896*2
        W = 896
        f_u = max(W_i/W,1)
        f_d = min(H_i/H,1)
        f = f_u*f_d
        H_n = int(H_i*f)
        W_n = int(W_i*f)
        H_n = H_n - H_n%128
        W_n = W_n - W_n%128
        img = cv2.resize(img,(W_n,H_n))
        
        bbox = bboxes.iloc[:,1:5].values
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