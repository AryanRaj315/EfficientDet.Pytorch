import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np


class SPINEDetection(data.Dataset):

    def __init__(self, root, bboxes_df, corners_df, fileame_df, image_set='training',
     transform=None, img_size = (1536,512), downsample = True):
        self.root = osp.join(root, image_set)
        self.transform = transform
        self.bboxes_df = bboxes_df
        self.corners_df = corners_df.clip(0.,0.999)
        self.fileame_df = fileame_df
        self.H, self.W = img_size
        self.downsample = downsample

    def __getitem__(self, index):
        img_id = self.fileame_df.iloc[index,0]
        bboxes = self.bboxes_df[self.bboxes_df.image_id == img_id]
        corners = self.corners_df.iloc[index,:]
        img = cv2.imread(osp.join(self.root, img_id))
        bbox = bboxes.iloc[:,1:5].values
        labels = bboxes.iloc[:,5].values
        if(self.downsample):
            (img, bbox) = self.downsample_it(img, bbox)        
        H,W,_ = img.shape
        corner_arr = []
        for i in range(17):
            x1,x2,x3,x4 = np.round((W)*corners.iloc[4*i:4*(i+1)]).astype(int)
            y1,y2,y3,y4 = np.round((H)*corners.iloc[68+4*i:68+4*(i+1)]).astype(int)
            corner_arr.extend([(x1,y1),(x2,y2),(x3,y3),(x4,y4)])
#         print(len(corner_arr))
        if self.transform is not None:
            annotation = {'image': img, 'bboxes': bbox, 'category_id': labels,'keypoints':corner_arr}
            augmentation = self.transform(**annotation)
            img = augmentation['image']
            bbox = augmentation['bboxes']
            labels = augmentation['category_id']
            corner_arr = augmentation['keypoints']
#         print(corner_arr[0:4])
        aug_corners = []
        for i in range(17):
            [(x1,y1),(x2,y2),(x3,y3),(x4,y4)] = corner_arr[4*i:4*i+4]
            aug_corners.append(np.round(np.array([x1,x2,x3,x4,y1,y2,y3,y4])).astype(int))
        aug_corners_ = np.array(aug_corners).astype(int)
#         print(aug_corners)
        return {'image': img, 'bboxes': bbox, 'corners': aug_corners_, 'category_id': labels} 

    def downsample_it(self, img, bbox):
        H_i,W_i,_ = img.shape
        H_n = H_i
        W_n = W_i
        H_max = self.H
        W_max = self.W
        ar = H_max/W_max
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
           
        H_n = int(128*round(H_n/128))
        W_n = int(128*round(W_n/128))
        
        img = cv2.resize(img,(W_n,H_n))
        bbox[:,1]*= H_n/H_i
        bbox[:,3]*= H_n/H_i
        bbox[:,0]*= W_n/W_i
        bbox[:,2]*= W_n/W_i
        bbox = bbox.astype(int)
        return (img, bbox)
    
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