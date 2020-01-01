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
     transform=None, dataset_name=''):
        self.root = osp.join(root, image_set)
        self.transform = transform
        self.name = dataset_name
        self.bboxes_df = bboxes_df
        self.corners_df = corners_df
        self.fileame_df = fileame_df

    def __getitem__(self, index):
        img_id = self.fileame_df.iloc[index,0]
        bboxes = self.bboxes_df[self.bboxes_df.image_id == img_id]
        corners = self.corners_df.iloc[index,:]
        corner_arr = []
        for i in range(17):
            x1,x2,x3,x4 = corners.iloc[4*i:4*(i+1)]
            y1,y2,y3,y4 = corners.iloc[68+4*i:68+4*(i+1)]
            corner_arr.append(np.array([x1,x2,x3,x4,y1,y2,y3,y4]))

        corner_arr = np.array(corner_arr)

        img = cv2.imread(osp.join(self.root, img_id))
        bbox = bboxes.iloc[:,1:5].values
        labels = bboxes.iloc[:,5].values
        if self.transform is not None:
            annotation = {'image': img, 'bboxes': bbox, 'category_id': labels}
            augmentation = self.transform(**annotation)
            img = augmentation['image']
            bbox = augmentation['bboxes']
            labels = augmentation['category_id']


        return {'image': img, 'bboxes': bbox, 'corners': corner_arr, 'category_id': labels} 


    def __len__(self):
        return len(self.fileame_df)