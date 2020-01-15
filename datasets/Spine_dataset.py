import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import albumentations as albu

def horizontal_flip(img,bboxs,corners,labels):
    hflip_corners = []
    for corner in corners:
        hflip_corner = np.array([corner[1],corner[0],corner[3],corner[2],corner[5],corner[4],corner[7],corner[6]])
        hflip_corners.append(hflip_corner)
        
    list_transforms = [albu.HorizontalFlip(p=1.0)]
    hflip = albu.Compose(list_transforms, bbox_params=albu.BboxParams(format='pascal_voc', min_area=0, min_visibility=0, label_fields=['category_id']))
    annotation = {'image': img, 'bboxes': bboxs, 'category_id': labels}
    augmentation = hflip(**annotation)
    img = augmentation['image']
    bbox = augmentation['bboxes']
    labels = augmentation['category_id']
    return img,bbox,np.array(hflip_corners),labels

class SPINEDetection(data.Dataset):

    def __init__(self, root, bboxes_df, corners_df, fileame_df, image_set='training',
     transform=None, img_size = (1408,768)):
        self.root = osp.join(root, image_set)
        self.transform = transform
        self.bboxes_df = bboxes_df
        self.corners_df = corners_df
        self.fileame_df = fileame_df
        self.H, self.W = img_size

    def __getitem__(self, index):
        img_id = self.fileame_df.iloc[index,0]
        bboxes = self.bboxes_df[self.bboxes_df.image_id == img_id]
        corners = self.corners_df.iloc[index,:]

        img = cv2.imread(osp.join(self.root, img_id))
        H,W,_ = img.shape
        
        corner_arr = []
        for i in range(17):
            x1,x2,x3,x4 = np.round((self.W)*corners.iloc[4*i:4*(i+1)])
            y1,y2,y3,y4 = np.round((self.H)*corners.iloc[68+4*i:68+4*(i+1)])
            corner_arr.append(np.array([x1,x2,x3,x4,y1,y2,y3,y4]))
        
        corner_arr = np.array(corner_arr)
        bbox = bboxes.iloc[:,1:5].values
        labels = bboxes.iloc[:,5].values
        
        if(np.random.rand(1)[0]>0.5):
            img, bbox, corner_arr, labels = horizontal_flip(img,bbox,corner_arr,labels)
        
        if self.transform is not None:
            annotation = {'image': img, 'bboxes': bbox, 'category_id': labels}
            augmentation = self.transform(**annotation)
            img = augmentation['image']
            bbox = augmentation['bboxes']
            labels = augmentation['category_id']


        return {'image': img, 'bboxes': bbox, 'corners': corner_arr, 'category_id': labels} 


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