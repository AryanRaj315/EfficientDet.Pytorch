import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import glob
import random

def addSkullandPelvis(image, bbox,skull, pelvis):
    skl = cv2.imread(random.choice(skull))
    plv = cv2.imread(random.choice(pelvis))
    h,w,_ = image.shape
    # Resizing the skull and Pelvis for vertical concatenation
    skl = cv2.resize(skl, (image.shape[1], skl.shape[0]))
    height = skl.shape[1]
    sec = int(2*height/3)
    skl = skl[sec:-50, :, :]
    plv = cv2.resize(plv, (image.shape[1], plv.shape[0]))
    # vertically concat the three images
    img = cv2.vconcat([skl, image, plv])
    bbox[:, 1] = bbox[:, 1] + skl.shape[0]
    bbox[:, 3] = bbox[:, 3] + skl.shape[0]
    return img, bbox

class SPINEDetection(data.Dataset):

    def __init__(self, root, bboxes_df, fileame_df, image_set='training',
     transform=None, img_size = (1920,640)):
        self.root = osp.join(root, image_set)
        self.transform = transform
        self.bboxes_df = bboxes_df
        self.fileame_df = fileame_df
        self.H, self.W = img_size
        self.image_set = image_set
        Images = glob.glob('../Images/*.*')
        self.skull = []
        self.pelvis = []
        for i in range(len(Images)):
            name = Images[i].split('/')[2][:5]
            if name == 'Skull':
                self.skull.append(Images[i])
            else:
                self.pelvis.append(Images[i])
#         print(self.skull)

    def __getitem__(self, index):
        img_id = self.fileame_df.iloc[index,0]
        bboxes = self.bboxes_df[self.bboxes_df.image_id == img_id]

        img = cv2.imread(osp.join(self.root, img_id))
        bbox = bboxes.iloc[:,1:5].values
        if random.randrange(1,10)>5:
            img, bbox = addSkullandPelvis(img, bbox, self.skull, self.pelvis)
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

    def __getitem__(self, index):
        img = cv2.imread(osp.join(self.root, '01-July-2019-'+str(index+1)+'.jpg'))

        if self.transform is not None:
            annotation = {'image': img}
            augmentation = self.transform(**annotation)
            img = augmentation['image']
            
        return img 


    def __len__(self):
        return 98