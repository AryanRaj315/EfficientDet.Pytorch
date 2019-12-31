import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np

IMAGES = 'images'
ANNOTATIONS = 'annotations'
INSTANCES_SET = 'instances_{}.json'
_CLASSES = ('vertebrae')

class SPINEDetection(data.Dataset):

    def __init__(self, root, bboxes_df, corners_df, fileame_df, image_set='training',
     transform=None, dataset_name=''):
        # sys.path.append(osp.join(root, COCO_API))
        # from pycocotools.coco import COCO
        self.root = osp.join(root, IMAGES, image_set)
        # self.coco = COCO(osp.join(root, ANNOTATIONS,
        #                           INSTANCES_SET.format(image_set)))
        # self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.name = dataset_name
        self.bboxes_df = bboxes_df
        self.corners_df = corners_df
        self.fileame_df = fileame_df

    def __getitem__(self, index):
        img_id = self.fileame_df.iloc[index,0]
        bboxes = self.bboxes_df[self.bboxes_df.image_id == img_id]
        corners = self.corners_df[self.corners_df.image_id == img_id]

        img = cv2.imread(osp.join(self.root, img_id))

        ##########

        # Needs Transformation to model
        
        ##########

        # if self.transform is not None:
        #     annotation = {'image': img, 'bboxes': bbox, 'category_id': labels}
        #     augmentation = self.transform(**annotation)
        #     img = augmentation['image']
        #     bbox = augmentation['bboxes']
        #     labels = augmentation['category_id']

        return {'image': img, 'bboxes': bbox, 'corners': corners, 'category_id': labels} 


    def __len__(self):
        return len(self.fileame_df)