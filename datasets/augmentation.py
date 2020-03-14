import albumentations as albu
from albumentations.pytorch.transforms import ToTensor
import torch
import numpy as np
import cv2


def get_augumentation(phase, width=512, height=1536, min_area=0., min_visibility=0.):
    list_transforms = []
#     list_transforms.extend([albu.Resize(height,width)])
    if phase == 'train':
        list_transforms.extend([
            albu.OneOf([
                albu.RandomBrightnessContrast(brightness_limit=0.5,
                                              contrast_limit=0.4),
                albu.RandomGamma(gamma_limit=(50, 150)),
            ]),
            albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=25, p=.5),
            albu.GaussianBlur(),
            albu.GaussNoise(),
            albu.HueSaturationValue(),
            albu.HorizontalFlip(p=0.5),
        ])
    list_transforms.extend([
        albu.Normalize(mean=(0.485, 0.456, 0.406),
                       std=(0.229, 0.224, 0.225), p=1),
        ToTensor()
    ])
    if(phase == 'test'):
        return albu.Compose(list_transforms)
    return albu.Compose(list_transforms, bbox_params=albu.BboxParams(format='pascal_voc', min_area=min_area,
                                                                     min_visibility=min_visibility, label_fields=['category_id']))


def detection_collate(batch):
    imgs = [s['image'] for s in batch]
    annots = [s['bboxes'] for s in batch]
    labels = [s['category_id'] for s in batch]
    max_num_annots = max(len(annot) for annot in annots)
    annot_padded = np.ones((len(annots), max_num_annots, 5))*-1

    if max_num_annots > 0:
        for idx, (annot, lab) in enumerate(zip(annots, labels)):
            if len(annot) > 0:
                annot_padded[idx, :len(annot), :4] = torch.Tensor(annot)
                annot_padded[idx, :len(annot), 4] = torch.Tensor(lab)
    return (torch.stack(imgs, 0), torch.FloatTensor(annot_padded))