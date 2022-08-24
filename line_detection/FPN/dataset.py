import os
import glob
import cv2
import sys

import numpy as np

from torch.utils.data import Dataset, DataLoader

from utils import letterbox


IMG_PREFIX = ['jpg', 'png', 'jpeg']


class IDCardSegmentDataset(Dataset):
    def __init__(self, data_dir, img_shape=(500, 300), ignores_list=['background',], augmentation=None, preprocessing=None):
        self.images_dir = os.path.join(data_dir, 'images')
        self.masks_dir = os.path.join(data_dir, 'SegmentationClass')

        self.images_fps = glob.glob(f'{self.images_dir}/*.*')
        self.images_fps = [f for f in self.images_fps if f.split('.')[-1].lower() in IMG_PREFIX]
        self.images_fps.sort()

        self.masks_fps = glob.glob(f'{self.masks_dir}/*.*')
        self.masks_fps = [f for f in self.masks_fps if f.split('.')[-1].lower() in IMG_PREFIX]
        self.masks_fps.sort()

        self.colors_dict = {}
        with open(os.path.join(data_dir, 'labelmap.txt'), 'r') as fr:
            lines = fr.readlines()
            for line in lines[1:]:
                feat_infos = line.split(':')
                feat_name = feat_infos[0]
                if feat_name in ignores_list:
                    continue
                feat_color = feat_infos[1].split(',')
                feat_color = [int(ch_color) for ch_color in feat_color]
                assert len(feat_color) == 3
                self.colors_dict[feat_name] = feat_color
        
        self.classes = list(self.colors_dict.keys())
        self.class_values = list(range(len(self.classes)))

        self.img_shape = img_shape
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        # Read data
        image = cv2.imread(self.images_fps[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read masks data
        mask = cv2.imread(self.masks_fps[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        masks = [(mask == feat_color).prod(axis=-1) for _, feat_color in self.colors_dict.items()]
        
        mask = np.stack(masks, axis=-1).astype('float')
        # for i, (feat_name, _) in enumerate(self.colors_dict.items()):
        #     m = mask[..., i]
        #     m = np.stack([m * 255]*3, axis=-1).astype(np.uint8)
        #     cv2.imwrite(f'{feat_name}.jpg', m)
        # print(mask.shape)
        # exit(0)

        assert image.shape[:2] == mask.shape[:2]
        image = cv2.resize(image, self.img_shape)
        mask = cv2.resize(mask, self.img_shape)

        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.images_fps)