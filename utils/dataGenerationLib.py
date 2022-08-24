import os
import sys
import glob
import cv2
import yaml
import numpy as np
import xmltodict

import torch

sys.path.append('line_detection/FPN')
sys.path.append('keypoint_detection/scrfd')
sys.path.append('informationExtractor/vietocr')

from keypoint_detection.scrfd.mmdet.apis import inference_detector, init_detector, show_result_pyplot, result_pyplot_save, result_pyplot_bboxes_kps_save
from line_detection.FPN.utils import load_model as load_segmentation_model
from line_detection.FPN.utils import get_class_infos, get_min_max_x_y
from line_detection.FPN.augmentation import get_augmentation, get_preprocessing_fn, get_preprocessing
from visualize_v1 import kpts_image_save

IMG_PREFIX = ['jpg', 'jpeg', 'png']


class OCRDatasetGenerator:
    def __init__(self, hyps_file='utils/dataGenerationHyps/OCRHyps.yaml'):
        
        with open(hyps_file, errors='ignore') as f:
            hyps_dict = yaml.safe_load(f)
        # Normal options
        self.device = 'cuda' if not hyps_dict['no_gpu'] else 'cpu'
        self.save_dir = hyps_dict['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_generated_card = hyps_dict['save_generated_card']
        self.delay = hyps_dict['delay']

        # Keypoints options
        kpts_config = os.path.join(hyps_dict['kpts_checkpoint'], 'config.py')
        kpts_weights = os.path.join(hyps_dict['kpts_checkpoint'], 'best.pth')

        self.kpts_model = init_detector(kpts_config, kpts_weights, self.device)
        self.kpts_thr = hyps_dict['kpts_thr']
        self.kpts_out_width = hyps_dict['kpts_out_width']
        self.kpts_out_height = hyps_dict['kpts_out_height']

        # Line options
        line_weights = os.path.join(hyps_dict['line_checkpoint'], 'best_model.pth')
        self.colors_dict = get_class_infos(hyps_dict['labelmap'], hyps_dict['ignores_list'])
        self.colormap = np.arange(len(self.colors_dict))[:, None] + 1
        self.converted_color_map = np.repeat(np.repeat(self.colormap[:, :, np.newaxis, np.newaxis], 640, axis=2), 640, axis=3)
        self.segmentation_model = load_segmentation_model(line_weights)
        self.segmentation_augmentation = get_augmentation(is_test=True, img_shape=(640, 640))
        self.segmentation_preprocessing = get_preprocessing(get_preprocessing_fn(hyps_dict['line_model_bacbone'], hyps_dict['line_model_encoder_pretrained']))

    def scan_from_dir(self, img_root, data_type='train'):
        """
        Call the function when you want scan card id from images root

        Args:
            - img_root                  : images root
            - data_type                 : dataset surfix dir
            - return_kpts_coords        : Return keypoints coordinates information for 4 id corners if 'True' else return None.
                                        Default: 'False'
            - return_segment_coords     : Return segmentation map for each information of id card if 'True' else return None.
                                        Default: 'False'
            - return_card_information   : Return card information (id_identity, name, birthday, countryside, address) of id card if 'True' else return None.
                                        Default: 'True'
        """
        save_dir = os.path.join(self.save_dir, data_type)
        os.makedirs(save_dir, exist_ok=True)

        assert os.path.isdir(img_root)
        img_dir = os.path.join(img_root, 'images')
        img_annotations_file = os.path.join(img_root, 'annotations.xml')
        img_annotations = self.get_img_informtations(img_annotations_file)
        

        for img_name, img_infos in img_annotations.items():
            name_save = img_name.split('.')[0]
            img_f = os.path.join(img_dir, img_name)
            img = cv2.imread(img_f)

            ### Kpts detection stage
            _, kps_result = inference_detector(self.kpts_model, img)

            if len(kps_result[0]) == 0:
                print(f'{name_save}: ID Card not found')
                continue
            kps_res = kps_result[0][0]
            kps_conf = kps_res[-1]
            if kps_conf < self.kpts_thr:
                print(f'{name_save}: ID Card not found')
                continue
            
            pts_src = kps_res[:-1].reshape(4, 2)
            
            pts_dst = np.float32([[0, 0], [self.kpts_out_width, 0],
                            [self.kpts_out_width, self.kpts_out_height], [0, self.kpts_out_height]])
            M = cv2.getPerspectiveTransform(pts_src, pts_dst)
            generated_card = cv2.warpPerspective(img, M, (self.kpts_out_width, self.kpts_out_height))

            self.information_extraction(generated_card, img_infos, name_save, data_type)

    def get_img_informtations(self, img_annotations_file):
        with open(img_annotations_file) as f:
            data_raw = xmltodict.parse(f.read())
            f.close()
        # print(data_raw)

        data_dict = {}

        for raw_info in data_raw['annotations']['image']:
            img_name = raw_info['@name']
            data_dict[img_name] = {}
            for feat in raw_info['polygon']:
                data_dict[img_name][feat['@label']] = feat['attribute']['#text']
        return data_dict

    def information_extraction(self, card_image, img_infos, image_name, data_type):
        save_dir = os.path.join(self.save_dir, data_type)
        save_generated_card_dir = os.path.join(self.save_dir, data_type + '_generated_card')
        os.makedirs(save_generated_card_dir, exist_ok=True)

        txt_file = open(os.path.join(self.save_dir, data_type + '_annotation.txt'), 'a')

        ### segmentation stage
        
        segment_coords = {}
        ocr_info = {}
        for feat_name in ['identity_number', 'name', 'birthday', 'countryside', 'address']:
            if feat_name in list(self.colors_dict.keys()):
                segment_coords[feat_name] = None
                ocr_info[feat_name] = ''

        # ori_img = cv2.cvtColor(generated_card.copy(), cv2.COLOR_RGB2BGR)
        ori_img = card_image.copy()
        ori_img_overlay = ori_img.copy()

        padimg = self.segmentation_augmentation(image=ori_img)['image']
        padimg_overlay = padimg.copy()
        
        pad_w = (padimg.shape[1] - ori_img.shape[1]) // 2 + 1
        pad_h = (padimg.shape[0] - ori_img.shape[0]) // 2 + 1

        augimg = self.segmentation_preprocessing(image=padimg)['image']

        x_tensor = torch.from_numpy(augimg).to(self.device).unsqueeze(0)

        pr_mask = self.segmentation_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        
        pr_mask = np.repeat(pr_mask[:, np.newaxis, :, :], 3, axis=1) * self.converted_color_map
        pr_mask = np.sum(pr_mask, axis=0).transpose((1, 2, 0)).astype(np.uint8)


        for i, (feat_name, color) in enumerate(self.colors_dict.items(), start=1):
            if feat_name not in list(img_infos.keys()):
                continue

            if img_infos[feat_name].strip().lower() == 'nan':
                continue

            indices = np.where(np.all(pr_mask == i, axis=-1))
            coords = np.array(list(zip(indices[0], indices[1])))
            # print(coords)

            if len(coords) == 0:
                continue

            segment_coords[feat_name] = coords

            min_x, min_y, max_x, max_y = get_min_max_x_y(coords)
            width = max_x - min_x + 1
            height = max_y - min_y + 1
            white_area = np.ones((height, width, 3), dtype=np.uint8) * 127
            white_area[coords[:, 0] - min_y, coords[:, 1] - min_x] = padimg[coords[:, 0], coords[:, 1]]
            
            cv2.imwrite(os.path.join(save_dir, f'{image_name}_{feat_name}.jpg'), white_area)
            txt_file.writelines(os.path.join(data_type, f'{image_name}_{feat_name}.jpg') +'\t' + img_infos[feat_name] + '\n')
            # ocr_info[feat_name] = self.ocr_model.predict(white_area)

            mask_map_aug = np.zeros_like(padimg)
            mask_map_aug[coords[:, 0], coords[:, 1]] = color
            padimg_overlay = cv2.addWeighted(src1=padimg_overlay, alpha=0.99,
                                    src2=mask_map_aug, beta=0.33, gamma=0.0)
            
            mask_map_ori = np.zeros_like(ori_img)
            mask_map_ori[coords[:, 0] - pad_h, coords[:, 1] - pad_w] = color
            ori_img_overlay = cv2.addWeighted(src1=ori_img_overlay, alpha=0.99,
                                    src2=mask_map_ori, beta=0.33, gamma=0.0)

        if save_generated_card_dir and image_name:
            cv2.imwrite(os.path.join(save_generated_card_dir, f'{image_name}.jpg'), ori_img_overlay)