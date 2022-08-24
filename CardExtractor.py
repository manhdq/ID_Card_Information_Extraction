import glob
import os
from pydoc import ispath
import cv2
import sys
import argparse
import time
import yaml
import numpy as np
import torch
# import mmcv
import matplotlib.pyplot as plt

from utils.general import count_parameters

# from viet_address_correction.address_correction import AddressCorrection
from keypoint_detection.scrfd.helper.detection.apis import inference_detector, inference_batch_detector, init_detector, show_result_pyplot, \
                                                        result_pyplot_save, result_pyplot_bboxes_kps_save
from line_detection.FPN.utils import load_model as load_segmentation_model
from line_detection.FPN.utils import get_class_infos, get_min_max_x_y
from line_detection.FPN.augmentation import get_augmentation, get_preprocessing_fn, get_preprocessing
from vietocr.ocr_common import OCRCommon
from utils.visualize import kpts_image_save

IMG_EXT = ['jpg', 'png']


class CardExtractor:
    def __init__(self,
                hyps_file='hyps/cmt_default_hyps.yaml'
        ):
        """
        ID Card Informtation Extraction Package.
        
        Args:
            - hyps_file: hyperparameters file for the package.
                        You can modified hyps or create a new one for the package in folder hyps.
        """
        with open(hyps_file, errors='ignore') as f:
            hyps_dict = yaml.safe_load(f)
        # Normal options
        self.device = 'cuda:0' if not hyps_dict['no_gpu'] else 'cpu'
        self.save_dir = hyps_dict['save_dir']
        self.delay = hyps_dict['delay']
        self.return_kpts_coords = hyps_dict['return_kpts_coords']
        self.return_segment_coords = hyps_dict['return_segment_coords']
        self.return_card_information = hyps_dict['return_card_information']
        self.print_infos = hyps_dict['print_infos']
        self.quantity = hyps_dict['quantity']
        self.batch_size = hyps_dict['batch_size']

        # Keypoints options
        self.use_kpts_model = hyps_dict['use_kpts_model']
        if self.use_kpts_model:
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
        self.segmentation_model = load_segmentation_model(line_weights, map_location=self.device)
        self.segmentation_augmentation = get_augmentation(is_test=True, img_shape=(640, 640))
        self.segmentation_preprocessing = get_preprocessing(get_preprocessing_fn(hyps_dict['line_model_bacbone'], hyps_dict['line_model_encoder_pretrained']))

        # OCR options
        self.ocr_config_name = hyps_dict['ocr_config_name']
        self.ocr_weights = hyps_dict['ocr_weights']
        self.ocr_model = OCRCommon(self.ocr_config_name, self.ocr_weights, self.device)

        print(f'- Keypoints model parameters: {count_parameters(self.kpts_model)}')
        print(f'- Line model parameters: {count_parameters(self.segmentation_model)}')
        print(f'- OCR model parameters: {count_parameters(self.ocr_model.detector.model)}')

    def scan_from_img(self, data):
        """
        Call the function when you want scan card id from image

        Args:
            - img                       :(str | np.adarray) numpy image or image file of the image
            - name_save                 : file save name for id card information extracted from cam.
                                        If `name_save is None`, the information will not be saved.
            - use_kpts_model            : Since kpts model not so good and if the card very near the camera lens,
                                        You can set this option `False` to not use kpts model, otherwise set `True`
            - return_kpts_coords        : Return keypoints coordinates information for 4 id corners if 'True' else return None.
                                        Default: 'False'
            - return_segment_coords     : Return segmentation map for each information of id card if 'True' else return None.
                                        Default: 'False'
            - return_card_information   : Return card information (id_identity, name, birthday, countryside, address) of id card if 'True' else return None.
                                        Default: 'True'
            - print_infos               : Print information extracted from the id card

        Returns:
            - kpts_coords               : shape(4, 2): coordinates for 4 corners of the card.
                                        (top-left), (top-right), (bottom-left), (bottom-right)
            - segment_coords            : segmentaiton map for each informtation of id card
            - card_information          : dict: Informtation of the card
        """
        ## Model 1: kpts extraction
        # img: list(tensor: (1,3,640,640)), img_metas: list(metas), return_loss: bool, rescale: bool, inference_end: bool, **kwargs
        # list(list(list(bbox(N, 5): np.ndarray), list(kpts(N, 9): np.ndarray)))

        ## Model 2: Line Segmentation
        # img: tensor(1,3,640,640)
        # mask: tensor(1,8,640,640)

        ## Model 3: Text Extraction
        # img: np.ndarray(W, H, 3)
        # info: str

        if isinstance(data, list):
            # Do nothing
            pass
        elif os.path.isfile(data):
            ext = data.split('.')[-1].lower()
            assert ext in IMG_EXT
            data = [data]
        elif os.path.isdir(data):
            data = glob.glob(os.path.join(data, '*.*'))
            data = [img_path for img_path in data if img_path.split('.')[-1].lower() in IMG_EXT]
        elif isinstance(data, np.ndarray):
            data = [data]
        else:
            raise
        if self.quantity != 'all':
            data = data[:self.quantity]
        
        assert len(data)

        card_informations = {}

        while True:
            img_caches = []
            generated_caches = []
            name_save_caches = []

            data_caches = data[:self.batch_size]
            for idx, img_path in enumerate(data_caches):
                if isinstance(img_path, str):
                    name_save = img_path.split(os.sep)[-1].split('.')[0]
                    card_informations[name_save] = {"image_path": img_path, "kpts_coords": None, "segment_coords": None, "card_information": None}
                    img = cv2.imread(img_path)

                    img_caches.append(img)
                    name_save_caches.append(name_save)
                else:
                    name_save = f'img_{idx}'
                    card_informations[name_save] = {"image_path": None, "kpts_coords": None, "segment_coords": None, "card_information": None}
                    img = img_path.copy()

                    img_caches.append(img)
                    name_save_caches.append(name_save)

                card_informations[name_save]['card_information'] = {
                    'identity_number': None,
                    'name': None,
                    'birthday': None,
                    'countryside': None,
                    'address': None,
                    'date_of_expory': None,
                    'nationality': None,
                    'sex': None
                }

            ### Kpts detection stage
            if self.use_kpts_model:
                # _, kps_result = inference_batch_detector(self.kpts_model, data_caches)
                results = inference_batch_detector(self.kpts_model, img_caches)

                for name, img, res in zip(name_save_caches, img_caches, results):

                    if len(res[1][0]) == 0:
                        print(f'{name}: ID Card not found')
                        generated_caches.append(None)
                        continue
                    kps_res = res[1][0][0]
                    kps_conf = kps_res[-1]
                    if kps_conf < self.kpts_thr:
                        print(f'{name}: ID Card not found')
                        generated_caches.append(None)
                        continue
                    
                    if self.save_dir and name_save:
                        inf_save_dir = os.path.join(self.save_dir, name_save)
                        os.makedirs(inf_save_dir, exist_ok=True)
                    pts_src = kps_res[:-1].reshape(4, 2)
                    if self.return_kpts_coords:
                        card_informations[name] = pts_src
                    
                    if self.save_dir and name_save:
                        kpts_image_save(img, pts_src, inf_save_dir)
                    pts_dst = np.float32([[0, 0], [self.kpts_out_width, 0],
                                    [self.kpts_out_width, self.kpts_out_height], [0, self.kpts_out_height]])
                    M = cv2.getPerspectiveTransform(pts_src, pts_dst)

                    generated_card = cv2.warpPerspective(img, M, (self.kpts_out_width, self.kpts_out_height))
                    generated_caches.append(generated_card)
            else:
                for img in img_caches:
                    generated_card = img.copy()
                    generated_card = cv2.resize(generated_card, (self.kpts_out_width, self.kpts_out_height))
                    generated_caches.append(generated_card)

            # segment_coords, card_information = self.information_extraction(generated_card, name_save,
            #                                                             return_segment_coords=self.return_segment_coords,
            #                                                             return_card_information=self.return_card_information,
            #                                                             print_infos=self.print_infos)
            self.information_batch_extraction(card_informations, generated_caches, name_save_caches,
                                            return_segment_coords=self.return_segment_coords,
                                            return_card_information=self.return_card_information,
                                            print_infos=self.print_infos)

            del generated_caches
            del name_save_caches
            del img_caches
            del data_caches

            if len(data) < self.batch_size:
                break
            data = data[self.batch_size:]

        out_card_informations = {}
        for image_name, image_infos in card_informations.items():
            out_card_informations[image_name] = image_infos['card_information']

        return out_card_informations

    def erode_leftovers(self, mask, kernel_size=(3, 3)):
        kernel = np.ones(kernel_size, np.uint8)
        clean_mask = cv2.erode(mask, kernel, iterations=1)
        clean_mask = cv2.dilate(clean_mask, kernel, iterations=1)

        return clean_mask

    def information_batch_extraction(self, card_informations, card_images, image_names, 
                                    return_segment_coords=False, return_card_information=True, print_infos=False):

        ### segmentation and OCR stage
        tensor_caches = []
        # image_name_caches = []
        for image_name, card_image in zip(image_names, card_images):
            if self.save_dir and image_name:
                inf_save_dir = os.path.join(self.save_dir, image_name)
                os.makedirs(inf_save_dir, exist_ok=True)
            
            if card_image is None:
                continue
            
            segment_coords = {}
            ocr_info = {}
            for feat_name in ['identity_number', 'name', 'birthday', 'countryside', 'address']:
                if feat_name in list(self.colors_dict.keys()):
                    segment_coords[feat_name] = None
                    ocr_info[feat_name] = None

            # ori_img = cv2.cvtColor(generated_card.copy(), cv2.COLOR_RGB2BGR)
            ori_img = card_image.copy()
            ori_img_overlay = ori_img.copy()

            padimg = self.segmentation_augmentation(image=ori_img)['image']
            padimg_overlay = padimg.copy()
            
            pad_w = (padimg.shape[1] - ori_img.shape[1]) // 2 + 1
            pad_h = (padimg.shape[0] - ori_img.shape[0]) // 2 + 1

            augimg = self.segmentation_preprocessing(image=padimg)['image']

            x_tensor = torch.from_numpy(augimg).to(self.device)
            tensor_caches.append(x_tensor)
            # image_name_caches.append(image_name)
        x_tensor = torch.stack(tensor_caches)

        pr_masks = self.segmentation_model.predict(x_tensor)
        
        pr_masks = (pr_masks.cpu().numpy().round())
        
        # pr_mask = np.repeat(pr_mask[:, np.newaxis, :, :], 3, axis=1) * self.converted_color_map
        # pr_mask = np.sum(pr_mask, axis=0).transpose((1, 2, 0)).astype(np.uint8)
        pr_masks = pr_masks.astype(np.uint8)

        for image_name, card_image, pr_mask in zip(image_names, card_images, pr_masks):
            print()
            print("="*20)
            print(image_name)

            if card_image is None:
                if print_infos:
                    for k, v in card_informations[image_name]['card_information']:
                        print(f"{k}: {v}")
                continue

            if self.save_dir and image_name:
                inf_save_dir = os.path.join(self.save_dir, image_name)

            segment_coords = {}
            ocr_info = {}
            for feat_name in ['identity_number', 'name', 'birthday', 'countryside', 'address']:
                if feat_name in list(self.colors_dict.keys()):
                    segment_coords[feat_name] = None
                    ocr_info[feat_name] = None

            ori_img = card_image.copy()
            ori_img_overlay = ori_img.copy()

            padimg = self.segmentation_augmentation(image=ori_img)['image']
            padimg_overlay = padimg.copy()
            
            pad_w = (padimg.shape[1] - ori_img.shape[1]) // 2 + 1
            pad_h = (padimg.shape[0] - ori_img.shape[0]) // 2 + 1

            augimg = self.segmentation_preprocessing(image=padimg)['image']

            for i, (feat_name, color) in enumerate(self.colors_dict.items(), start=1):
                cur_mask = pr_mask[i - 1]
                cur_mask = self.erode_leftovers(cur_mask, kernel_size=(5,5))
                # cv2.imwrite(f'clean_{feat_name}.jpg', cur_mask * 255)

                indices = np.where(cur_mask == 1)
                coords = np.array(list(zip(indices[0], indices[1])))
                # print(coords)

                if len(coords) == 0:
                    ocr_info[feat_name] = None
                    continue

                segment_coords[feat_name] = coords

                min_x, min_y, max_x, max_y = get_min_max_x_y(coords)
                width = max_x - min_x + 1
                height = max_y - min_y + 1
                white_area = np.zeros((height, width, 3), dtype=np.uint8) ###
                white_area[coords[:, 0] - min_y, coords[:, 1] - min_x] = padimg[coords[:, 0], coords[:, 1]]

                if self.save_dir and image_name:
                    cv2.imwrite(os.path.join(inf_save_dir, f'{feat_name}.jpg'), white_area)
                
                white_areas = []
                height_area = white_area.shape[0]
                if height_area > 30 and feat_name == 'address':
                    sep = height_area // 2
                    white_areas = [white_area[:sep], white_area[sep:]]
                else:
                    white_areas = [white_area,]
                cur_infos = []
                for sub_white_area in white_areas:
                    cur_infos.append(self.ocr_model.predict(sub_white_area))
                
                # ocr_info[feat_name] = self.ocr_model.predict(white_area)
                ocr_info[feat_name] = ' '.join(cur_infos)
                # print(ocr_info[feat_name])

                mask_map_aug = np.zeros_like(padimg)
                mask_map_aug[coords[:, 0], coords[:, 1]] = color
                padimg_overlay = cv2.addWeighted(src1=padimg_overlay, alpha=0.99,
                                        src2=mask_map_aug, beta=0.33, gamma=0.0)
                
                mask_map_ori = np.zeros_like(ori_img)

                mask_map_ori[coords[:, 0] - pad_h, coords[:, 1] - pad_w] = color
                ori_img_overlay = cv2.addWeighted(src1=ori_img_overlay, alpha=0.99,
                                        src2=mask_map_ori, beta=0.33, gamma=0.0)

            # addr_corr = AddressCorrection()
            # if print_infos:
            #     for feat_name in ocr_info.keys():
            #         if feat_name in list(self.colors_dict.keys()):
            #             print(f"{feat_name}: {ocr_info[feat_name]}")
            #             print(f"     After refined --> {addr_corr.address_correction(ocr_info[feat_name])}\n")

            if print_infos:
                for feat_name in ocr_info.keys():
                    if feat_name in list(self.colors_dict.keys()):
                        print(f"{feat_name}: {ocr_info[feat_name]}")

            if self.save_dir and image_name:
                cv2.imwrite(os.path.join(inf_save_dir, 'original.jpg'), ori_img)
                cv2.imwrite(os.path.join(inf_save_dir, 'original_augment.jpg'), padimg)
                cv2.imwrite(os.path.join(inf_save_dir, 'original_augment_overlay.jpg'), padimg_overlay)
                cv2.imwrite(os.path.join(inf_save_dir, 'original_overlay.jpg'), ori_img_overlay)

            if return_segment_coords:
                card_informations[image_name]['segment_coords'] = segment_coords
            if return_card_information:
                card_informations[image_name]['card_information'] = ocr_info