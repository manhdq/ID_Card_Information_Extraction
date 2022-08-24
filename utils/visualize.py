import os
import sys
import cv2
import numpy as np


def kpts_image_save(image, kpts, save_file):
    if not isinstance(kpts, np.ndarray):
        kpts = np.array(kpts)
    assert kpts.shape[1] == 2
    kpts = kpts.astype(int)

    if os.path.isdir(save_file):
        save_file = os.path.join(save_file, 'kpts_demo.jpg')

    kpts_image = image.copy()
    kpts_image = cv2.polylines(kpts_image, [kpts.astype(np.int32)], True, (0, 0, 255), 5)
    for kpt in kpts:
        # print(kpt)
        kpts_image = cv2.circle(kpts_image, tuple(kpt), radius=10, color=(255, 0, 0), thickness=5)
        kpts_image = cv2.circle(kpts_image, tuple(kpt), radius=5, color=(0, 0, 255), thickness=5)
    
    cv2.imwrite(save_file, kpts_image)
