optimizer = dict(type='Adam', lr=0.000625, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)
lr_mult = 8
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.001,
    step=[4*lr_mult, 6*lr_mult])
total_epochs = 8*lr_mult
checkpoint_config = dict(interval=5)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = None
load_from = '/home/manhdq/ID_Card_Information_Extraction/keypoint_detection/scrfd/work_dirs/scrfd_cccd_500m_bnkps_dummy/best.pth'
resume_from = None
workflow = [('train', 1)]
dataset_type = 'RetinaFaceDataset'
data_root = '/home/manhdq/ID_Card_Information_Extraction/datasets/cccd'
train_root = ['/home/manhdq/ID_Card_Information_Extraction/datasets/cccd/valid_cccd_clean_v2',
            '/home/manhdq/ID_Card_Information_Extraction/datasets/cccd/giaosu_tien']
val_root = '/home/manhdq/ID_Card_Information_Extraction/datasets/cccd/valid_cccd_clean'
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0], to_rgb=True)
albu_train_transforms=[
        dict(
            type='OneOf',
            transforms=[
                dict(
                    type='Affine',
                    scale=None,
                    rotate=(-90, 90),
                    shear=None,
                    interpolation=0,
                    fit_output=True,
                    ),
                # dict(
                #     type='RandomRotate90',
                #     ),
            ],
            p=0.6
        ),
        dict(
            type='OneOf',
            transforms=[
                dict(
                    type='MotionBlur',
                    ),
                dict(
                    type='GaussianBlur',
                    blur_limit=3,
                    ),
            ],
            p=0.2
        ),
        # dict(
        #     type='OneOf',
        #     transforms=[
        #         dict(
        #             type='IAAEmboss',
        #             ),
        #         dict(
        #             type='RandomBrightnessContrast',
        #             ),
        #         dict(
        #             type='RandomBrightness',
        #             ),
        #         dict(
        #             type='RandomContrast',
        #             ),
        #     ],
        #     p=0.2
        # ),
        # dict(
        #     type='OneOf',
        #     transforms=[
        #         dict(
        #             type='ISONoise',
        #             p=0.1),
        #         dict(
        #             type='GaussNoise',
        #             p=0.1),
        #     ],
        #     p=0.2
        # ),
        # dict(
        #     type='RandomGamma',
        #     p=0.2),
        dict(
            type='ToGray',
            p=0.2),
        # dict(
        #     type='JpegCompression',
        #     quality_lower=30, 
        #     quality_upper=80,
        #     p=1),
    ]
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_keypoints=True),
    dict(
        type='RandomSquareCrop',
        crop_choice=[0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=False),
    # dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomFlip', flip_ratio=0.5, direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[128.0, 128.0, 128.0],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore',
            'gt_keypointss'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[128.0, 128.0, 128.0],
                to_rgb=True),
            dict(type='Pad', size=(640, 640), pad_val=0),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=8,
    train=dict(
        type='RetinaFaceDataset',
        ann_file=['/home/manhdq/ID_Card_Information_Extraction/datasets/cccd/valid_cccd_clean_v2/annotations.txt',
                '/home/manhdq/ID_Card_Information_Extraction/datasets/cccd/giaosu_tien/annotations.txt'],
        img_prefix=['/home/manhdq/ID_Card_Information_Extraction/datasets/cccd/valid_cccd_clean_v2/images',
                '/home/manhdq/ID_Card_Information_Extraction/datasets/cccd/giaosu_tien/images'],
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True, with_keypoints=True),
            # dict(
            #     type='Albu',
            #     transforms=albu_train_transforms,
            #     bbox_params=dict(
            #             type='BboxParams',
            #             format='pascal_voc',
            #             label_fields=['gt_labels'],
            #             min_visibility=0.5),
            #     keypoint_params=dict(
            #             type='KeypointParams',
            #             format='xy'),
            #     refine_bbox_from_keypoint=True,
            #     keymap={
            #         'img': 'image',
            #         'gt_bboxes': 'bboxes',
            #         'gt_keypointss': 'keypoints'
            #     },
            #     update_pad_shape=False,
            #     skip_img_without_anno=True),
            dict(
                type='RandomSquareCrop',
                crop_choice=[
                    0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0
                ]),
            dict(type='Resize', img_scale=(640, 640), keep_ratio=False),
            dict(type='RandomFlip', flip_ratio=0.5, direction=['horizontal', 'vertical', 'diagonal']),
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[128.0, 128.0, 128.0],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=[
                    'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore',
                    'gt_keypointss'
                ])
        ]),
    val=dict(
        type='RetinaFaceDataset',
        ann_file='/home/manhdq/ID_Card_Information_Extraction/datasets/cccd/valid_cccd_clean/annotations.txt',
        img_prefix='/home/manhdq/ID_Card_Information_Extraction/datasets/cccd/valid_cccd_clean/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_keypoints=True),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=0.0),
                    dict(
                        type='Normalize',
                        mean=[127.5, 127.5, 127.5],
                        std=[128.0, 128.0, 128.0],
                        to_rgb=True),
                    dict(type='Pad', size=(640, 640), pad_val=0),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    infer=dict(
        type='RetinaFaceDataset',
        ann_file='/home/manhdq/ID_Card_Information_Extraction/datasets/cccd/valid_cccd_clean/annotations.txt',
        img_prefix='/home/manhdq/ID_Card_Information_Extraction/datasets/cccd/valid_cccd_clean/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_keypoints=True),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=0.0),
                    dict(
                        type='Normalize',
                        mean=[127.5, 127.5, 127.5],
                        std=[128.0, 128.0, 128.0],
                        to_rgb=True),
                    dict(type='Pad', size=(640, 640), pad_val=0),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore',
                    'gt_keypointss'])
                ])
        ]),
    test=dict(
        type='RetinaFaceDataset',
        ann_file='/home/manhdq/ID_Card_Information_Extraction/datasets/cccd/valid_cccd_clean/annotations.txt',
        img_prefix='/home/manhdq/ID_Card_Information_Extraction/datasets/cccd/valid_cccd_clean/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=0.0),
                    dict(
                        type='Normalize',
                        mean=[127.5, 127.5, 127.5],
                        std=[128.0, 128.0, 128.0],
                        to_rgb=True),
                    dict(type='Pad', size=(640, 640), pad_val=0),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
model = dict(
    type='SCRFD',
    backbone=dict(
        type='MobileNetV1',
        block_cfg=dict(
            stage_blocks=(2, 3, 2, 6), stage_planes=[16, 16, 40, 72, 152,
                                                     288])),
    neck=dict(
        type='PAFPN',
        in_channels=[40, 72, 152, 288],
        out_channels=16,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=3),
    bbox_head=dict(
        type='SCRFDHead',
        num_classes=1,
        in_channels=16,
        stacked_convs=2,
        feat_channels=64,
        norm_cfg=dict(type='BN', requires_grad=True),
        #norm_cfg=dict(type='GN', num_groups=16, requires_grad=True),
        cls_reg_share=True,
        strides_share=False,
        dw_conv=True,
        scale_mode=0,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales=[1, 2],
            base_sizes=[16, 64, 256],
            strides=[8, 16, 32]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=False,
        reg_max=8,
        loss_bbox=dict(type='DIoULoss', loss_weight=1.0),
        use_kps=True,
        loss_kps=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.5),
        train_cfg=dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            nms_pre=-1,
            min_bbox_size=0,
            score_thr=0.02,
            nms=dict(type='nms', iou_threshold=0.45),
            max_per_img=-1)))
train_cfg = dict(
    assigner=dict(type='ATSSAssigner', topk=9),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
val_cfg = dict(
    nms_pre=-1,
    min_bbox_size=0,
    score_thr=0.02,
    nms=dict(type='nms', iou_threshold=0.45),
    max_per_img=-1)
test_cfg = dict(
    nms_pre=-1,
    min_bbox_size=0,
    score_thr=0.02,
    nms=dict(type='nms', iou_threshold=0.45),
    max_per_img=-1)
epoch_multi = 1
evaluation = dict(interval=5, metric='mAP')