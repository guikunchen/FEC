_base_ = [
    '_base_/models/mask_rcnn_r50_fpn.py',
    '_base_/datasets/coco_instance.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]
# optimizer
model = dict(
    backbone=dict(
        type='fec_base_det',
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=\
                '</path/to/ckpt>',
            ),
        ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5))


find_unused_parameters=True
