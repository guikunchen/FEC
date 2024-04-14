_base_ = [
    '../_base_/models/fpn_r50.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='EncoderDecoder',
    # pretrained='https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s12.pth.tar', # for old version of mmsegmentation 
    backbone=dict(
        type='fec_small_seg',
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', 
            checkpoint='</path/to/ckpt>',
            ),
        ),
    neck=dict(in_channels=[32, 64, 196, 320]),
    decode_head=dict(num_classes=150))


find_unused_parameters=True
gpu_multiples = 1
# optimizer
optimizer = dict(type='AdamW', lr=0.0001*gpu_multiples, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000//gpu_multiples)
checkpoint_config = dict(by_epoch=False, interval=8000//gpu_multiples)
evaluation = dict(interval=8000//gpu_multiples, metric='mIoU')
device = 'cuda'
