# optimizer
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.0001)
# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=False,
)
# learning policy
lr_config = dict(
    policy='step',
    gamma=0.2,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 9, 11])

runner = dict(type='EpochBasedRunner', max_epochs=12)
