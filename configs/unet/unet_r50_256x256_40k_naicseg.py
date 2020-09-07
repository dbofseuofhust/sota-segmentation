_base_ = [
    '../_base_/models/unet_r50.py',
    '../_base_/datasets/naicseg.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

model = dict(
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        include_stem=True))

