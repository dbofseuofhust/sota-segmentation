_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/naicseg.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

# ohem
model=dict(
    decode_head=dict(
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)) )