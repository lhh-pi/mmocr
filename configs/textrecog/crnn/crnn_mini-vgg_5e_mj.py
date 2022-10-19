# training schedule for 1x
_base_ = [
    '../_base_/datasets/mjsynth.py',
    '../_base_/datasets/cute80.py',
    '../_base_/datasets/iiit5k.py',
    '../_base_/datasets/svt.py',
    '../_base_/datasets/svtp.py',
    '../_base_/datasets/icdar2013.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adadelta_5e.py',
    '_base_crnn_mini-vgg.py',
]

# 每 10 个 epoch 储存一次权重
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=100), )
# 设置最大 epoch 数为 400，每 10 个 epoch 运行一次验证
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1000, val_interval=100)
# 令学习率为常量，即不进行学习率衰减
# param_scheduler = [dict(type='ConstantLR', factor=1.0),]

# dataset settings
train_list = [_base_.mj_rec_train]
# test_list = [
#     _base_.cute80_rec_test, _base_.iiit5k_rec_test, _base_.svt_rec_test,
#     _base_.svtp_rec_test, _base_.ic13_rec_test, _base_.ic15_rec_test
# ]
test_list = [
    _base_.cute80_rec_test, ]

default_hooks = dict(logger=dict(type='LoggerHook', interval=50), )

train_dataloader = dict(
    batch_size=64,
    num_workers=24,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=test_list,
        pipeline=_base_.test_pipeline))
val_dataloader = test_dataloader

# val_evaluator = dict(
#     dataset_prefixes=['CUTE80', 'IIIT5K', 'SVT', 'SVTP', 'IC13', 'IC15'])

val_evaluator = dict(
    dataset_prefixes=['CUTE80', ])
test_evaluator = val_evaluator

auto_scale_lr = dict(base_batch_size=64 * 4)
