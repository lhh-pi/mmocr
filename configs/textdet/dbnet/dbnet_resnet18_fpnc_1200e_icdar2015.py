_base_ = [
    '_base_dbnet_resnet18_fpnc.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]

# 每 10 个 epoch 储存一次权重
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=10), )
# 设置最大 epoch 数为 400，每 10 个 epoch 运行一次验证
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=400, val_interval=10)
# 令学习率为常量，即不进行学习率衰减
param_scheduler = [dict(type='ConstantLR', factor=1.0),]

# dataset settings
ic15_det_train = _base_.ic15_det_train
ic15_det_train.pipeline = _base_.train_pipeline
ic15_det_test = _base_.ic15_det_test
ic15_det_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=ic15_det_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=ic15_det_test)

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=16)
