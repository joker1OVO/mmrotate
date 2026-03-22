# evaluation
evaluation = dict(interval=4, metric='mAP') # 表示每训练 1 个 epoch 后进行模型评估。
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001) # 使用 随机梯度下降（SGD）作为优化器.学习率（learning rate）设置为 0.0025。动量（momentum）设置为 0.9，用于加速 SGD 的收敛。权重衰减（weight decay）设置为 0.0001，用于正则化，防止过拟合。
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2),type='GradientCumulativeOptimizerHook',cumulative_iters=2) # 梯度累积：2，梯度裁剪（gradient clipping）配置，用于防止梯度爆炸。梯度的最大范数为 35。使用 L2 范数进行梯度裁剪。
# learning policy
lr_config = dict(
    policy='step', # 学习率调整策略为 step，即按固定步长调整学习率。
    warmup='linear', # 使用 线性预热（linear warmup） 策略，逐渐增加学习率。
    warmup_iters=500, # 预热阶段的迭代次数为 500。
    warmup_ratio=1.0 / 3, #预热阶段的学习率从初始学习率的 1/3 开始。
    step=[8, 11]) # 在第 8 和第 11 个 epoch 时，学习率会按一定比例下降（通常是乘以 0.1）。
runner = dict(type='EpochBasedRunner', max_epochs=12) # 使用基于 epoch 的训练循环。最大训练 epoch 数为 12。
checkpoint_config = dict(interval=4) # 每训练 4 个 epoch 后保存一次模型检查点。
