# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
python pretrain.py -mode train\
    -cfg  lr=1e-4\
    gradient_accumulation_steps=2 batch_size=16\
    seed=11\
    epoch_num=50\
    save_type=min_loss\
    only_target_loss=False\
    cuda_device=$1\
    train_us=True\
    exp_no=US-baseline