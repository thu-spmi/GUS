# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
python pretrain.py -mode train\
    -cfg  lr=1e-4\
    gradient_accumulation_steps=2 batch_size=16\
    seed=11\
    epoch_num=50\
    save_type=max_score\
    cuda_device=$1\
    exp_no=DS-baseline