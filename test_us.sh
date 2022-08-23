python pretrain.py -mode test\
    -cfg gpt_path=$2  cuda_device=$1\
    eval_batch_size=32\
    train_us=True\
    user_nlu=True\
    full_goal=False
