python pretrain.py -mode test\
    -cfg gpt_path=$2  cuda_device=$1\
    turn_level=True\
    input_history=False\
    input_prev_resp=True\
    eval_batch_size=32\
    use_existing_result=False

