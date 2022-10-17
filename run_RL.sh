path1=experiments_21/DS-baseline/best_score_model
path2=experiments_21/US-baseline/best_loss_model
exp_name=RL-exp
python session_RL.py -mode train\
 -cfg use_scheduler=False\
 lr=2e-5\
 weight_decay_count=100\
 seed=11\
 epoch_num=100\
 training_batch_size=16\
 rl_accumulation_steps=12\
 interaction_batch_size=32\
 DS_path=$path1 US_path=$path2\
 DS_device=$1 US_device=$1\
 rl_for_bspn=True\
 non_neg_reward=False\
 rl_dial_per_epoch=128\
 joint_train_ds=True\
 joint_train_us=False\
 simple_reward=False\
 exp_no=$exp_name\
 rl_iterate=True\
 beam_search=True\
 interact_with_ABUS=False\
 full_goal=False\
 dev_size=192\
 random_goal=True
