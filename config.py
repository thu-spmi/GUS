import logging, time, os

class _Config:
    def __init__(self):
        self._multiwoz_damd_init()

    def _multiwoz_damd_init(self):

        self.notes=''
        # file path setting
        self.data_path = './data/multi-woz-2.1-processed/'
        self.data_file = 'data_for_rl.json'
        self.dev_list = 'data/MultiWOZ_2.1/valListFile.txt'
        self.test_list = 'data/MultiWOZ_2.1/testListFile.txt'
        self.dbs = {
            'attraction': 'db/attraction_db_processed.json',
            'hospital': 'db/hospital_db_processed.json',
            'hotel': 'db/hotel_db_processed.json',
            'police': 'db/police_db_processed.json',
            'restaurant': 'db/restaurant_db_processed.json',
            'taxi': 'db/taxi_db_processed.json',
            'train': 'db/train_db_processed.json',
        }
        self.domain_file_path = 'data/multi-woz-2.1-processed/domain_files.json'
        self.slot_value_set_path = 'db/value_set_processed.json'
        self.exp_path = 'to be generated'
        self.log_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

        


        # experiment settings
        self.mode = 'train'
        self.cuda = True
        self.cuda_device = [0]
        self.exp_no = '' 
        self.seed = 11
        self.save_log = True # tensorboard 
        self.evaluate_during_training = True # evaluate during training
        self.truncated = False

        # supervised training settings
        self.gpt_path = 'distilgpt2'
        self.lr = 1e-4 # learning rate
        self.warmup_steps = -1 # we use warm up ratio if warm up steps is -1
        self.warmup_ratio= 0.2 
        self.weight_decay = 0.0 
        self.gradient_accumulation_steps = 4
        self.batch_size = 8
        self.loss_reg=True # regularization for gradient accumulation
        self.gradient_checkpoint=False # use gradient checkpoint to accelerate training

        self.model_act=True
        self.save_type='max_score'# 'min_loss'/'max_reward'
        self.dataset=1 # 0 for multiwoz2.0, 1 for multiwoz2.1
        self.delex_as_damd = True 
        self.turn_level=True # turn-level training or session-level training
        self.input_history=False # whether or not add the whole dialog history into the training sequence if train with turn-level 
        self.input_prev_resp=True # whether or not add the prev response into the training sequence if input_history is False
        self.fix_data=True # correct the dataset

        self.lr_decay = 0.5
        self.use_scheduler=True
        self.epoch_num = 50
        self.early_stop=False
        self.early_stop_count = 5
        self.weight_decay_count = 10
        
        self.only_target_loss=True # only calculate the loss on target context
        self.clip_grad=True

        # evaluation settings
        self.eval_load_path = 'to be generated'
        self.use_true_prev_bspn = False
        self.use_true_prev_aspn = False
        self.use_true_db_pointer = False
        self.use_true_prev_resp = False
        self.use_true_curr_bspn = False
        self.use_true_curr_aspn = False
        self.use_all_previous_context = True
        self.use_true_bspn_for_ctr_eval = False
        self.use_true_domain_for_ctr_eval = True
        self.use_true_domain_for_ctr_train = True
        self.fast_validate=True # use batch generation for fast validationg
        self.eval_batch_size=32 # batch size during fast validationg
        self.val_set='test'
        self.col_samples=True # collect wrong predictions samples for observation
        self.use_existing_result=True
        self.result_file='validate_result.json' # validation results file for RL
        self.venue_overwrite=False
        self.debugging=False

        self.exp_domains = ['all'] # e.g. ['attraction', 'hotel'], ['except', 'attraction']
        self.log_path = ''
        self.low_resource = False

        # parameters for rl training
        self.rl_train=False
        self.delex_resp=True # whether to use delexicalized responses
        self.on_policy=True # on policy or off policy
        self.fix_DST=False # use a fixed DST model
        self.DST_path=''
        

        self.rl_dial_per_epoch=512
        self.rl_save_path='/mnt/workspace/liuhong/RL_exp'
        self.rl_iterate=True # alternate RL with supervised learning 
        self.rl_iterate_num=1
        # turn level reward: evaluate every turn
        # session level reward: evaluate the whole dialog and punish dialogs that are too long
        self.turn_level_reward=False 
        self.non_neg_reward=False # non-negative reward value (sigmoid)

        self.offline_RL=False
        self.validate_mode='offline' # offline/online validation
        self.rl_for_bspn=True # whether calculate the loss on bspn during policy gradient
        self.rl_for_resp=True # whether calculate the loss on resp during policy gradient
        
        self.sys_nlu=False # whether to add NLU module to dialog system, if True, the belief state will contain requestable slots
        #user simulator setting
        self.rl_with_us=True # whether or not interact with user simulator 
        self.train_us=False # setting when pre-training user simulator 
        self.train_sys=False
        self.user_nlu=True # whether to add NLU module to user simulator
        self.strict_eval=True # strict evaluation
        self.same_policy_as_agenda=True # if True, the US will ask questions repeatedly until the DS provides the corresponding slot values
        self.interact_with_ABUS=False # train the DS with ABUS
        self.dev_size=192 # RL dev size
        self.init_eval=False # whether evaluate the model before training
        self.goal_path='' 

        self.joint_train=True # train DS and US together in RL exp
        self.joint_train_us=True # optimize us during RL
        self.joint_train_ds=True # optimize ds during RL
        self.goal_from_data=True # whether or not use goals in original data
        self.random_goal=False # randomly generate goal using rule-based method
        self.full_goal=False # replace the goal state with the full goal
        self.traverse_data=True # traverse all data in training set for one RL epoch
        self.save_by_reward=True # save the model with max average reward
        self.deliver_state=False # deliver the dialog state from US to DS during interaction
        self.consider_nobook=False # whether to consider the no-book situation
        self.consider_offerbooked=False # whether to consider the offerbooked situation
        self.gen_goal_state=False # generate goal state instead of updating it by rules
        

        self.sys_act_ctrl=False # post-process for generated system acts
        self.simple_reward=True # use success as reward
        self.simple_training=False
        self.transpose_batch=False
        
        self.DS_path="" # The pretrained DS path
        self.US_path="" # The pretrained US path
        self.DS_device=0
        self.US_device=0

        self.add_end_reward=False # give punishment if the goal state is not empty in the end
        self.interaction_batch_size=32 # batch size when the two agents interact with each other
        self.training_batch_size=8 # training batch size during RL
        self.rl_accumulation_steps=4 # training accumulaion steps during RL

        self.add_rl_baseline=False # add baseline function to reduce reward variance during RL
        self.RL_ablation=False # ablation experiment: only SL no RL during RL training
        self.iterative_update=False # iteratively update DS and US by cycle

        self.eval_as_simpletod=True # evaluate the results as SimpleTOD
        self.beam_search=False # use beam search when generating system acts
        self.beam_size_u=3
        self.beam_size_s=8
        self.fix_db=True

        # old settings
        self.vocab_size = 3000
        self.enable_aspn = True
        self.enable_bspn = True
        self.bspn_mode = 'bspn' # 'bspn' or 'bsdx'
        self.enable_dspn = False # removed
        self.enable_dst = False
        self.same_eval_act_f1_as_hdsa = False

        

    def __str__(self):
        s = ''
        for k,v in self.__dict__.items():
            s += '{} : {}\n'.format(k,v)
        return s


    def _init_logging_handler(self, mode):
        stderr_handler = logging.StreamHandler()
        if not os.path.exists('./log'):
            os.mkdir('./log')
        if self.save_log and mode in ['semi_ST', 'semi_VL', 'semi_jsa', 'train', 'pretrain']:
            if self.dataset==0:
                #file_handler = logging.FileHandler('./log/log_{}_{}_{}_{}_sd{}.txt'.format(self.log_time, mode, '-'.join(self.exp_domains), self.exp_no, self.seed))
                file_handler = logging.FileHandler('./log/log_{}_{}_sd{}.txt'.format(mode, self.exp_no, self.seed))
            elif self.dataset==1:
                #file_handler = logging.FileHandler('./log21/log_{}_{}_{}_{}_sd{}.txt'.format(self.log_time, mode, '-'.join(self.exp_domains), self.exp_no, self.seed))
                file_handler = logging.FileHandler('./log21/log_{}_{}_sd{}.txt'.format(mode, self.exp_no, self.seed))
        elif 'test' in mode and os.path.exists(self.eval_load_path):
            eval_log_path = os.path.join(self.eval_load_path, 'eval_log.json')
            # if os.path.exists(eval_log_path):
            #     os.remove(eval_log_path)
            file_handler = logging.FileHandler(eval_log_path)
            file_handler.setLevel(logging.INFO)
        else:
            pass
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(handlers=[stderr_handler, file_handler])
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

global_config = _Config()