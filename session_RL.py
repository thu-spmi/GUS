import torch
import random
import time
import os
import logging
import argparse
import numpy as np
import json
import shutil
import torch.nn as nn
import torch.nn.functional as F
from config import global_config as cfg
from rl_utils.rl_util import *
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2LMHeadModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from session import turn_level_session
from test_with_ABUS import get_ABUS
def parse_arg_cfg(args):
    # add args to cfg
    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            elif dtype is list:
                v = v.split(',')
                if k == 'cuda_device':
                    v = [int(no) for no in v]
            else:
                v = dtype(v)
            setattr(cfg, k, v)
    return
class RL_session():
    def __init__(self, sess):
        self.sess=sess
        self.reader=sess.reader
        self.evaluator=sess.evaluator
        self.DS, self.US = sess.DS, sess.US
        self.DS_tok, self.US_tok = sess.DS_tok, sess.US_tok
        if not cfg.on_policy:
            self.DS_off, self.US_off =sess.DS_off, sess.US_off
        self.global_output1=2
        self.global_output2=2
        # tensorboard
        if cfg.save_log:
            if not os.path.exists('./log_rl'):
                os.mkdir('./log_rl')
            log_path='./log_rl/log_{}'.format(cfg.exp_no)
            if os.path.exists(log_path):
                shutil.rmtree(log_path)
                os.mkdir(log_path)
            else:
                os.mkdir(log_path)
            self.tb_writer = SummaryWriter(log_dir=log_path)
        else:
            self.tb_writer = None
        
        if cfg.interact_with_ABUS:
            self.ABUS_analyzer=get_ABUS(cuda_device=self.DS.device.index)


    def run_RL(self):
        json.dump(cfg.__dict__,open(os.path.join(cfg.exp_path,'cfg_all.json'),'w'),indent=2)
        cfg.origin_batch_size=cfg.batch_size
        cfg.batch_size=cfg.batch_size*cfg.gradient_accumulation_steps
        # The total turn samples per epoch: dial_per_epoch*avg_turn_num, we think avg_turn_num=8
        self.global_step=0
        if cfg.add_rl_baseline:
            self.avg_ds_reward=0
            self.avg_us_reward=0
            self.total_ds_turn=0
            self.total_us_turn=0
        if cfg.joint_train_ds:
            if cfg.on_policy:
                self.optimizer, self.scheduler=self.get_optimizers(num_dials=cfg.rl_dial_per_epoch*8, model=self.DS)
            else:
                self.optimizer, self.scheduler=self.get_optimizers(num_dials=cfg.rl_dial_per_epoch*8, model=self.DS_off)
        if cfg.joint_train_us:
            if cfg.on_policy:
                self.optimizer_us, self.scheduler_us=self.get_optimizers(num_dials=cfg.rl_dial_per_epoch*8, model=self.US)
            else:
                self.optimizer_us, self.scheduler_us=self.get_optimizers(num_dials=cfg.rl_dial_per_epoch*8, model=self.US_off)
        # sample or generate some goals for validation
        if cfg.random_goal:
            goal_batches=[]
            dial_id_batches=None
            for _ in range(cfg.dev_size//cfg.interaction_batch_size):
                goal_batch=[]
                for k in range(cfg.interaction_batch_size):
                    self.sess.user_policy.init_session()
                    goal=self.sess.user_policy.get_goal()
                    goal=self.reader.goal_norm(goal)
                    goal_batch.append(goal)
                goal_batches.append(goal_batch)
        else:
            dial_id_batches=[]
            goal_batches=None
            for _ in range(cfg.dev_size//cfg.interaction_batch_size):
                dial_id_batches.append(random.sample(self.reader.train_list + self.reader.dev_list, cfg.interaction_batch_size))
        if cfg.init_eval:
            logging.info('Initial validation')
            self.validate(self.DS, self.US, dial_id_batches, goal_batches)
        
        max_score=0
        early_stop_count=cfg.early_stop_count
        weight_decay_count=cfg.weight_decay_count
        lr=cfg.lr
        logging.info('Dialogs per rl epoch:{}'.format(cfg.rl_dial_per_epoch))
        self.DS_training_steps=0
        self.US_training_steps=0
        for epoch in range(cfg.epoch_num):
            st=time.time()
            if cfg.iterative_update:
                cfg.joint_train_ds=True if epoch%2==0 else False
                cfg.joint_train_us=not cfg.joint_train_ds
            avg_DS_reward, avg_US_reward=self.run_RL_epoch()
            logging.info('Epoch:{}, time:{:.3f} min, DS steps:{}, US steps:{}'.format(
                epoch, (time.time()-st)/60, self.DS_training_steps, self.US_training_steps))
            logging.info('Training -- Avg DS reward:{:3f}, avg US reward:{:3f}'.format(avg_DS_reward, avg_US_reward))
            
            temp=cfg.beam_search
            cfg.beam_search=False
            if cfg.on_policy:
                DS_reward, US_reward, avg_turn, success, match =self.validate(self.DS, self.US, dial_id_batches, goal_batches)
            else:# validate the updating model
                DS_reward, US_reward, avg_turn, success, match =self.validate(self.DS_off, self.US_off, dial_id_batches, goal_batches)
            cfg.beam_search=temp
            eval_metric=success+match

            if eval_metric>max_score:
                max_score=eval_metric
                if cfg.on_policy:
                    self.save_model(self.DS, self.US)
                else:# save the updating model
                    self.save_model(self.DS_off, self.US_off)
                logging.info('model saved in {}'.format(cfg.exp_path))
                early_stop_count=cfg.early_stop_count
            else:
                early_stop_count-=1
                weight_decay_count-=1
            if early_stop_count==0 and cfg.early_stop:
                print('early stop')
                break
            if weight_decay_count==0 and not cfg.use_scheduler:
                if self.optimizer:
                    for group in self.optimizer.param_groups:
                        group['lr'] = group['lr']*cfg.lr_decay
                if self.optimizer_us:
                    for group in self.optimizer_us.param_groups:
                        group['lr'] = group['lr']*cfg.lr_decay
                lr=group['lr']
                print("learning rate decay to {}".format(lr))
                weight_decay_count = cfg.weight_decay_count
                if lr<1e-9:
                    print('learning rate too small, break')
                    break

            if self.tb_writer:
                #self.tb_writer.add_scalar('Train_DS_reward', avg_DS_reward, epoch)
                #self.tb_writer.add_scalar('Train_US_reward', avg_US_reward, epoch)        
                self.tb_writer.add_scalar('Dev_DS_reward', DS_reward, epoch)
                self.tb_writer.add_scalar('Dev_US_reward', US_reward, epoch)
                self.tb_writer.add_scalar('Avg_turns', avg_turn, epoch)
                self.tb_writer.add_scalar('Match', match, epoch)
                self.tb_writer.add_scalar('Success', success, epoch)
        #self.save_model(self.DS, self.US, last_model=True)

    def run_RL_epoch(self):
        avg_US_reward=0
        avg_DS_reward=0
        ds_backward_count=0
        us_backward_count=0
        for _ in range(cfg.rl_dial_per_epoch//cfg.interaction_batch_size):
            for iter in range(1+cfg.rl_iterate_num):
                if iter==0:
                    if cfg.offline_RL:
                        continue
                    if cfg.RL_ablation:# skip RL section for ablation study
                        continue
                    if cfg.interact_with_ABUS:
                        gen_batch_dict=self.ABUS_analyzer.comprehensive_analyze(sys_agent=self.sess, model_name='turn-level-GPT',\
                            total_dialog=cfg.interaction_batch_size, return_dial=True)
                        gen_batch, DS_reward_batch, _=self.handle_ABUS_dialog(gen_batch_dict)
                    else:
                        gen_batch, US_reward_batch, DS_reward_batch, _ , _, _, US_rewards, DS_rewards=\
                            self.sess.interact_by_batch(self.DS, self.US, cfg.interaction_batch_size, return_reward=True)
                        if DS_rewards[-1]<0 or US_rewards[-1]<-1.5:
                            logging.info('Generated batch with repeated tokens')
                            continue
                        avg_US_reward+=US_rewards[0]
                        avg_DS_reward+=DS_rewards[0]
                        US_rewards/=cfg.interaction_batch_size
                        DS_rewards/=cfg.interaction_batch_size
                        self.tb_writer.add_scalar('us_reward', US_rewards[0], self.global_step)
                        self.tb_writer.add_scalar('us_reqt_reward', US_rewards[1], self.global_step)
                        self.tb_writer.add_scalar('us_goal_reward', US_rewards[2], self.global_step)
                        self.tb_writer.add_scalar('us_repeat_reward', US_rewards[3], self.global_step)
                        self.tb_writer.add_scalar('us_token_reward', US_rewards[-1], self.global_step)
                        self.tb_writer.add_scalar('us_goal_comp_reward', US_rewards[4], self.global_step)
                        self.tb_writer.add_scalar('avg_turn_num', US_rewards[5], self.global_step)
                        self.tb_writer.add_scalar('ds_reward', DS_rewards[0], self.global_step)
                        self.tb_writer.add_scalar('ds_reqt_reward', DS_rewards[1], self.global_step)
                        self.tb_writer.add_scalar('ds_repeat_reward', DS_rewards[2], self.global_step)
                        self.tb_writer.add_scalar('ds_success_reward', DS_rewards[3], self.global_step)
                        self.tb_writer.add_scalar('ds_token_reward', DS_rewards[-1], self.global_step)
                    self.global_step+=1
                else:
                    # sample batches from dataset
                    gen_batch=[]
                    dial_id_batch=random.sample(self.reader.train_list, cfg.interaction_batch_size)
                    for dial_id in dial_id_batch:
                        dial_id=dial_id+'.json' if '.json' not in dial_id else dial_id
                        gen_batch.append(self.reader.data[dial_id]['log'])
                    US_reward_batch=[[1]*len(dial) for dial in gen_batch]
                    DS_reward_batch=[[1]*len(dial) for dial in gen_batch]
                if cfg.on_policy: #need to change mode from eval to train
                    self.DS.train()
                    if self.US:
                        self.US.train()
                else:
                    self.DS_off.train()
                    if self.US_off:
                        self.US_off.train()
                # Two different tokenizer
                assert cfg.joint_train_ds or cfg.joint_train_us
                if cfg.joint_train_ds:
                    gen_batch_ids=self.reader.convert_batch_tokens_to_ids(gen_batch, self.DS_tok)
                    ds_turn_batches, ds_label_batches, ds_reward_batches = self.reader.transpose_ds_turn_batch(
                        gen_batch_ids, DS_reward_batch)
                    for i, (turn_batch, label_batch, reward_batch) in enumerate(zip(ds_turn_batches, ds_label_batches, ds_reward_batches)):
                        if self.global_output1>0:
                            logging.info(self.DS_tok.decode(list(turn_batch[0])))
                            self.global_output1-=1
                        input_tensor = torch.from_numpy(turn_batch).long().to(self.DS.device)
                        label_tensor = torch.from_numpy(label_batch).long().to(self.DS.device)
                        outputs=self.DS(input_tensor) if cfg.on_policy else self.DS_off(input_tensor)
                        if cfg.add_rl_baseline:
                            if cfg.simple_training:
                                loss=self.calculate_loss(outputs, input_tensor, reward_batch, base=self.avg_ds_reward)
                            else:
                                loss=self.calculate_loss(outputs, label_tensor, reward_batch, base=self.avg_ds_reward)
                            self.avg_ds_reward=self.avg_ds_reward*self.total_ds_turn+sum(reward_batch)
                            self.total_ds_turn+=len(reward_batch)
                            self.avg_ds_reward/=self.total_ds_turn
                            self.tb_writer.add_scalar('total_avg_ds_reward', self.avg_ds_reward, self.DS_training_steps)
                        else:
                            if cfg.simple_training:
                                loss=self.calculate_loss(outputs, input_tensor, reward_batch)
                            else:
                                loss=self.calculate_loss(outputs, label_tensor, reward_batch)
                        if cfg.loss_reg:
                            loss/=cfg.rl_accumulation_steps
                        loss.backward()
                        if cfg.clip_grad:
                            torch.nn.utils.clip_grad_norm_(self.DS.parameters(), 5.0)
                        # we optimize after every minibatch
                        ds_backward_count+=1
                        if ds_backward_count==cfg.rl_accumulation_steps or i==len(ds_turn_batches)-1:
                            self.optimizer.step()
                            if self.scheduler:
                                self.scheduler.step()
                            self.optimizer.zero_grad()
                            self.tb_writer.add_scalar('DS-lr', self.optimizer.param_groups[0]["lr"], self.DS_training_steps)
                            self.DS_training_steps+=1
                            ds_backward_count=0
                if cfg.joint_train_us:
                    gen_batch_ids=self.reader.convert_batch_tokens_to_ids(gen_batch, self.US_tok)
                    us_turn_batches, us_label_batches, us_reward_batches = self.reader.transpose_us_turn_batch(
                        gen_batch_ids, US_reward_batch, self.US_tok)
                    for i, (turn_batch, label_batch, reward_batch) in enumerate(zip(us_turn_batches, us_label_batches, us_reward_batches)):
                        if self.global_output2>0:
                            logging.info(self.US_tok.decode(list(turn_batch[0])))
                            self.global_output2-=1
                        input_tensor = torch.from_numpy(turn_batch).long().to(self.US.device)
                        label_tensor = torch.from_numpy(label_batch).long().to(self.US.device)
                        outputs=self.US(input_tensor) if cfg.on_policy else self.US_off(input_tensor)
                        if cfg.add_rl_baseline:
                            if cfg.simple_training or cfg.user_nlu:
                                loss=self.calculate_loss(outputs, input_tensor, reward_batch, base=self.avg_us_reward)
                            else:
                                loss=self.calculate_loss(outputs, label_tensor, reward_batch, base=self.avg_us_reward)
                            self.avg_us_reward=self.avg_us_reward*self.total_us_turn+sum(reward_batch)
                            self.total_us_turn+=len(reward_batch)
                            self.avg_us_reward/=self.total_us_turn
                            self.tb_writer.add_scalar('total_avg_us_reward', self.avg_us_reward, self.US_training_steps)
                        else:
                            if cfg.simple_training or cfg.user_nlu:
                                # we train the US with the whole sequence as target on two conditions
                                # 1) simple training
                                # 2) the US has an NLU module so that the previous aspn also becomes target 
                                loss=self.calculate_loss(outputs, input_tensor, reward_batch)
                            else:
                                loss=self.calculate_loss(outputs, label_tensor, reward_batch)
                        if cfg.loss_reg:
                            loss/=cfg.rl_accumulation_steps
                        loss.backward()
                        if cfg.clip_grad:
                            torch.nn.utils.clip_grad_norm_(self.US.parameters(), 5.0)
                        us_backward_count+=1
                        if us_backward_count==cfg.rl_accumulation_steps or i==len(us_turn_batches)-1:
                            self.optimizer_us.step()
                            if self.scheduler_us:
                                self.scheduler_us.step()
                            self.optimizer_us.zero_grad()
                            self.tb_writer.add_scalar('US-lr', self.optimizer_us.param_groups[0]["lr"], self.US_training_steps)
                            self.US_training_steps+=1
                            us_backward_count=0

        avg_DS_reward/=cfg.rl_dial_per_epoch
        avg_US_reward/=cfg.rl_dial_per_epoch
        return avg_DS_reward, avg_US_reward
    
    def evaluation(self, goal_list=None):
        logging.info('DS path:{}, US path:{}'.format(cfg.DS_path, cfg.US_path))
        pointer=0
        goal_list=[self.reader.goal_norm(goal) for goal in goal_list]
        if goal_list:
            goal_batches=[]
            dial_num=32 if cfg.debugging else len(goal_list)
            while(pointer<=dial_num):
                if pointer+cfg.interaction_batch_size<=len(self.reader.test_list):
                    goal_batches.append(goal_list[pointer:pointer+cfg.interaction_batch_size])
                else:
                    goal_batches.append(goal_list[pointer:])
                pointer+=cfg.interaction_batch_size
            self.validate(self.DS, self.US, init_goal_batches=goal_batches)
        else:
            dial_id_batches=[]
            dial_num=32 if cfg.debugging else len(self.reader.test_list)
            while(pointer<=dial_num):
                if pointer+cfg.interaction_batch_size<=len(self.reader.test_list):
                    dial_id_batches.append(self.reader.test_list[pointer:pointer+cfg.interaction_batch_size])
                else:
                    dial_id_batches.append(self.reader.test_list[pointer:])
                pointer+=cfg.interaction_batch_size
            self.validate(self.DS, self.US, dial_id_batches)

    def validate(self, DS, US, dial_id_batches=None, init_goal_batches=None):
        logging.info("Start validation")
        avg_US_reward=0
        avg_DS_reward=0
        success=0
        match=0
        goal_comp=0
        total=0
        avg_turn=0
        all_dials=[]
        st=time.time()
        if os.path.exists(os.path.join(cfg.exp_path, cfg.result_file)) and cfg.mode=='test' and cfg.use_existing_result:
            all_dials=json.load(open(os.path.join(cfg.exp_path, cfg.result_file), 'r')) 
            total_success, total_match = 0, 0
            for dial in all_dials:
                if 'final_goal' not in dial[-1]:
                    continue
                final_goal=dial[-1]['final_goal']
                final_goal=self.reader.aspan_to_act_dict(final_goal, side='user') if isinstance(final_goal, str) else final_goal
                success, match=self.evaluator.get_metrics(final_goal,  dial)
                total_success+=success
                total_match+=match
            logging.info('Success:{:.3f}, Match:{:.3f}'.format(total_success/len(all_dials), total_match/len(all_dials)))
            return 0

        if cfg.interact_with_ABUS:
            total+=cfg.interaction_batch_size
            gen_batch_dict=self.ABUS_analyzer.comprehensive_analyze(sys_agent=self.sess, model_name='turn-level-GPT',\
                total_dialog=cfg.interaction_batch_size, return_dial=True)
            gen_batch, _, metrics=self.handle_ABUS_dialog(gen_batch_dict)
            success+=metrics[0]
            match+=metrics[1]
            avg_turn+=sum([len(dial) for dial in gen_batch])
        else:
            if init_goal_batches is None:
                for dial_id_batch in dial_id_batches:
                    total+=len(dial_id_batch)
                    gen_batch, US_reward_batch, DS_reward_batch, batch_success, batch_match, batch_comp, _, _=\
                        self.sess.interact_by_batch(DS, US, len(dial_id_batch), dial_id_batch, return_reward=True)
                    avg_turn+=sum([len(dial) for dial in gen_batch])
                    avg_US_reward+=sum([np.mean(reward) for reward in US_reward_batch])
                    avg_DS_reward+=sum([np.mean(reward) for reward in DS_reward_batch])
                    success+=batch_success
                    match+=batch_match
                    goal_comp+=batch_comp
                    all_dials+=gen_batch
            else:
                for init_goal_batch in init_goal_batches:
                    total+=len(init_goal_batch)
                    gen_batch, US_reward_batch, DS_reward_batch, batch_success, batch_match, batch_comp, _, _=\
                        self.sess.interact_by_batch(DS, US, len(init_goal_batch), init_goal_batch=init_goal_batch, return_reward=True)
                    avg_turn+=sum([len(dial) for dial in gen_batch])
                    avg_US_reward+=sum([np.mean(reward) for reward in US_reward_batch])
                    avg_DS_reward+=sum([np.mean(reward) for reward in DS_reward_batch])
                    success+=batch_success
                    match+=batch_match
                    goal_comp+=batch_comp
                    all_dials+=gen_batch
        success/=total
        match/=total
        goal_comp/=total
        avg_US_reward/=total
        avg_DS_reward/=total
        avg_turn/=total
        logging.info('Validation dialogs:{},  time:{:.2f} min'.format(total, (time.time()-st)/60))
        logging.info('Avg_US_reward:{:3f}, avg_DS_reward:{:3f}, avg turns:{}, success rate:{:.4f}, match rate:{:.4f}, goal complete rate:{:.4f}'.format(avg_US_reward, avg_DS_reward, avg_turn, success, match, goal_comp))
        if os.path.exists(cfg.exp_path):
            json.dump(all_dials, open(os.path.join(cfg.exp_path, cfg.result_file), 'w'), indent=2)
        return avg_DS_reward, avg_US_reward, avg_turn, success, match

    def save_model(self, DS, US, last_model=False):
        if last_model:
            DS.save_pretrained(os.path.join(cfg.exp_path,'last_epoch_DS'))
            self.DS_tok.save_pretrained(os.path.join(cfg.exp_path,'last_epoch_DS'))
            US.save_pretrained(os.path.join(cfg.exp_path,'last_epoch_US'))
            self.US_tok.save_pretrained(os.path.join(cfg.exp_path,'last_epoch_US'))
        else:
            if cfg.joint_train_ds:
                DS.save_pretrained(os.path.join(cfg.exp_path,'best_DS'))
                self.DS_tok.save_pretrained(os.path.join(cfg.exp_path,'best_DS'))
            if cfg.joint_train_us:
                US.save_pretrained(os.path.join(cfg.exp_path,'best_US'))
                self.US_tok.save_pretrained(os.path.join(cfg.exp_path,'best_US'))

    def get_optimizers(self, num_dials, model):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr)
        num_training_steps = num_dials*cfg.epoch_num // cfg.training_batch_size
        num_warmup_steps = cfg.warmup_steps if cfg.warmup_steps >= 0 else int(num_training_steps*cfg.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,\
            num_training_steps=num_training_steps) if cfg.use_scheduler else None
        return optimizer, scheduler

    def compute_resp_prob(self, data):
        sys_model=GPT2LMHeadModel.from_pretrained('experiments_21/all_sys-model_sd11_lr0.0001_bs16_ga2/best_loss_model')
        # the tokenizer of sys_model should be the same as that of self.model
        all_batches, seq_num = self.reader.get_sys_batch(data, batch_size=16, mode='test')
        total_log_prob=0
        with torch.no_grad():
            for batch in all_batches:
                input_batch=torch.from_numpy(batch).long().to(sys_model.device)
                output_batch=sys_model(input_batch)
                loss = self.calculate_loss_and_accuracy(output_batch, input_batch)
                avg_log_prob=-loss.item()
                total_log_prob+=avg_log_prob
        return total_log_prob/len(all_batches)
    
    def calculate_loss(self, outputs, labels, rewards, base=None):
        # logits: B, T, V
        # labels: B, T
        # rewards: B
        batch_size=labels.size(0)
        lm_logits = outputs[0]

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=cfg.pad_id, reduction='sum')
        loss=0
        for i in range(batch_size):
            logit=shift_logits[i,:,:]
            label=shift_labels[i,:]
            if base:
                reward=rewards[i]-base
            else:
                reward=rewards[i]
            loss += reward*loss_fct(logit.view(-1, logit.size(-1)), label.view(-1))

        # avg loss
        not_ignore = shift_labels.ne(cfg.pad_id)
        num_targets = not_ignore.long().sum().item()

        loss /= num_targets
        return loss
    
    def handle_ABUS_dialog(self, dial_batch_dict):
        dial_batch=[]
        reward_batch=[]
        total_success, total_match=0, 0
        for dial_id, dial in dial_batch_dict.items():
            dial_batch.append(dial['log'])
            success, match=self.evaluator.get_metrics(dial['goal'], dial['log'])
            reward_batch.append([success]*len(dial['log']))
            total_success+=success
            total_match+=match
        return dial_batch, reward_batch, (total_success, total_match)

def fix_seed():
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()

    parse_arg_cfg(args)
    cfg.mode=args.mode
    if cfg.seed!=11:# not default seed, change the exp name
        cfg.exp_no+='-seed{}'.format(cfg.seed)
    cfg.exp_path=os.path.join(cfg.rl_save_path,cfg.exp_no)
    if not os.path.exists(cfg.exp_path):
        os.mkdir(cfg.exp_path)
    if 'test' in args.mode:
        cfg.eval_load_path=cfg.DS_path
    cfg._init_logging_handler(args.mode)
    cfg.rl_train=True
    if not cfg.rl_iterate:
        cfg.rl_iterate_num=0
    if cfg.interact_with_ABUS:
        cfg.joint_train_us=False
        cfg.iterative_update=False
    fix_seed()
    sess=turn_level_session(cfg.DS_path, cfg.US_path, cfg.DS_device, cfg.US_device)
    session=RL_session(sess)
    if 'train' in args.mode:
        session.run_RL()
    else:
        cfg.exp_path=cfg.DS_path
        if cfg.goal_path!='':
            logging.info(cfg.goal_path)
            goal_list=json.load(open(cfg.goal_path, 'r'))
            session.evaluation(goal_list)
        else:
            session.evaluation()