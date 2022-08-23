from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from eval import MultiWozEvaluator
from reader import MultiWozReader
import utils
from torch.optim import Adam
import torch
import torch.nn as nn
import torch.nn.functional as F
from mwzeval.metrics import Evaluator
import math, copy

import os
import shutil
import random
import argparse
import time
import logging
import json
import numpy as np
from compute_joint_acc import compute_jacc
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config import global_config as cfg
class Model(object):
    
    def __init__(self, device=[0]):
        self.device=device[0]
        tokenizer_path=cfg.gpt_path
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        # cfg.tokenizer = tokenizer

        # initialize multiwoz reader
        self.reader = MultiWozReader(self.tokenizer)
        self.get_special_ids()
        logging.info([self.sos_b_id, self.sos_a_id, self.sos_r_id, self.eos_b_id, self.eos_a_id,self.eos_r_id])

        # create model: gpt2
        self.model=GPT2LMHeadModel.from_pretrained(cfg.gpt_path)
        self.model.resize_token_embeddings(len(self.tokenizer))
        if cfg.gradient_checkpoint:
            self.model.config.gradient_checkpointing=True
        
        self.model.to(self.device)
        logging.info("Model loaded from {}".format(cfg.gpt_path))

        self.evaluator = MultiWozEvaluator(self.reader)
        self.std_evaluator=Evaluator(bleu=1, success=1, richness=0)
        if cfg.save_log:
            log_path='./log21/log_{}'.format(cfg.exp_no) if cfg.dataset==1 else './log/log_{}'.format(cfg.exp_no)
            if os.path.exists(log_path):
                shutil.rmtree(log_path)
                os.mkdir(log_path)
            else:
                os.mkdir(log_path)
            self.tb_writer = SummaryWriter(log_dir=log_path)
        else:
            self.tb_writer = None
        cfg.origin_batch_size=cfg.batch_size

        self.nll_loss=nn.NLLLoss(ignore_index=cfg.pad_id)
        self.eps=1e-45
        if 'test' not in cfg.mode:
            json.dump(cfg.__dict__,open(os.path.join(cfg.exp_path,'cfg_all.json'),'w'),indent=2)
        self.global_output=4
    
    def get_special_ids(self):
        self.sos_b_id=self.tokenizer.convert_tokens_to_ids('<sos_b>')
        self.sos_a_id=self.tokenizer.convert_tokens_to_ids('<sos_a>')
        self.sos_r_id=self.tokenizer.convert_tokens_to_ids('<sos_r>')
        self.eos_b_id=self.tokenizer.convert_tokens_to_ids('<eos_b>')
        self.eos_a_id=self.tokenizer.convert_tokens_to_ids('<eos_a>')
        self.eos_r_id=self.tokenizer.convert_tokens_to_ids('<eos_r>')
        self.sos_db_id=self.tokenizer.convert_tokens_to_ids('<sos_db>')
        self.eos_db_id=self.tokenizer.convert_tokens_to_ids('<eos_db>')
        self.sos_u_id=self.tokenizer.convert_tokens_to_ids('<sos_u>')
        self.eos_u_id=self.tokenizer.convert_tokens_to_ids('<eos_u>')
        if cfg.train_us:
            self.sos_g_id=self.tokenizer.convert_tokens_to_ids('<sos_g>')
            self.eos_g_id=self.tokenizer.convert_tokens_to_ids('<eos_g>')
            self.sos_ua_id=self.tokenizer.convert_tokens_to_ids('<sos_ua>')
            self.eos_ua_id=self.tokenizer.convert_tokens_to_ids('<eos_ua>')

    def pretrain_turn_level(self):
        num_dials=len(self.reader.train)
        all_batches = self.reader.get_batches('train')
        
        set_stats = self.reader.set_stats['train']
        num_turns=set_stats['num_turns']
        optimizer, scheduler = self.get_sep_optimizers(num_turns,self.model)

        # log info
        logging.info("***** Running turn-level training *****")
        logging.info("  Num Turns = %d", set_stats['num_turns'])
        logging.info("  Num Dialogs = %d", set_stats['num_dials'])
        logging.info("  Num Epochs = %d", cfg.epoch_num)

        log_inputs = 6
        global_step = 0

        min_loss = 1000
        min_eval_loss=1000
        max_score=0
        early_stop_count=cfg.early_stop_count
        #epoch_th=-1
        epoch_th=0.2*cfg.epoch_num if 'distilgpt2' in cfg.gpt_path else -1
        warmup_epochs=cfg.warmup_steps*cfg.gradient_accumulation_steps*cfg.batch_size//num_dials \
            if cfg.warmup_steps>=0 else int(cfg.epoch_num*cfg.warmup_ratio)
        c1, c2=0,0
        for epoch in range(cfg.epoch_num):
            epoch_step = 0
            tr_loss = 0.0
            btm = time.time()
            oom_time = 0
            self.model.zero_grad()
            random.shuffle(all_batches)
            #data_iterator = self.reader.get_data_iterator(all_batches)

            for batch_idx, batch0 in enumerate(all_batches):
                dial_batch=self.reader.transpose_batch(batch0)
                pv_batch = None
                c1+=1
                for turn_num, turn_batch in enumerate(dial_batch):
                    if turn_num==0:
                        init_goals=turn_batch['goal'] if 'goal' in turn_batch else turn_batch['gpan']
                    c2+=1
                    first_turn = (turn_num == 0)
                    side='user' if cfg.train_us else 'sys'
                    if cfg.full_goal:
                        inputs, labels = self.reader.convert_batch_turn(
                            turn_batch, pv_batch, first_turn, side=side, init_goals=init_goals)
                    else:
                        inputs, labels = self.reader.convert_batch_turn(
                            turn_batch, pv_batch, first_turn, side=side)
                    if cfg.train_us:
                        pv_batch = self.reader.get_pv_batch(pv_batch, resp=turn_batch['resp'], 
                            aspn=turn_batch['aspn'], goal=turn_batch['gpan'], user_act=turn_batch['usr_act'], side='user') 
                    else:
                        pv_batch = self.reader.get_pv_batch(pv_batch, user=turn_batch['user'],
                            resp=turn_batch['resp'], bspn=turn_batch['bspn'], side='sys')
                    try:  # avoid OOM
                        self.model.train()
                        if log_inputs > 0:  # log inputs for the very first two turns
                            logging.info('Input examples:\n{}'.format(self.tokenizer.decode(inputs['contexts'][0])))
                            log_inputs-=1
                        inputs = self.add_torch_input(inputs)
                        outputs = self.model(inputs['contexts_tensor'])
                        if cfg.only_target_loss:
                            labels=self.add_torch_input(labels)    
                            loss = self.calculate_loss_and_accuracy(outputs, labels=labels['contexts_tensor'])
                        else:
                            loss = self.calculate_loss_and_accuracy(outputs, labels=inputs['contexts_tensor'])
                        loss.backward()
                        tr_loss += loss.item()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                        epoch_step += 1

                        # step, wrt gradient_accumulation_steps, clip grad norm
                        if epoch_step % cfg.gradient_accumulation_steps == 0 or(
                            batch_idx==len(all_batches)-1 and turn_num==len(dial_batch)-1):
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                            # global_step: actual step the optimizer took
                            global_step += 1
                            

                    except RuntimeError as exception:
                        if "out of memory" in str(exception):
                            max_length = max(inputs['lengths'])
                            oom_time += 1
                            logging.info("WARNING: ran out of memory,times: {}, batch size: {}, max_len: {}".format(
                                oom_time, cfg.batch_size, max_length))
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                        else:
                            logging.info(str(exception))
                            raise exception
            if epoch==0:
                logging.info('Num dials:{}, num_turns:{}'.format(c1, c2))
            logging.info('Epoch:{}, Train epoch time: {:.2f} min, epoch loss: {:.4f}'.format(
                epoch, (time.time()-btm)/60, tr_loss))
            if cfg.evaluate_during_training:
                if cfg.save_type=='min_loss':
                    eval_loss=self.eval(model=self.model)
                    logging.info('model evaluation loss:{}'.format(eval_loss))
                    if self.tb_writer:
                        self.tb_writer.add_scalar('loss_eval',eval_loss,epoch)
                    if eval_loss<min_eval_loss:
                        min_eval_loss=eval_loss
                        self.save_model(path='best_loss_model',model=self.model)
                        early_stop_count=cfg.early_stop_count
                    else:
                        if epoch>=warmup_epochs:#early stop after warm up
                            early_stop_count-=1
                            logging.info('early stop count:%d'%early_stop_count)
                            if early_stop_count==0 and cfg.early_stop:
                                logging.info('early stopped')
                                break
                elif cfg.save_type=='max_score' and epoch>epoch_th:
                    if cfg.train_us:
                        bleu, P, R, F1=self.validate_us(data='dev')
                        self.tb_writer.add_scalar('P',P,epoch)
                        self.tb_writer.add_scalar('R',R,epoch)
                        self.tb_writer.add_scalar('F1',F1,epoch)
                        self.tb_writer.add_scalar('bleu',bleu,epoch)
                        score=F1*100
                    else:
                        eval_result=self.validate_fast(data='dev')
                        self.tb_writer.add_scalar('joint_goal',eval_result['joint_acc'],epoch)
                        self.tb_writer.add_scalar('match',eval_result['match'],epoch)
                        self.tb_writer.add_scalar('success',eval_result['success'],epoch)
                        self.tb_writer.add_scalar('bleu',eval_result['bleu'],epoch)
                        self.tb_writer.add_scalar('combined_score',eval_result['score'],epoch)
                        score=eval_result['score']
                    if score>max_score:
                        early_stop_count=cfg.early_stop_count
                        max_score=score
                        self.save_model(path='best_score_model',model=self.model)
                    else:
                        if epoch>=warmup_epochs:
                            early_stop_count-=1
                            logging.info('early stop count:%d'%early_stop_count)
                            if early_stop_count==0 and cfg.early_stop:
                                logging.info('early stopped')
                                break
            else:#save the model with minimal training loss
                pass

    def get_sep_optimizers(self, num_dials, model, num_batches=None):
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
        if not num_batches:
            num_training_steps = num_dials*cfg.epoch_num // (cfg.gradient_accumulation_steps*cfg.origin_batch_size)
        else:
            num_training_steps = num_batches*cfg.epoch_num
        num_warmup_steps = cfg.warmup_steps if cfg.warmup_steps >= 0 else int(num_training_steps*cfg.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,\
            num_training_steps=num_training_steps) if cfg.use_scheduler else None
        logging.info('Training steps:{}, warmup steps:{}, steps per epoch:{}'.format(num_training_steps, 
            num_warmup_steps, num_batches))
        return optimizer, scheduler

    def add_torch_input(self, inputs):
        # to tensor and to device
        contexts_tensor = torch.from_numpy(inputs['contexts_np']).long()
        contexts_tensor = contexts_tensor.to(self.device, non_blocking=True)
        inputs['contexts_tensor'] = contexts_tensor
        return inputs


    def calculate_loss_and_accuracy(self, outputs, labels):
        # GPT2-chicahat/train.py
        lm_logits = outputs[0]

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        pad_id = cfg.pad_id
        loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # avg loss
        not_ignore = shift_labels.ne(pad_id)
        num_targets = not_ignore.long().sum().item()

        loss /= num_targets
        return loss

    def convert_eval_batch(self, data, contexts, turn_num,bs_gen,prior=False,db_gen=None,resp_gen=None,aspn_gen=None, gen_db=False):
        
        if gen_db:#在使用后验网络生成数据库结果时使用
            new_contexts=[]
            for id, context in enumerate(contexts):
                new_contexts.append(context[:-1] + bs_gen[id] + [self.sos_db_id])
            return new_contexts
        else:
            for id,context in enumerate(contexts):
                if turn_num==0:
                    if prior:
                        if db_gen is None:#还没有生成bs_gen以及对应的db_gen
                            contexts[id]=data[id][turn_num]['user']+[self.sos_b_id]
                        else:
                            sos_id=self.sos_a_id if cfg.model_act else self.sos_r_id
                            contexts[id]=context[:-1]+bs_gen[id]+db_gen[id]+[sos_id]
                    else:
                        if db_gen is None:
                            contexts[id]=data[id][turn_num]['user']+data[id][turn_num]['resp']+[self.sos_b_id]
                        else:
                            contexts[id]= context[:-1] + bs_gen[id]+ db_gen[id] + [self.sos_a_id]
                else:
                    #context中已经含有sos_b了
                    if prior:
                        if resp_gen is None:
                            sos_id=self.sos_a_id if cfg.model_act else self.sos_r_id
                            contexts[id]=context[:-1] +bs_gen[id]+db_gen[id]+[sos_id]
                        else:
                            contexts[id]=context[:-1] + resp_gen[id] + data[id][turn_num]['user']+[self.sos_b_id]
                    else:
                        if resp_gen is None:
                            if cfg.model_act:
                                contexts[id]=context[:-1] + bs_gen[id] + db_gen[id] + [self.sos_a_id]#to generate aspn
                            else:
                                contexts[id]=context[:-1] + bs_gen[id] +[self.sos_r_id]
                        else:
                            if cfg.model_act:
                                contexts[id]=context[:-1] + aspn_gen[id] + data[id][turn_num]['user']\
                                    +data[id][turn_num]['resp']+[self.sos_b_id]#to generate bspn
                            else:
                                contexts[id]=context[:-1]+data[id][turn_num]['user']+data[id][turn_num]['resp']+[self.sos_b_id]
            return contexts


    def get_bspn(self,bs_tensor, return_db=False, turn_domain=None, bs_prob=None):
        # return_db: return db results of bspn
        # turn_domain: a list of domain
        if not isinstance(bs_tensor,list):
            bs_batch=bs_tensor.cpu().tolist()
            if bs_prob is not None:
                bs_prob=bs_prob.cpu().tolist()
        else:
            bs_batch=bs_tensor
        bs_gen=[]
        db_gen=[]
        bp_gen=[]
        bs_ex_gen=[]
        eos_b_id=self.eos_b_id
        sos_b_id=self.sos_b_id
        for i,bs in enumerate(bs_batch):
            if bs_prob:
                prob=bs_prob[i]
            if eos_b_id in bs:
                idx=bs.index(eos_b_id)+1
                bs=[sos_b_id]+bs[:idx]
                if bs_prob:
                    prob=[0]+prob[:idx]
            else:
                bs[-1]=eos_b_id
                bs=[sos_b_id]+bs
                if bs_prob:
                    prob[-1]=1
                    prob=[0]+prob
            if bs.count(sos_b_id)>1:
                last=bs[::-1].index(sos_b_id)+1
                bs=bs[-last:]
                if bs_prob:
                    prob=prob[-last:]
            
            if cfg.sys_nlu: # the generated bspn contain not only informable slots but also requestable slots
                bs_ex_gen.append(bs)
                bs=self.reader.delete_reqt_in_bspn(bs, mode='id')
            bs_gen.append(bs)
            if bs_prob:
                assert len(bs)==len(prob)
                bp_gen.append(prob)

            if return_db:
                db_result=self.reader.bspan_to_DBpointer(self.tokenizer.decode(bs), turn_domain[i])
                db = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('<sos_db> '+ db_result + ' <eos_db>'))
                db_gen.append(db)
        return bs_gen, db_gen, bs_ex_gen, bp_gen

    def get_aspn(self,aspn_tensor):
        if not isinstance(aspn_tensor, list):
            aspn_batch=aspn_tensor.cpu().tolist()
        else:
            aspn_batch=aspn_tensor
        aspn_gen=[]
        eos_a_id=self.eos_a_id
        sos_a_id=self.sos_a_id
        for i ,aspn in enumerate(aspn_batch):
            if eos_a_id in aspn:
                aspn=[sos_a_id]+aspn[:aspn.index(eos_a_id)+1]
            else:
                aspn[-1]=eos_a_id
                aspn=[sos_a_id]+aspn
            if aspn.count(sos_a_id)>1:
                last=aspn[::-1].index(sos_a_id)+1
                aspn=aspn[-last:]
            aspn_gen.append(aspn)
        return aspn_gen

    def get_resp(self,resp_tensor, resp_prob=None):
        resp_batch=resp_tensor.cpu().tolist()
        if resp_prob is not None:
            resp_prob=resp_prob.cpu().tolist()
        resp_gen=[]
        rp_gen=[]
        eos_r_id=self.eos_r_id
        sos_r_id=self.sos_a_id if cfg.model_act else self.sos_r_id
        for i,resp in enumerate(resp_batch):
            if resp_prob:
                prob=resp_prob[i]
            if eos_r_id in resp:
                idx=resp.index(eos_r_id)+1
                resp=[sos_r_id]+resp[:idx]
                if resp_prob:
                    prob=[0]+prob[:idx]
            else:
                resp[-1]=eos_r_id
                resp=[sos_r_id]+resp
                if resp_prob:
                    prob[-1]=1
                    prob=[0]+prob
            if resp.count(sos_r_id)>1:
                last=resp[::-1].index(sos_r_id)+1
                resp=resp[-last:]
                if resp_prob:
                    prob=prob[-last:]
            resp_gen.append(resp)
            if resp_prob:
                assert len(resp)==len(prob)
                rp_gen.append(prob)
        if resp_prob:
            return resp_gen, rp_gen
        return resp_gen
    
    def get_user(self,u_tensor):
        u_batch=u_tensor.cpu().tolist()
        u_gen=[]

        for i ,u in enumerate(u_batch):
            if self.eos_u_id in u:
                u=[self.sos_ua_id]+u[:u.index(self.eos_u_id)+1]
            else:
                u[-1]=self.eos_u_id
                u=[self.sos_ua_id]+u
            if u.count(self.sos_ua_id)>1:
                last=u[::-1].index(self.sos_ua_id)+1
                u=u[-last:]
            u_gen.append(u)
        return u_gen

    def get_turn_domain(self, turn_domain_batch, bs_batch, pv_bs_batch=None):

        db_batch=[]
        for i, bspn in enumerate(bs_batch):
            bspn_tokens=self.tokenizer.decode(bspn)
            cons=self.reader.bspan_to_constraint_dict(bspn_tokens)
            cur_domain=list(cons.keys())
            if len(cur_domain)==0:
                db_result = self.tokenizer.encode('<sos_db> [db_nores] <eos_db>')
            else:
                if len(cur_domain)==1:
                    turn_domain_batch[i]=cur_domain
                else:
                    if pv_bs_batch is None:
                        max_slot_num=0 # We choose the domain with most slots as the current domain
                        for domain in cur_domain:
                            if len(cons[domain])>max_slot_num:
                                turn_domain_batch[i]=[domain]
                                max_slot_num=len(cons[domain])
                    else:
                        pv_domain=list(self.reader.bspan_to_constraint_dict(self.tokenizer.decode(pv_bs_batch[i])).keys())
                        for domain in cur_domain:
                            if domain not in pv_domain: # new domain
                                # if domains are all the same, self.domain will not change
                                turn_domain_batch[i]=[domain]
                db_result = self.reader.bspan_to_DBpointer(bspn_tokens, turn_domain_batch[i]) #[db_x]
                db_result = self.tokenizer.encode('<sos_db> '+ db_result + ' <eos_db>')
            db_batch.append(db_result)
        return turn_domain_batch, db_batch

    def save_model(self, path=None, model=None):
        if not path:
            save_path = os.path.join(cfg.exp_path, 'best_model')
        else:
            save_path = os.path.join(cfg.exp_path, path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logging.info('Saving model checkpoint to %s', save_path)
        if not model:
            self.model.save_pretrained(save_path)
        else:
            model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        # save cfg
    
    def eval(self,data='dev',model=None):
        model.eval()
        all_batches = self.reader.get_batches(data)
        total_batch=len(all_batches)
        total_loss=0
        with torch.no_grad():
            data_iterator = self.reader.get_data_iterator(all_batches)
            for batch_idx, dial_batch in enumerate(data_iterator):
                pv_batch = None
                for turn_num, turn_batch in enumerate(dial_batch):
                    if turn_num==0:
                        init_goals=turn_batch['goal'] if 'goal' in turn_batch else turn_batch['gpan']
                    first_turn = (turn_num == 0)
                    side='user' if cfg.train_us else 'sys'
                    if cfg.full_goal:
                        inputs, labels = self.reader.convert_batch_turn(
                            turn_batch, pv_batch, first_turn, side=side, init_goals=init_goals)
                    else:
                        inputs, labels = self.reader.convert_batch_turn(turn_batch, pv_batch, first_turn, side=side)
                    if cfg.train_us:
                        pv_batch = self.reader.get_pv_batch(pv_batch, resp=turn_batch['resp'], 
                            aspn=turn_batch['aspn'], goal=turn_batch['gpan'], user_act=turn_batch['usr_act'], side='user') 
                    else:
                        pv_batch = self.reader.get_pv_batch(pv_batch, user=turn_batch['user'],
                            resp=turn_batch['resp'], bspn=turn_batch['bspn'], side='sys')
                    inputs=self.add_torch_input(inputs)#B,T
                    labels=self.add_torch_input(labels)#B,T
                    outputs = model(inputs['contexts_tensor'])
                    loss=self.calculate_loss_and_accuracy(outputs,labels['contexts_tensor'])
                    total_loss+=loss.item()
        return total_loss/total_batch


    def validate_fast(self, data='dev', dial_id_list=None):
        
        self.model.eval()
        eval_data = self.reader.get_eval_data(data)
        if cfg.debugging:
            eval_data=eval_data[:32]
        origin_batch_size=cfg.batch_size
        cfg.batch_size=cfg.eval_batch_size
        batches=self.reader.get_batches('test',data=eval_data)
        result_path=os.path.join(cfg.eval_load_path,'result.json')
        
        if os.path.exists(result_path) and cfg.mode=='test' and cfg.use_existing_result:
            #results,field=self.reader.load_result(result_path)
            results=json.load(open(result_path, 'r'))
            joint_acc=compute_jacc(results)
            input_data=self.prepare_for_std_eval(data=results)
            cfg.use_true_bspn_for_ctr_eval=False
            bleu, success, match = self.evaluator.validation_metric(results)
            score = 0.5 * (success + match) + bleu
            logging.info('[Old] validation %2.2f  %2.2f  %2.2f  %.2f  %.3f' % (match, success, bleu, score, joint_acc))
            if self.std_evaluator:
                std_metrics = self.std_evaluator.evaluate(input_data)
                bleu=std_metrics['bleu']['damd']
                match=std_metrics['success']['inform']['total']
                success=std_metrics['success']['success']['total']
                score = 0.5 * (success + match) + bleu
                logging.info(std_metrics)
                logging.info('[std] validation %2.2f  %2.2f  %2.2f  %.2f  %.3f' % (match, success, bleu, score, joint_acc))

            eval_results = {}
            eval_results['bleu'] = bleu
            eval_results['success'] = success
            eval_results['match'] = match
            eval_results['score'] = score
            eval_results['joint_acc']=joint_acc
            return eval_results
        
        # valid_losses = []
        result_collection = {}
        st=time.time()
        for batch in batches:
            try:
                if batch==[]:
                    continue
                if cfg.turn_level:
                    batch=self.generate_batch_turn_level(batch)
                else:
                    batch=self.generate_batch_session_level(batch)
                for dialog in batch:
                    result_collection.update(self.reader.inverse_transpose_turn(dialog))
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    logging.info("WARNING: ran out of memory during validation and batch will be divided by half, batch size:{}, turn num:{}"\
                        .format(len(batch),len(batch[0])))
                    if hasattr(torch.cuda, 'empty_cache'):
                        with torch.cuda.device(self.device):
                            torch.cuda.empty_cache()
                    #divide the batch in half if out of memory
                    batches.insert(0,batch[:len(batch)//2])
                    batches.insert(1,batch[len(batch)//2:])
                else:
                    logging.info(str(exception))
                    raise exception
        results, _ = self.reader.wrap_result_lm(result_collection)
        logging.info('Inference time:{:.3f} min'.format((time.time()-st)/60))

        joint_acc=compute_jacc(results)
        input_data=self.prepare_for_std_eval(data=results)
        cfg.use_true_bspn_for_ctr_eval=False
        bleu, success, match = self.evaluator.validation_metric(results)
        score = 0.5 * (success + match) + bleu
        logging.info('[Old] validation %2.2f  %2.2f  %2.2f  %.2f  %.3f' % (match, success, bleu, score, joint_acc))
        if self.std_evaluator:
            std_metrics = self.std_evaluator.evaluate(input_data)
            bleu=std_metrics['bleu']['damd']
            match=std_metrics['success']['inform']['total']
            success=std_metrics['success']['success']['total']
            score = 0.5 * (success + match) + bleu
            if cfg.mode=='test':
                logging.info(std_metrics)
            
            logging.info('[Std] validation %2.2f  %2.2f  %2.2f  %.2f  %.3f' % (match, success, bleu, score, joint_acc))
        if data=='train':# divide into 10 parts
            step=len(results)//10
            for i in range(10):
                res_path=os.path.join(cfg.eval_load_path, 'result_{}.json'.format(i))
                json.dump(results[i*step:(i+1)*step], open(res_path, 'w'), indent=2)
        else:
            json.dump(results, open(result_path, 'w'), indent=2)
        #self.reader.save_result('w', results, field,result_name='result.csv')

        eval_results = {}
        eval_results['bleu'] = bleu
        eval_results['success'] = success
        eval_results['match'] = match
        eval_results['score'] = score
        eval_results['joint_acc']=joint_acc
        cfg.batch_size=origin_batch_size
        return eval_results

    def generate_batch_turn_level(self, batch):
        
        batch=self.reader.transpose_batch(batch)

        bs_max_len=75
        resp_max_len=80
        sos_b_id=self.sos_b_id
        eos_b_id=self.eos_b_id
        sos_r_id=self.sos_r_id
        eos_a_id=self.eos_a_id
        eos_r_id=self.eos_r_id

        batch_size=len(batch[0]['user'])
        contexts=[[] for i in range(batch_size)]
        bs_gen=[]
        db_gen=[]
        resp_gen=[]
        bs_prob=None
        bs_prob_t=[0]*40
        pv_batch=None
        pv_bspn_batch=None
        turn_domain_batch=[[] for _ in range(batch_size)]

        device=self.device
        with torch.no_grad():
            for turn_num, turn_batch in enumerate(batch):
                # generate bspn
                past_key_values=None
                end_flag=np.zeros(batch_size)
                contexts=self.reader.convert_eval_batch_turn(turn_batch,pv_batch, mode='gen_bspn')
                
                if self.global_output>0 and cfg.mode=='test':
                    logging.info(self.tokenizer.decode(contexts[0]))
                    self.global_output-=1
                
                inputs,attentions=self.reader.batch_align(contexts,left_len=bs_max_len,return_attn=True)
                inputs=torch.tensor(inputs).to(device)
                attentions=torch.tensor(attentions).to(device)
                for i in range(bs_max_len):
                    position_ids = attentions.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attentions == 0, 1)
                    if past_key_values is not None:
                        position_ids=position_ids[:, -1].unsqueeze(-1)
                    outputs=self.model(inputs,attention_mask=attentions,position_ids=position_ids,\
                            return_dict=True,use_cache=True,past_key_values=past_key_values)

                    past_key_values=outputs.past_key_values
                    maximum=F.softmax(outputs.logits[:, -1, :], -1).max(-1)

                    #preds=outputs.logits[:,-1,:].argmax(-1)#B
                    preds=maximum.indices
                    probs=maximum.values
                    if i==0:
                        bs_tensor=preds.unsqueeze(1)
                    else:
                        bs_tensor=torch.cat([bs_tensor,preds.unsqueeze(1)],dim=1)
                    attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(device)),dim=1)
                    inputs=preds.unsqueeze(1)
                    end_flag+=(preds.cpu().numpy()==eos_b_id).astype(float)
                    if sum(end_flag==0)==0:
                        break
                if cfg.use_true_domain_for_ctr_eval:
                    bs_gen, db_gen, bs_ex_gen, bp_gen=self.get_bspn(bs_tensor,return_db=True, turn_domain=turn_batch['turn_domain'], bs_prob=bs_prob)
                else:
                    bs_gen, db_gen, bs_ex_gen, bp_gen=self.get_bspn(bs_tensor, bs_prob=bs_prob)
                    turn_domain_batch, db_gen=self.get_turn_domain(turn_domain_batch, bs_gen, pv_bspn_batch)
                # generate aspn and resp
                past_key_values=None
                end_flag=np.zeros(batch_size)
                if cfg.sys_nlu:
                    contexts=self.reader.convert_eval_batch_turn(turn_batch,pv_batch, mode='gen_ar', bspn_gen=bs_ex_gen,db_gen=db_gen)
                else:
                    contexts=self.reader.convert_eval_batch_turn(turn_batch,pv_batch, mode='gen_ar', bspn_gen=bs_gen,db_gen=db_gen)
                
                if self.global_output>0 and cfg.mode=='test':
                    logging.info(self.tokenizer.decode(contexts[0]))
                inputs,attentions=self.reader.batch_align(contexts,left_len=resp_max_len,return_attn=True)
                inputs=torch.tensor(inputs).to(device)
                attentions=torch.tensor(attentions).to(device)
                for i in range(resp_max_len):
                    position_ids = attentions.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attentions == 0, 1)
                    if past_key_values is not None:
                        position_ids=position_ids[:, -1].unsqueeze(-1)
                    outputs=self.model(inputs,attention_mask=attentions,position_ids=position_ids,\
                            return_dict=True,use_cache=True,past_key_values=past_key_values)
                    past_key_values=outputs.past_key_values
                    maximum=F.softmax(outputs.logits[:, -1, :], -1).max(-1)
                    #preds=outputs.logits[:,-1,:].argmax(-1)#B
                    preds=maximum.indices
                    probs=maximum.values
                    if i==0:
                        resp_tensor=preds.unsqueeze(1)
                    else:
                        resp_tensor=torch.cat([resp_tensor,preds.unsqueeze(1)],dim=1)
                    attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(device)),dim=1)
                    inputs=preds.unsqueeze(1)
                    end_flag+=(preds.cpu().numpy()==eos_r_id).astype(float)
                    if sum(end_flag==0)==0:
                        break
                resp_gen=self.get_resp(resp_tensor)
                aspn_gen=[]
                aspn_probs=[]
                resp_probs=[]
                for i, temp in enumerate(resp_gen):
                    if eos_a_id in temp:
                        idx=temp.index(eos_a_id)+1
                        aspn=temp[:idx]
                    else:
                        aspn=temp[:-1]+[eos_a_id]
                    
                    if sos_r_id in temp:
                        idx=temp.index(sos_r_id)
                        resp=temp[idx:]
                    else:
                        resp=[sos_r_id]+temp[1:]
                    resp_gen[i]=resp
                    aspn_gen.append(aspn)
                pv_batch=self.reader.get_pv_batch(pv_batch, turn_batch['user'], resp_gen, bs_gen)
                turn_batch['bspn_gen']=bs_gen
                if cfg.sys_nlu:
                    turn_batch['bspn-ex_gen']=bs_ex_gen
                turn_batch['aspn_gen']=aspn_gen
                turn_batch['resp_gen']=resp_gen
                turn_batch['db_gen']=db_gen
                pv_bspn_batch=bs_gen
        return self.reader.inverse_transpose_batch(batch)
    
    def generate_batch_us(self, batch):
        batch=self.reader.transpose_batch(batch)
        max_len=75

        batch_size=len(batch[0]['dial_id'])
        contexts=[[] for i in range(batch_size)]
        bs_gen=[]
        db_gen=[]
        resp_gen=[]
        pv_batch=None
        device=self.model.device
        self.model.eval()
        with torch.no_grad():
            for turn_num, turn_batch in enumerate(batch):
                # we generate user act and user utterance together
                past_key_values=None
                end_flag=np.zeros(batch_size)
                contexts=self.reader.convert_eval_batch_turn_us(turn_batch, pv_batch)
                
                if self.global_output>0 and cfg.mode=='test':
                    logging.info(self.tokenizer.decode(contexts[0]))
                    self.global_output-=1
                
                inputs,attentions=self.reader.batch_align(contexts,left_len=max_len,return_attn=True)
                inputs=torch.tensor(inputs).to(device)
                attentions=torch.tensor(attentions).to(device)
                for i in range(max_len):
                    position_ids = attentions.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attentions == 0, 1)
                    if past_key_values is not None:
                        position_ids=position_ids[:, -1].unsqueeze(-1)
                    outputs=self.model(inputs,attention_mask=attentions,position_ids=position_ids,\
                            return_dict=True,use_cache=True,past_key_values=past_key_values)

                    past_key_values=outputs.past_key_values
                    preds=outputs.logits[:,-1,:].argmax(-1)#B
                    if i==0:
                        u_tensor=preds.unsqueeze(1)
                    else:
                        u_tensor=torch.cat([u_tensor,preds.unsqueeze(1)],dim=1)
                    attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(device)),dim=1)
                    inputs=preds.unsqueeze(1)
                    end_flag+=(preds.cpu().numpy()==self.eos_u_id).astype(float)
                    if sum(end_flag==0)==0:
                        break
                u_gen=self.get_user(u_tensor)
                user_gen=[]
                usr_act_gen=[]
                for i, temp in enumerate(u_gen):
                    if self.eos_ua_id in temp:
                        usr_act=temp[:temp.index(self.eos_ua_id)+1]
                    else:
                        usr_act=temp[:-1]+[self.eos_ua_id]
                    if self.sos_u_id in temp:
                        user=temp[temp.index(self.sos_u_id):]
                    else:
                        user=[self.sos_u_id]+temp[1:]
                    user_gen.append(user)
                    usr_act_gen.append(usr_act)
                pv_batch=self.reader.get_pv_batch(pv_batch, resp=turn_batch['resp'], aspn=turn_batch['aspn'], side='user')
                turn_batch['usr_act_gen']=usr_act_gen
                turn_batch['user_gen']=user_gen
        return self.reader.inverse_transpose_batch(batch)

    def validate_us(self, data='dev'):
        eval_data = self.reader.get_eval_data(data)
        if cfg.debugging:
            eval_data=eval_data[:100]
        result_path=os.path.join(cfg.eval_load_path, 'result.json')
        cfg.batch_size=cfg.eval_batch_size
        batches=self.reader.get_batches('test',data=eval_data)

        # valid_losses = []
        result_collection = []
        st=time.time()
        for batch in batches:
            try:
                if batch==[]:
                    continue
                batch=self.generate_batch_us(batch)
                result_collection+=self.reader.convert_batch_ids_to_tokens(batch)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    logging.info("WARNING: ran out of memory during validation and batch will be divided by half, batch size:{}, turn num:{}"\
                        .format(len(batch),len(batch[0])))
                    if hasattr(torch.cuda, 'empty_cache'):
                        with torch.cuda.device(self.device):
                            torch.cuda.empty_cache()
                    #divide the batch in half if out of memory
                    batches.insert(0,batch[:len(batch)//2])
                    batches.insert(1,batch[len(batch)//2:])
                else:
                    logging.info(str(exception))
                    raise exception
        bleu, P, R, F1=self.evaluator.evaluate_us(result_collection)
        logging.info('BLEU:{:.2f}, Avg_Precious:{:.3f}, Avg_Recall:{:.3f}, Avg_F1:{:.3f}'.format(
            bleu, P, R, F1
        ))
        logging.info('Evaluation time:{:.2f} min'.format((time.time()-st)/60))
        json.dump(result_collection, open(result_path, 'w'), indent=2)
        return bleu, P, R, F1

    def prepare_for_std_eval(self, path=None, data=None):
        if path:
            data=json.load(open(path, 'r', encoding='utf-8'))
        new_data={}
        dials=self.evaluator.pack_dial(data)
        for dial_id in dials:
            new_data[dial_id]=[]
            dial=dials[dial_id]
            for turn in dial:
                if turn['user']=='':
                    continue
                entry={}
                entry['response']=turn['resp_gen']
                entry['state']=self.reader.bspan_to_constraint_dict(turn['bspn_gen'])
                new_data[dial_id].append(entry)
        if path:
            new_path=path[:-5]+'std.json'
            json.dump(new_data, open(new_path, 'w'), indent=2)
        return new_data

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

def main():
    if not os.path.exists('./experiments'):
        os.mkdir('./experiments')

    if not os.path.exists('./experiments_21'):
        os.mkdir('./experiments_21')

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()

    cfg.mode = args.mode
    parse_arg_cfg(args)
    if 'test' in args.mode:
        cfg.eval_load_path=cfg.gpt_path
    else:  # train
        #print('exp_no:',cfg.exp_no)
        if cfg.exp_path in ['', 'to be generated']:
            experiments_path = './experiments_21' if cfg.dataset==1 else './experiments'
            if cfg.exp_no=='':
                cfg.exp_no=time.strftime("%Y-%m-%d", time.localtime())
            cfg.exp_path = os.path.join(experiments_path, cfg.exp_no)
            if not os.path.exists(cfg.exp_path):
                os.mkdir(cfg.exp_path)

    cfg._init_logging_handler(args.mode)
    device=cfg.cuda_device

    # fix random seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # initialize model
    m = Model(device)

    if args.mode =='pretrain' or args.mode=='train':
        m.pretrain_turn_level()
    elif args.mode =='semi_VL':
        if cfg.train_us:
            m.semi_VL_US()
        else:
            m.semi_VL()
    elif args.mode =='semi_jsa':
        m.semi_jsa()
    elif args.mode == 'semi_ST':
        m.semi_ST()
    elif args.mode =='test_pos':
        m.validate_pos(data='test')
    else:  # test
        logging.info('Load model from :{}'.format(cfg.eval_load_path))
        m.validate_fast('test')


if __name__ == "__main__":
    main()
