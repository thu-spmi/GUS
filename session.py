import torch
import random
import numpy as np
import re
import math
import copy
import torch.nn as nn
import torch.nn.functional as F
from config import global_config as cfg
from reader import MultiWozReader
from eval import MultiWozEvaluator
from utils import modified_encode
from rl_utils.rl_util import *
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from collections import Counter
from convlab2.policy.rule.multiwoz import RulePolicy
stopwords = ['and','are','as','at','be','been','but','by', 'for','however','if', 
             'not','of','on','or','so','the','there','was','were','whatever','whether','would']
class turn_level_session(object):

    def __init__(self, DS_path, US_path=None, device1='cpu', device2='cpu', interacting=False):
        self.DS=GPT2LMHeadModel.from_pretrained(DS_path)
        self.DS.to(device1)
        if cfg.fix_DST:
            cfg.rl_for_bspn=False # do not update belief state decoder during RL
            self.DST=GPT2LMHeadModel.from_pretrained(cfg.DST_path)
            self.DST.to(device1)
        if not cfg.on_policy:#off policy
            # DS_off for updating and DS for interaction
            self.DS_off=GPT2LMHeadModel.from_pretrained(DS_path)
            self.DS_off.to(device1)
            self.DS_off.train()
        self.DS_tok=GPT2Tokenizer.from_pretrained(DS_path)
        if US_path:
            self.US=GPT2LMHeadModel.from_pretrained(US_path)
            self.US.to(device2)
            if not cfg.on_policy:
                self.US_off=GPT2LMHeadModel.from_pretrained(US_path)
                self.US_off.to(device2)
                self.US_off.train()
            self.US_tok=GPT2Tokenizer.from_pretrained(US_path)
        else:
            self.US=None
            print('No user simulator for the turn-level session')
        self.reader = MultiWozReader(self.DS_tok)
        self.evaluator=MultiWozEvaluator(self.reader)
        self.get_special_ids()
        self.end_tokens=set(['[general]', '[bye]', '[welcome]', '[thank]', '[reqmore]', '<sos_a>', '<eos_a>', '<sos_ua>', '<eos_ua>'])
        # for interaction with ABUS in ConvLab-2
        self.name='turn-level-sys'
        self.interacting=interacting
        self.init_session()
        ##
        if cfg.random_goal:
            self.user_policy = RulePolicy(character='usr')

    def get_special_ids(self):
        if hasattr(self, 'US_tok'):
            self.sos_ua_id=self.US_tok.convert_tokens_to_ids('<sos_ua>')
            self.eos_ua_id=self.US_tok.convert_tokens_to_ids('<eos_ua>')
            self.sos_u_id=self.US_tok.convert_tokens_to_ids('<sos_u>')
            self.eos_u_id=self.US_tok.convert_tokens_to_ids('<eos_u>')

        self.sos_b_id=self.DS_tok.convert_tokens_to_ids('<sos_b>')
        self.eos_b_id=self.DS_tok.convert_tokens_to_ids('<eos_b>')
        self.sos_a_id=self.DS_tok.convert_tokens_to_ids('<sos_a>')
        self.eos_a_id=self.DS_tok.convert_tokens_to_ids('<eos_a>')
        self.sos_r_id=self.DS_tok.convert_tokens_to_ids('<sos_r>')
        self.eos_r_id=self.DS_tok.convert_tokens_to_ids('<eos_r>')   

    def interact(self, goal=None, return_reward=False):
        # Initialization and restrictions
        max_turns=20
        gen_dial=[]
        # If goal is None, sample a goal from training set
        if goal is None:
            dial_id=random.sample(self.reader.train_list, 1)[0]
            if '.json' not in dial_id:
                dial_id+='.json'
            init_goal=self.reader.aspan_to_act_dict(self.reader.data[dial_id]['log'][0]['gpan'], side='user')
            final_goal=copy.deepcopy(init_goal)
            goal=init_goal
        self.turn_domain=''
        self.goal_list=[]
        for i in range(max_turns):
            turn={}
            if i==0:
                gpan='<sos_g> '+self.reader.goal_to_gpan(goal)+' <eos_g>'
                user_act, user = self.get_user_utterance(gpan, pv_resp='<sos_r> <eos_r>')
                bspn, db, aspn, resp = self.get_sys_response(user)
            else:
                pv_constraint=self.reader.bspan_to_constraint_dict(pv_bspan)
                pv_user_act_dict=self.reader.aspan_to_act_dict(pv_user_act, side='user')
                pv_sys_act_dict=self.reader.aspan_to_act_dict(pv_aspn, side='sys')
                goal=self.reader.update_goal(goal, final_goal, pv_user_act_dict, pv_sys_act_dict)# update the goal
                gpan='<sos_g> '+self.reader.goal_to_gpan(goal)+' <eos_g>'
                user_act, user = self.get_user_utterance(gpan, pv_resp=pv_resp)
                bspn, db, aspn, resp = self.get_sys_response(user, pv_bspan, pv_resp)
            self.goal_list.append(goal)
            turn['gpan'], turn['usr_act'], turn['user'], turn['bspn'], turn['db'], \
                turn['aspn'], turn['resp'] = gpan, user_act, user, bspn, db, aspn, resp
            gen_dial.append(turn)
            pv_resp=resp
            pv_bspan=bspn
            pv_aspn=aspn
            pv_user_act=user_act
            if (set(user_act.split()).issubset(self.end_tokens) and set(aspn.split()).issubset(self.end_tokens)) or goal=={}:
                break
        US_rewards=self.get_US_reward(gen_dial, self.goal_list)
        DS_rewards=self.get_DS_reward(gen_dial, init_goal)
        success, match=self.evaluator.get_metrics(init_goal,  gen_dial)
        #print('US rewards:', US_rewards)
        #print('DS rewards:', DS_rewards)
        #print('Success:', success, 'Match:', inform)
        if cfg.return_reward:
            return gen_dial, US_rewards, DS_rewards, success, match
        return gen_dial

    def get_sys_response(self, user_utter, pv_b=None, pv_resp=None, pv_aspn=None):
        user_utter=user_utter.lower()
        # First generate bspn then query for db finally genrate act and response
        bs_max_len=60
        act_max_len=20
        resp_max_len=60
        self.DS.eval()

        with torch.no_grad():
            if pv_resp is None: # first turn
                input_ids=self.reader.modified_encode(user_utter) + [self.sos_b_id]
            else:
                input_ids=self.reader.modified_encode(pv_b+pv_resp+user_utter) + [self.sos_b_id]
            max_len=1024-bs_max_len
            if len(input_ids)>max_len:
                input_ids=input_ids[-max_len:]
            context_length=len(input_ids)
            # generate belief state
            outputs = self.DS.generate(input_ids=torch.tensor([input_ids]).to(self.DS.device),
                                        pad_token_id=self.DS_tok.eos_token_id,
                                        max_length=context_length + bs_max_len, eos_token_id=self.eos_b_id)
            generated = outputs[0].cpu().numpy().tolist()
            bspn = self.DS_tok.decode(generated[context_length-1:]) #start with <sos_b>
            cons=self.reader.bspan_to_constraint_dict(bspn)
            cur_domain=list(cons.keys())
            if cur_domain==[]:
                self.turn_domain=['general']
                db_result = '<sos_db> '+ '[db_0]' + ' <eos_db>'
                db = self.DS_tok.encode(db_result)#token ids
            else:
                if len(cur_domain)==1:
                    self.turn_domain=cur_domain
                else:
                    if pv_b is None: # In rare cases, there are more than one domain in the first turn
                        max_slot_num=0 # We choose the domain with most slots as the current domain
                        for domain in cur_domain:
                            if len(cons[domain])>max_slot_num:
                                self.turn_domain=[domain]
                                max_slot_num=len(cons[domain])
                    else:
                        pv_domain=list(self.reader.bspan_to_constraint_dict(pv_b).keys())
                        for domain in cur_domain:
                            if domain not in pv_domain: # new domain
                                # if domains are all the same, self.domain will not change
                                self.turn_domain=[domain]

                #bspn=bspn.replace('portugese', 'portuguese')
                db_result = self.reader.bspan_to_DBpointer(bspn, self.turn_domain) #[db_x]
                db_result = '<sos_db> '+ db_result + ' <eos_db>'
                db = self.DS_tok.encode(db_result)#token ids
            # generate system act
            input_ids=generated + db + [self.sos_a_id]
            max_len=1024-act_max_len
            if len(input_ids)>max_len:
                input_ids=input_ids[-max_len:]
            context_length=len(input_ids)
            outputs = self.DS.generate(input_ids=torch.tensor([input_ids]).to(self.DS.device),
                                        pad_token_id=self.DS_tok.eos_token_id,
                                        max_length=context_length + act_max_len, eos_token_id=self.eos_a_id)
            generated = outputs[0].cpu().numpy().tolist()
            aspn = self.DS_tok.decode(generated[context_length-1:])
            # handle special case
            if ('not looking to make a booking' in user_utter or 'i do not need to book' in user_utter) and bspn==pv_b:
                # if user doesn't want to make a booking and no more information provided
                # then the system will ask for more information instead of offer booking
                if pv_aspn:
                    aspn=pv_aspn.replace('[offerbook]', '')
                    generated[context_length-1:]=modified_encode(self.DS_tok, aspn)
            # generate system response
            input_ids=generated + [self.sos_r_id]
            max_len=1024-resp_max_len
            if len(input_ids)>max_len:
                input_ids=input_ids[-max_len:]
            context_length=len(input_ids)
            outputs = self.DS.generate(input_ids=torch.tensor([input_ids]).to(self.DS.device),
                                        pad_token_id=self.DS_tok.eos_token_id,
                                        max_length=context_length + resp_max_len, eos_token_id=self.eos_r_id)
            generated = outputs[0].cpu().numpy().tolist()
            resp = self.DS_tok.decode(generated[context_length-1:])

        return bspn, db_result, aspn, resp

    def get_user_utterance(self, gpan, pv_resp):
        # First generate user act then user utterance
        act_max_len=25
        utter_max_len=55
        self.US.eval()

        with torch.no_grad():
            input_ids=modified_encode(self.US_tok,gpan+pv_resp) + [self.sos_ua_id]
            max_len=1024-act_max_len
            if len(input_ids)>max_len:
                input_ids=input_ids[-max_len:]
            context_length=len(input_ids)
            outputs = self.US.generate(input_ids=torch.tensor([input_ids]).to(self.US.device),
                                        pad_token_id=self.US_tok.eos_token_id,
                                        max_length=context_length + act_max_len, eos_token_id=self.eos_ua_id)
            generated = outputs[0].cpu().numpy().tolist()
            user_act = self.US_tok.decode(generated[context_length-1:]) #start with <sos_ua>

            input_ids=generated + [self.sos_u_id]
            max_len=1024-utter_max_len
            if len(input_ids)>max_len:
                input_ids=input_ids[-max_len:]
            context_length=len(input_ids)
            outputs = self.US.generate(input_ids=torch.tensor([input_ids]).to(self.US.device),
                                        pad_token_id=self.US_tok.eos_token_id,
                                        max_length=context_length + utter_max_len, eos_token_id=self.eos_u_id)
            generated = outputs[0].cpu().numpy().tolist()
            user = self.US_tok.decode(generated[context_length-1:])


        return user_act, user

    def find_best_usr_act(self, act_list, goal, pv_aspn=None):
        pv_sys_act=self.reader.aspan_to_act_dict(pv_aspn, side='sys') if pv_aspn else None
        max_reward=0
        best_act=None
        for usr_act in act_list:
            reqt_reward=0
            goal_reward=0
            token_reward=repeat_token_reward(usr_act)
            user_act=self.reader.aspan_to_act_dict(usr_act, side='user')
            if pv_sys_act:
                for domain in pv_sys_act:
                    if 'request' in pv_sys_act[domain]:
                        if domain not in user_act or 'inform' not in user_act[domain]:
                            reqt_reward-=2
                            continue
                        for slot in pv_sys_act[domain]['request']:
                            if slot in user_act[domain]['inform']:
                                reqt_reward+=1
                            else:
                                pass
                                #reqt_reward-=1
            for domain in user_act:
                for intent, sv in user_act[domain].items():
                    if domain not in goal:
                        goal_reward-=len(sv)
                        continue
                    if isinstance(sv, list):# intent=='request'
                        if 'request' not in goal[domain]:
                            goal_reward-=len(sv)
                            continue
                        for slot in sv:
                            if slot=='price' and slot not in goal[domain][intent]:
                                slot='pricerange'
                            if slot in goal[domain][intent]:
                                goal_reward+=1
                            else:
                                goal_reward-=2
                    elif isinstance(sv, dict):# intent=='inform'
                        if 'inform' not in goal[domain] and 'book' not in goal[domain]:
                            goal_reward-=len(sv)
                            continue
                        goal_dict={}
                        for intent_g in ['inform', 'book']:
                            if intent_g in goal[domain]:
                                for k, v in goal[domain][intent_g].items():
                                    goal_dict[k]=v
                        for slot, value in sv.items():
                            if slot=='price' and slot not in goal_dict:
                                slot='pricerange'
                            if slot not in goal_dict:
                                goal_reward-=2
                            elif value!=goal_dict[slot]:
                                goal_reward-=2
                            else:
                                goal_reward+=1
            if len(user_act)>1:
                goal_reward/=len(user_act)
            reward=reqt_reward+goal_reward+token_reward
            if reward>max_reward:
                max_reward=reward
                best_act=usr_act
        best_act=act_list[0] if best_act is None else best_act
        return best_act
    
    def find_best_sys_act(self, act_list, usr_act):
        user_act=self.reader.aspan_to_act_dict(usr_act, side='user')
        max_reward=0
        best_act=None
        for aspn in act_list:
            reqt_reward=0
            token_reward=repeat_token_reward(aspn)
            sys_act=self.reader.aspan_to_act_dict(aspn, side='sys')
            for domain in user_act:
                if 'request' in user_act[domain]:
                    if domain not in sys_act or ('inform'  not in sys_act[domain] and 'recommend' not in sys_act[domain]):
                        reqt_reward-=len(user_act[domain]['request'])
                        continue
                    for slot in user_act[domain]['request']:
                        if 'inform' in sys_act[domain] and slot in sys_act[domain]['inform']:
                            reqt_reward+=1
                        elif 'recommend' in sys_act[domain] and slot in sys_act[domain]['recommend']:
                            reqt_reward+=1
                        else:
                            reqt_reward-=1
            reward=reqt_reward+token_reward
            if reward>max_reward:
                max_reward=reward
                best_act=aspn
        if best_act is None:
            best_act=act_list[0]
        return best_act
        

    def interact_by_batch(self, DS, US, batch_size=cfg.interaction_batch_size, dial_id_batch=None, init_goal_batch=None, return_reward=False):
        max_turns=20
        gen_batch=[[] for _ in range(batch_size)]
        end_batch=[0 for _ in range(batch_size)]
        gpan_batch=[]
        goal_batch=[]# current goal batch
        final_goal_batch=[]
        final_gpan_batch=[]
        bs_max_len=50
        act_max_len=20
        resp_max_len=60
        gpan_max_len=50
        if dial_id_batch is None and init_goal_batch is None:
            if cfg.random_goal:
                init_goal_batch=[]
                for _ in range(batch_size):
                    self.user_policy.init_session()
                    goal=self.user_policy.get_goal()
                    goal=self.reader.goal_norm(goal)
                    init_goal_batch.append(goal)
            else:# sample from dataset
                dial_id_batch=random.sample(self.reader.train_list, batch_size)
        self.goal_list_batch=[[] for _ in range(batch_size)]
        if init_goal_batch is None:
            init_goal_batch=[]
            for batch_id, dial_id in enumerate(dial_id_batch):
                dial_id=dial_id+'.json' if '.json' not in dial_id else dial_id
                goal=self.reader.aspan_to_act_dict(self.reader.data[dial_id]['log'][0]['gpan'], side='user')
                init_goal_batch.append(goal)
                goal_batch.append(goal)
                final_goal_batch.append(copy.deepcopy(goal))
                self.goal_list_batch[batch_id].append(goal)
                gpan='<sos_g> '+self.reader.goal_to_gpan(goal)+' <eos_g>'
                final_gpan_batch.append(gpan)
        else:
            for batch_id, init_goal in enumerate(init_goal_batch):
                goal=init_goal
                #goal=self.reader.goal_norm(init_goal)
                goal_batch.append(goal)
                final_goal_batch.append(copy.deepcopy(goal))
                self.goal_list_batch[batch_id].append(goal)
                gpan='<sos_g> '+self.reader.goal_to_gpan(goal)+' <eos_g>'
                gpan_batch.append(gpan)
                final_gpan_batch.append(gpan)
        self.turn_domain_batch=['' for _ in range(batch_size)]
        pv_resp_batch=None
        pv_bspn_batch=None
        pv_aspn_batch=None
        pv_user_act_batch=None
        pv_gpan_batch=None
        for i in range(max_turns):
            if i>0: # update goals
                if cfg.user_nlu and not cfg.deliver_state and not cfg.gen_goal_state:
                    #generate pv_aspn batch
                    contexts=self.get_us_contexts(pv_resp_batch, user_nlu=True)
                    contexts_ids=self.convert_batch_tokens_to_ids(self.US_tok, contexts)
                    pv_aspn_batch_ids=self.generate_batch(self.US, contexts_ids, act_max_len, self.US_tok.convert_tokens_to_ids('<eos_a>'))
                    pv_aspn_batch=self.convert_batch_ids_to_tokens(self.US_tok, pv_aspn_batch_ids, self.US_tok.convert_tokens_to_ids('<sos_a>'), 
                       self.US_tok.convert_tokens_to_ids('<eos_a>'))
                if cfg.gen_goal_state:
                    # generate_goal_state
                    contexts=self.get_us_contexts(pv_resp_batch, pv_gpan_batch=pv_gpan_batch, pv_user_act_batch=pv_user_act_batch)
                    contexts_ids=self.convert_batch_tokens_to_ids(self.US_tok, contexts)
                    gpan_batch_ids=self.generate_batch(self.US, contexts_ids, gpan_max_len, self.US_tok.convert_tokens_to_ids('<eos_g>'))
                    gpan_batch=self.convert_batch_ids_to_tokens(self.US_tok, gpan_batch_ids, self.US_tok.convert_tokens_to_ids('<sos_g>'),
                        self.US_tok.convert_tokens_to_ids('<eos_g>'))
                    for b, gpan in enumerate(gpan_batch):
                        goal=self.reader.aspan_to_act_dict(gpan, side='user')
                        goal_batch[b]=goal
                        self.goal_list_batch[b].append(goal)

                else:
                    for batch_id, goal in enumerate(goal_batch):
                        pv_user_act_dict=self.reader.aspan_to_act_dict(pv_user_act_batch[batch_id], side='user')
                        pv_sys_act=self.reader.aspan_to_act_dict(pv_aspn_batch[batch_id], side='sys') if cfg.user_nlu else None
                        pv_sys_act=self.reader.correct_act(pv_sys_act, pv_user_act_dict) if cfg.user_nlu else None
                        pv_aspn_batch[batch_id]='<sos_a> '+self.reader.act_dict_to_aspan(pv_sys_act)+' <eos_a>'
                        goal, final_goal_batch[batch_id] = self.reader.update_goal(
                            goal, final_goal_batch[batch_id], pv_user_act_dict, pv_sys_act)
                        goal_batch[batch_id]=goal
                        self.goal_list_batch[batch_id].append(goal)
                        gpan_batch[batch_id]='<sos_g> '+self.reader.goal_to_gpan(goal)+' <eos_g>'
                        final_gpan_batch[batch_id]='<sos_g> '+self.reader.goal_to_gpan(final_goal_batch[batch_id])+' <eos_g>'
                    
            # generate user act batch
            if cfg.gen_goal_state:
                contexts=self.get_us_contexts(pv_resp_batch, gpan_batch, pv_gpan_batch=pv_gpan_batch, pv_user_act_batch=pv_user_act_batch)
            else:
                contexts=self.get_us_contexts(pv_resp_batch, gpan_batch, pv_aspn_batch=pv_aspn_batch, final_goal_batch=final_gpan_batch)
            contexts_ids=self.convert_batch_tokens_to_ids(self.US_tok, contexts)
            if cfg.beam_search:
                beam_ids, beam_probs=self.generate_batch(US, contexts_ids, act_max_len, self.eos_ua_id, beam=cfg.beam_size_u)
                user_act_batch=[]
                user_act_beam_batch=[]
                sampled_ids=torch.multinomial(beam_probs, 1) #B, 1
                for b, temp_ids in enumerate(beam_ids):
                    beam_batch=self.convert_batch_ids_to_tokens(self.US_tok, temp_ids, self.sos_ua_id, self.eos_ua_id)
                    user_act_batch.append(beam_batch[sampled_ids[b, 0]])
                    user_act_beam_batch.append(beam_batch)
            else:
                user_act_batch_ids=self.generate_batch(US, contexts_ids, act_max_len, self.eos_ua_id)
                user_act_batch=self.convert_batch_ids_to_tokens(self.US_tok, user_act_batch_ids, 
                    self.sos_ua_id, self.eos_ua_id)

            # generate user batch
            if cfg.gen_goal_state:
                contexts=self.get_us_contexts(pv_resp_batch, gpan_batch, user_act_batch, pv_gpan_batch=pv_gpan_batch, pv_user_act_batch=pv_user_act_batch)
            else:
                contexts=self.get_us_contexts(pv_resp_batch, gpan_batch, user_act_batch, pv_aspn_batch=pv_aspn_batch, final_goal_batch=final_gpan_batch)
            contexts_ids=self.convert_batch_tokens_to_ids(self.US_tok, contexts)
            user_batch_ids=self.generate_batch(US, contexts_ids, resp_max_len, self.eos_u_id)
            user_batch=self.convert_batch_ids_to_tokens(self.US_tok, user_batch_ids, 
                self.sos_u_id, self.eos_u_id)
            # generate bspn batch
            if cfg.deliver_state:
                bspn_batch=[]
                for k, user_act in enumerate(user_act_batch):
                    pv_bspn=None if pv_bspn_batch==None else pv_bspn_batch[k]
                    bspn=self.reader.update_bspn(pv_bspn, user_act)
                    bspn_batch.append(bspn)
            else:
                contexts=self.get_ds_contexts(user_batch, pv_bspn_batch, pv_resp_batch)
                contexts_ids=self.convert_batch_tokens_to_ids(self.DS_tok, contexts)
                if cfg.fix_DST: # use a fixed DST model to generate belief state
                    bspn_batch_ids=self.generate_batch(self.DST, contexts_ids, bs_max_len, self.eos_b_id)
                else:
                    bspn_batch_ids=self.generate_batch(DS, contexts_ids, bs_max_len, self.eos_b_id)
                bspn_batch=self.convert_batch_ids_to_tokens(self.DS_tok, bspn_batch_ids, 
                    self.sos_b_id, self.eos_b_id)
            db_batch=self.get_db_batch(bspn_batch, pv_bspn_batch)
            # generate act batch
            contexts=self.get_ds_contexts(user_batch, pv_bspn_batch, pv_resp_batch, bspn_batch, db_batch)
            contexts_ids=self.convert_batch_tokens_to_ids(self.DS_tok, contexts)
            if cfg.beam_search:
                beam_ids, beam_probs=self.generate_batch(DS, contexts_ids, act_max_len, self.eos_a_id, beam=cfg.beam_size_s)
                aspn_batch=[]
                aspn_beam_batch=[]
                sampled_ids=torch.multinomial(beam_probs, 1)
                for b, temp_ids in enumerate(beam_ids):
                    beam_batch=self.convert_batch_ids_to_tokens(self.DS_tok, temp_ids, self.sos_a_id, self.eos_a_id)
                    aspn_batch.append(beam_batch[sampled_ids[b, 0]])
                    #aspn_batch.append(self.find_best_sys_act(beam_batch, user_act_batch[b]))
                    aspn_beam_batch.append(beam_batch)
            else:
                aspn_batch_ids=self.generate_batch(DS, contexts_ids, act_max_len, self.eos_a_id)
                aspn_batch=self.convert_batch_ids_to_tokens(self.DS_tok, aspn_batch_ids, 
                    self.sos_a_id, self.eos_a_id)
            # generate resp batch
            contexts=self.get_ds_contexts(user_batch, pv_bspn_batch, pv_resp_batch, bspn_batch, db_batch,aspn_batch)
            contexts_ids=self.convert_batch_tokens_to_ids(self.DS_tok, contexts)
            resp_batch_ids=self.generate_batch(DS, contexts_ids, resp_max_len, self.eos_r_id)
            resp_batch=self.convert_batch_ids_to_tokens(self.DS_tok, resp_batch_ids, 
                self.sos_r_id, self.eos_r_id)
            

            # collect dialogs and judge stop
            for batch_id in range(batch_size):
                user_act=user_act_batch[batch_id]
                aspn=aspn_batch[batch_id]
                goal=goal_batch[batch_id]
                if not end_batch[batch_id]:
                    turn={}
                    if i>0 and cfg.user_nlu and not cfg.gen_goal_state:
                        turn['pv_aspn']=pv_aspn_batch[batch_id]
                    turn['gpan']=gpan_batch[batch_id]
                    turn['usr_act']=user_act_batch[batch_id]
                    turn['user']=user_batch[batch_id]
                    turn['bspn']=bspn_batch[batch_id]
                    turn['db']=db_batch[batch_id]
                    turn['aspn']=aspn_batch[batch_id]
                    turn['resp']=resp_batch[batch_id]
                    if cfg.beam_search:
                        turn['usr_act_beam']=user_act_beam_batch[batch_id]
                        turn['aspn_beam']=aspn_beam_batch[batch_id]
                    gen_batch[batch_id].append(turn)
                if (set(user_act.split()).issubset(self.end_tokens) and set(aspn.split()).issubset(self.end_tokens)) or goal=={}:
                    end_batch[batch_id]=1
                if len(gen_batch[batch_id])>2 and gen_batch[batch_id][-1]==gen_batch[batch_id][-2]:
                    end_batch[batch_id]=1
            if all(end_batch):
                break

            # before next turn
            pv_bspn_batch=bspn_batch
            pv_resp_batch=resp_batch
            pv_user_act_batch=user_act_batch
            pv_gpan_batch=gpan_batch
            if cfg.deliver_state:
                pv_aspn_batch=aspn_batch
        if return_reward:
            US_reward_batch=[]
            DS_reward_batch=[]
            total_success=0
            total_match=0
            total_comp=0
            US_rewards=np.zeros(7)
            DS_rewards=np.zeros(6)
            for batch_id, (final_goal, goal_list, gen_dial) in enumerate(zip(final_goal_batch, self.goal_list_batch, gen_batch)):
                gen_batch[batch_id][-1]['final_goal'] = self.reader.goal_to_gpan(final_goal)
                DS_reward_list, DS_reward = self.get_DS_reward(gen_dial, final_goal, return_avg_reward=True)
                success=1 if DS_reward[3]==10 else 0
                match=1 if DS_reward[3]>=7.5 else 0
                US_reward_list, US_reward = self.get_US_reward(gen_dial, goal_list, return_avg_reward=True, success=success)
                US_reward_batch.append(US_reward_list)
                DS_reward_batch.append(DS_reward_list)
                
                US_rewards += US_reward
                DS_rewards += DS_reward

                total_success+=success
                total_match+=match
                if goal_list[-1]=={}:
                    total_comp+=1
            
            return gen_batch, US_reward_batch, DS_reward_batch, total_success, total_match, total_comp, US_rewards, DS_rewards

        return gen_batch
    
    def generate_batch(self, model, contexts, max_len, eos_id, beam=1):
        # generate by batch
        # contexts: a list of ids
        # max_len: the max generated length
        # eos_id: the end id
        # return: a batch of ids with pre pad 
        batch_size=len(contexts)
        end_flag=np.zeros(batch_size)
        if beam>1:
            beam_box=[beam]*batch_size
            beam_result=[[] for _ in range(batch_size)]
            beam_probs=[]
        past_key_values=None
        inputs,attentions=self.reader.batch_align(contexts,left_len=max_len,return_attn=True)
        inputs=torch.tensor(inputs).to(model.device)
        attentions=torch.tensor(attentions).to(model.device)
        model.eval()
        with torch.no_grad():
            for i in range(max_len):
                if beam==1:
                    position_ids = attentions.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attentions == 0, 1)
                    if past_key_values is not None:
                        position_ids=position_ids[:, -1].unsqueeze(-1)
                    if inputs.size(0)==0:
                        raise ValueError(contexts, inputs.cpu().list(), attentions)
                    outputs=model(inputs,attention_mask=attentions,position_ids=position_ids,\
                            return_dict=True,use_cache=True,past_key_values=past_key_values)

                    past_key_values=outputs.past_key_values

                    preds=outputs.logits[:,-1,:].argmax(-1)#B
                    if i==0:
                        gen_tensor=preds.unsqueeze(1)
                    else:
                        gen_tensor=torch.cat([gen_tensor,preds.unsqueeze(1)],dim=1)
                    attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(model.device)),dim=1)
                    inputs=preds.unsqueeze(1)
                    end_flag+=(preds.cpu().numpy()==eos_id).astype(float)
                    if sum(end_flag==0)==0:
                        break
                else:
                    if i==0:
                        position_ids = attentions.long().cumsum(-1) - 1
                        position_ids.masked_fill_(attentions == 0, 1)
                        outputs=model(inputs,attention_mask=attentions,position_ids=position_ids,\
                                return_dict=True,use_cache=True,past_key_values=past_key_values)
                        past_key_values=[outputs.past_key_values]*beam
                        log_prob=F.log_softmax(outputs.logits[:, -1, :], -1) # B, V
                        beam_prob, beam_idx=torch.topk(log_prob, beam, -1) # B, beam
                        gen_tensor=beam_idx.unsqueeze(-1)# B, beam, 1
                        attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(model.device)),dim=1)
                        position_ids = attentions.long().cumsum(-1) - 1
                        position_ids.masked_fill_(attentions == 0, 1)
                        position_ids=position_ids[:, -1].unsqueeze(-1)
                        pv_beam_prob=beam_prob #B, beam
                        pv_beam_idx=beam_idx#B, beam
                    else:
                        for j in range(beam):
                            inputs=pv_beam_idx[:,j].unsqueeze(-1) # B, 1
                            outputs=model(inputs,attention_mask=attentions,position_ids=position_ids,\
                                return_dict=True,use_cache=True,past_key_values=past_key_values[j])
                            past_key_values[j]=outputs.past_key_values
                            log_prob=F.log_softmax(outputs.logits[:, -1, :], -1) # B, V
                            beam_prob, beam_idx=torch.topk(log_prob, beam, -1) # B, beam
                            if j==0:
                                prob_pool= beam_prob+pv_beam_prob[:, j].unsqueeze(-1).expand(-1, beam) # B, beam
                                id_pool=beam_idx
                            else:
                                prob_pool=torch.cat([prob_pool, beam_prob+pv_beam_prob[:, j].unsqueeze(-1).expand(-1, beam)],-1) # B, beam*beam
                                id_pool=torch.cat([id_pool, beam_idx], -1)# B, beam*beam
                        beam_prob, temp_id=torch.topk(prob_pool, beam, -1) #B, beam
                        beam_idx=torch.gather(id_pool, -1, temp_id)
                        temp_id=temp_id//beam
                        new_past_key_values=copy.deepcopy(past_key_values)
                        for b in range(batch_size):
                            gen_tensor[b, :, :]=gen_tensor[b, :, :].index_select(0, temp_id[b, :])
                            for t in range(beam):
                                for l in range(6):
                                    new_past_key_values[t][l][:, b, :,:,:]=past_key_values[temp_id[b, t]][l][:, b, :, :, :]
                        past_key_values=new_past_key_values
                        #past_key_values=[past_key_values[t] for t in temp_id.cpu().list()]
                        gen_tensor=torch.cat([gen_tensor, beam_idx.unsqueeze(-1)],-1) #B, beam, T
                        attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(model.device)),dim=1)
                        position_ids = attentions.long().cumsum(-1) - 1
                        position_ids.masked_fill_(attentions == 0, 1)
                        position_ids=position_ids[:, -1].unsqueeze(-1)
                        pv_beam_prob=beam_prob #B, beam
                        pv_beam_idx=beam_idx
                    for m in range(batch_size):
                        for n, gen in enumerate(gen_tensor.cpu().tolist()[m]):
                            if eos_id in gen:
                                beam_box[m]-=1
                                avg_prob=pv_beam_prob[m][n]/len(gen)
                                beam_result[m].append((gen, avg_prob))
                                pv_beam_prob[m][n]=-float('inf')
                        if i==max_len-1 and len(beam_result[m])==0:
                            gen=gen_tensor.cpu().tolist()[m][0]
                            avg_prob=pv_beam_prob[m][0]/len(gen)
                            beam_result[m].append((gen, avg_prob))

        if beam==1:
            return gen_tensor.cpu().tolist()
        else:
            for i, tup in enumerate(beam_result):
                beam_list=sorted(tup, key=lambda item:item[1], reverse=True)
                if len(beam_list)<beam:
                    # add some items with prob=0
                    add=[(beam_list[-1][0], -float('inf'))]*(beam-len(beam_list))
                    beam_list.extend(add)
                beam_result[i]=[item[0] for item in beam_list[:beam]]
                prob=F.softmax(torch.tensor([item[1] for item in beam_list[:beam]]).float(), -1)
                beam_probs.append(prob)
            beam_probs=torch.stack(beam_probs) # B, beam
            return beam_result, beam_probs
    
    def get_us_contexts(self, pv_resp_batch=None, gpan_batch=None, user_act_batch=None, user_nlu=False,
        pv_aspn_batch=None, final_goal_batch=None, pv_gpan_batch=None, pv_user_act_batch=None):
        contexts=[]
        if user_nlu:# pv_resp_batch is not None
            for pv_r in pv_resp_batch:
                context = pv_r + '<sos_a>'
                contexts.append(context)
        else:
            if pv_resp_batch==None:# first turn
                gpan_batch=final_goal_batch if cfg.full_goal else gpan_batch
                if user_act_batch is None:
                    for gpan in gpan_batch:
                        context = gpan + '<sos_ua>'
                        contexts.append(context)
                else:
                    for gpan, ua in zip(gpan_batch, user_act_batch):
                        context = gpan + ua + '<sos_u>'
                        contexts.append(context)
            else:
                if cfg.gen_goal_state:
                    if gpan_batch==None:
                        for pv_g, pv_ua, pv_r in zip(pv_gpan_batch, pv_user_act_batch, pv_resp_batch):
                            context = pv_g + pv_ua + pv_r + '<sos_g>' # to generate the goal state of current turn
                            contexts.append(context)
                    else:
                        if user_act_batch is None:
                            for pv_g, pv_ua, pv_r, g in zip(pv_gpan_batch, pv_user_act_batch, pv_resp_batch, gpan_batch):
                                context = pv_g + pv_ua + pv_r + g + '<sos_ua>'
                                contexts.append(context)
                        else:
                            for pv_g, pv_ua, pv_r, g, ua in zip(pv_gpan_batch, pv_user_act_batch, pv_resp_batch, gpan_batch, user_act_batch):
                                context = pv_g + pv_ua + pv_r + g + ua + '<sos_u>'
                                contexts.append(context)
                else:
                    gpan_batch=final_goal_batch if cfg.full_goal else gpan_batch
                    if pv_aspn_batch: # nlu included 
                        if user_act_batch is None:
                            for gpan, pv_r, pv_a in zip(gpan_batch, pv_resp_batch, pv_aspn_batch):
                                context = pv_r + pv_a + gpan + '<sos_ua>'
                                contexts.append(context)
                        else:
                            for gpan, pv_r, pv_a, ua in zip(gpan_batch, pv_resp_batch, pv_aspn_batch, user_act_batch):
                                context = pv_r + pv_a + gpan + ua + '<sos_u>'
                                contexts.append(context)
                    else:
                        if user_act_batch is None:
                            for gpan, pv_r in zip(gpan_batch, pv_resp_batch):
                                context = pv_r + gpan + '<sos_ua>'
                                contexts.append(context)
                        else:
                            for gpan, pv_r, ua in zip(gpan_batch, pv_resp_batch, user_act_batch):
                                context = pv_r + gpan + ua + '<sos_u>'
                                contexts.append(context)
        return contexts
    
    
    
    def get_ds_contexts(self, user_batch, pv_bspn_batch=None, pv_resp_batch=None, bspn_batch=None, 
        db_batch=None, aspn_batch=None):
        contexts=[]
        if pv_resp_batch is None: # first turn
            if bspn_batch is None:
                for u in user_batch:
                    contexts.append(u + '<sos_b>')
            elif aspn_batch is None:
                for u, b, db in zip(user_batch, bspn_batch, db_batch):
                    contexts.append(u + b + db + '<sos_a>')
            else:
                for u, b, db, a in zip(user_batch, bspn_batch, db_batch, aspn_batch):
                    contexts.append(u + b + db + a + '<sos_r>')
        else:
            if bspn_batch is None:
                for pv_b, pv_r, u in zip(pv_bspn_batch, pv_resp_batch, user_batch):
                    contexts.append(pv_b + pv_r + u + '<sos_b>')
            elif aspn_batch is None:
                for pv_b, pv_r, u, b, db in zip(pv_bspn_batch, pv_resp_batch, user_batch, bspn_batch, db_batch):
                    contexts.append(pv_b + pv_r + u + b + db + '<sos_a>')
            else:
                for pv_b, pv_r, u, b, db, a in zip(pv_bspn_batch, pv_resp_batch, user_batch, bspn_batch, db_batch, aspn_batch):
                    contexts.append(pv_b + pv_r + u + b + db + a + '<sos_r>')
        return contexts

    def get_db_batch(self, bs_batch, pv_bs_batch=None):

        db_batch=[]
        for i, bspn in enumerate(bs_batch):
            cons=self.reader.bspan_to_constraint_dict(bspn)
            cur_domain=list(cons.keys())
            if cur_domain==[]:
                db_result='<sos_db> '+ '[db_0]' + ' <eos_db>'
            else:
                if len(cur_domain)==1:
                    self.turn_domain_batch[i]=cur_domain
                else:
                    if pv_bs_batch is None:
                        max_slot_num=0 # We choose the domain with most slots as the current domain
                        for domain in cur_domain:
                            if len(cons[domain])>max_slot_num:
                                self.turn_domain_batch[i]=[domain]
                                max_slot_num=len(cons[domain])
                    else:
                        pv_domain=list(self.reader.bspan_to_constraint_dict(pv_bs_batch[i]).keys())
                        for domain in cur_domain:
                            if domain not in pv_domain: # new domain
                                # if domains are all the same, self.domain will not change
                                self.turn_domain_batch[i]=[domain]

                #bspn=bspn.replace('portugese', 'portuguese')
                db_result = self.reader.bspan_to_DBpointer(bspn, self.turn_domain_batch[i]) #[db_x]
                db_result = '<sos_db> '+ db_result + ' <eos_db>'
            db_batch.append(db_result)

        return db_batch

    def get_US_reward(self, dial, goal_list, return_avg_reward=False, success=None):
        turn_num=len(dial)
        rewards=[]
        avg_reward=0
        avg_reqt_reward=0
        avg_goal_reward=0
        avg_repeat_reward=0
        avg_token_reward=0
        goal_comp_rate=self.goal_complete_rate(goal_list[0], goal_list[-1])
        goal_comp_reward=10*goal_comp_rate

        global_reward = goal_comp_reward -turn_num
        pv_sys_act=None
        user_act_list=[]
        for turn, goal in zip(dial, goal_list):
            reqt_reward=0
            goal_reward=0
            repeat_reward=0
            end_reward=0
            user_act=self.reader.aspan_to_act_dict(turn['usr_act'], side='user')
            token_reward=repeat_token_reward(turn['usr_act'])
            if cfg.add_end_reward:
                if user_act=={} and goal!={}: # user act is empty but goal is not, punish
                    for domain in goal:
                        for intent, sv in goal[domain].items():
                            if intent=='request':
                                for s in sv:
                                    if pv_sys_act is None or domain not in pv_sys_act or \
                                        intent not in pv_sys_act[domain] or s not in pv_sys_act[domain][intent]:
                                        end_reward-=1
                            else:
                                end_reward-=len(sv)
            if pv_sys_act:
                for domain in pv_sys_act:
                    if 'request' in pv_sys_act[domain]:
                        if domain not in user_act or 'inform' not in user_act[domain]:
                            #reqt_reward-=len(pv_sys_act[domain]['request'])
                            continue
                        for slot in pv_sys_act[domain]['request']:
                            if slot in user_act[domain]['inform']:
                                reqt_reward+=0.1
                            else:
                                pass # We do not punish this case
            for domain in user_act:
                for intent, sv in user_act[domain].items():
                    if domain not in goal:
                        goal_reward-=0.2*len(sv)
                        continue
                    if isinstance(sv, list):# intent=='request'
                        if 'request' not in goal[domain]:
                            goal_reward-=0.2*len(sv)
                            continue
                        for slot in sv:
                            if slot=='price' and slot not in goal[domain][intent]:
                                slot='pricerange'
                            if slot in goal[domain][intent]:
                                goal_reward+=0.1
                            else:
                                goal_reward-=0.2
                    elif isinstance(sv, dict):# intent=='inform'
                        if 'inform' not in goal[domain] and 'book' not in goal[domain]:
                            goal_reward-=0.2*len(sv)
                            continue
                        goal_dict={}
                        for intent_g in ['inform', 'book']:
                            if intent_g in goal[domain]:
                                for k, v in goal[domain][intent_g].items():
                                    goal_dict[k]=v
                        for slot, value in sv.items():
                            if slot=='price' and slot not in goal_dict:
                                slot='pricerange'
                            if slot not in goal_dict:
                                goal_reward-=0.2
                            elif value!=goal_dict[slot]:
                                goal_reward-=0.2
                            else:
                                goal_reward+=0.1
            if user_act in user_act_list: # repeat the same action
                repeat_reward-=0.5
            user_act_list.append(user_act)
            pv_sys_act=self.reader.aspan_to_act_dict(turn['aspn'], side='sys')
            
            final_reward= reqt_reward + goal_reward + repeat_reward + goal_comp_rate
            if cfg.non_neg_reward:
                final_reward=1/(1+math.exp(-final_reward)) # sigmoid
                #final_reward=max(final_reward, 0)
            
            if cfg.simple_reward:
                final_reward=success
                #final_reward=goal_comp_rate
            rewards.append(final_reward)
            turn['US_reward']=str({'reward':final_reward, 'reqt_reward':reqt_reward, 'goal_reward':goal_reward, 
                'repeat_reward':repeat_reward, 'goal_comp':goal_comp_rate, 'turn_num':turn_num, 'token_reward':token_reward})
            avg_goal_reward+=goal_reward/turn_num
            avg_reqt_reward+=reqt_reward/turn_num
            avg_repeat_reward+=repeat_reward/turn_num
            avg_reward+=final_reward/turn_num
            avg_token_reward+=token_reward
        if return_avg_reward:
            return rewards, np.array([avg_reward, avg_reqt_reward, avg_goal_reward, avg_repeat_reward, goal_comp_reward, turn_num, token_reward])
        return rewards

    def get_DS_reward(self, dial, goal, return_avg_reward=False):
        turn_num=len(dial)
        rewards=[]
        avg_reward=0
        avg_reqt_reward=0
        avg_repeat_reward=0
        avg_token_reward=0
        sys_act_list=[]
        success, match=self.evaluator.get_metrics(goal, dial)
        if success==1:
            global_reward=10-turn_num
        else:
            global_reward=7.5*match-turn_num
        success_reward=global_reward+turn_num
        for turn in dial:
            reqt_reward=0
            repeat_reward=0
            token_reward=repeat_token_reward(turn['aspn'])
            user_act=self.reader.aspan_to_act_dict(turn['usr_act'], side='user')
            sys_act=self.reader.aspan_to_act_dict(turn['aspn'], side='sys')
            for domain in user_act:
                if 'request' in user_act[domain]:
                    if domain not in sys_act or ('inform'  not in sys_act[domain] and 'recommend' not in sys_act[domain]):
                        reqt_reward-=0.1*len(user_act[domain]['request'])
                        continue
                    for slot in user_act[domain]['request']:
                        if 'inform' in sys_act[domain] and slot in sys_act[domain]['inform']:
                            reqt_reward+=0.1
                        elif 'recommend' in sys_act[domain] and slot in sys_act[domain]['recommend']:
                            reqt_reward+=0.1
                        else:
                            reqt_reward-=0.1
            if sys_act in sys_act_list:
                repeat_reward-=0.5
            sys_act_list.append(sys_act)
            final_reward = reqt_reward + repeat_reward + success
            if cfg.non_neg_reward:
                final_reward=1/(1+math.exp(-final_reward)) # sigmoid
                #final_reward=max(final_reward,0)
            if cfg.simple_reward:
                final_reward=success
            rewards.append(final_reward)
            turn['DS_reward']=str({'reward':final_reward, 'reqt_reward':reqt_reward, 'repeat_reward':repeat_reward, 
             'success_reward':success_reward, 'turn_num':turn_num, 'token_reward':token_reward})
            avg_reward+=final_reward/turn_num
            avg_reqt_reward+=reqt_reward/turn_num
            avg_repeat_reward+=repeat_reward/turn_num
            avg_token_reward+=token_reward
        if return_avg_reward:
            return rewards, np.array([avg_reward, avg_reqt_reward, avg_repeat_reward, success_reward, turn_num, avg_token_reward])
        return rewards
        
    def goal_complete_rate(self, goal, final_goal):
        total_slot_num=0
        incomp_slot_num=0
        for domain in goal:
            for intent, sv in goal[domain].items():
                total_slot_num+=len(sv)
        if final_goal!={}:
            for domain in final_goal:
                for intent, sv in final_goal[domain].items():
                    incomp_slot_num+=len(sv)
        if total_slot_num==0:
            comp_rate=1
        else:
            comp_rate=(total_slot_num-incomp_slot_num)/total_slot_num
        return comp_rate


    def convert_batch_ids_to_tokens(self, tokenizer, input_ids, sos_id, eos_id):
        # input_ids: B*T
        # output: B*string
        outputs=[]
        for sent_ids in input_ids:
            if eos_id in sent_ids:
                sent_ids=sent_ids[:sent_ids.index(eos_id)+1]
            else:
                sent_ids[-1]=eos_id
            sent_ids=[sos_id]+sent_ids
            if sent_ids.count(sos_id)>1:# more than 1 start token
                last=sent_ids[::-1].index(sos_id)+1
                sent_ids=sent_ids[-last:]
            outputs.append(tokenizer.decode(sent_ids))
        return outputs

    def convert_batch_tokens_to_ids(self, tokenizer, contexts):
        outputs=[]
        for context in contexts:
            outputs.append(modified_encode(tokenizer, context))
        return outputs
    
    def response(self, user):
        user='<sos_u> '+user+' <eos_u>'
        if self.pv_bspn is None: # first turn
            bspn, db, aspn, resp = self.get_sys_response(user)
        else:
            bspn, db, aspn, resp = self.get_sys_response(user, self.pv_bspn, self.pv_resp, pv_aspn=self.pv_aspn)
        self.pv_bspn=bspn
        self.pv_resp=resp
        self.pv_aspn=aspn
        resp1=self.lex_resp(resp, bspn, aspn, self.turn_domain)
        resp1=resp1.lower()
        if 'guesthouse' in user.lower():
            resp1=resp1.replace('guest house', 'guesthouse')
        elif 'guest house' in user.lower():
            resp1=resp1.replace('guesthouse', 'guest house')
        resp1=resp1.replace('portugese', 'portuguese')
        turn={'user':user, 'bspn':bspn, 'db':db, 'aspn':aspn, 'resp':resp, 'lex_resp': resp1}
        self.dialog.append(turn)
        if self.interacting:
            return bspn, db, aspn, resp1
        return resp1

    def lex_resp(self, resp, bspn, aspn, turn_domain):
        value_map={}
        restored = resp
        restored=restored.replace('<sos_r>','')
        restored=restored.replace('<eos_r>','')
        restored.strip()
        restored = restored.capitalize()
        restored = restored.replace(' -s', 's')
        restored = restored.replace(' -ly', 'ly')
        restored = restored.replace(' -er', 'er')
        constraint_dict=self.reader.bspan_to_constraint_dict(bspn)#{'hotel': {'stay': '3'}, 'restaurant': {'people': '4'}}
        mat_ents = self.reader.db.get_match_num(constraint_dict, True)
        #print(mat_ents)
        #print(constraint_dict)
        if '[value_car]' in restored:
            restored = restored.replace('[value_car]','toyota')
            value_map['taxi']={}
            value_map['taxi']['car']='toyota'

        # restored.replace('[value_phone]', '830-430-6666')
        domain=[]
        for d in turn_domain:
            if d.startswith('['):
                domain.append(d[1:-1])
            else:
                domain.append(d)
        act_dict=self.reader.aspan_to_act_dict(aspn)
        if len(act_dict)==1:
            domain=list(act_dict.keys())

        if list(act_dict.keys())==['police']:
            if '[value_name]' in restored:
                restored=restored.replace('[value_name]', 'parkside police station')
            if '[value_address]' in restored:
                restored=restored.replace('[value_address]', 'parkside , cambridge')
            if '[value_phone]' in restored:
                restored=restored.replace('[value_phone]', '01223358966')
        if list(act_dict.keys())==['hospital']:
            if '[value_address]' in restored:
                restored=restored.replace('[value_address]', 'Hills Rd, Cambridge')
            if '[value_postcode]' in restored:
                restored=restored.replace('[value_postcode]', 'CB20QQ')
        for d in domain:
            constraint = constraint_dict.get(d,None)
            if d not in value_map:
                value_map[d]={}
            if constraint:
                if 'stay' in constraint and '[value_stay]' in restored:
                    restored = restored.replace('[value_stay]', constraint['stay'])
                    value_map[d]['stay']=constraint['stay']
                if 'day' in constraint and '[value_day]' in restored:
                    restored = restored.replace('[value_day]', constraint['day'])
                    value_map[d]['day']=constraint['day']
                if 'people' in constraint and '[value_people]' in restored:
                    restored = restored.replace('[value_people]', constraint['people'])
                    value_map[d]['people']=constraint['people']
                if 'time' in constraint and '[value_time]' in restored:
                    restored = restored.replace('[value_time]', constraint['time'])
                    value_map[d]['time']=constraint['time']
                if 'type' in constraint and '[value_type]' in restored:
                    restored = restored.replace('[value_type]', constraint['type'])
                    value_map[d]['type']=constraint['type']
                if d in mat_ents and len(mat_ents[d])==0:
                    for s in constraint:
                        if s == 'pricerange' and d in ['hotel', 'restaurant'] and 'price]' in restored:
                            restored = restored.replace('[value_price]', constraint['pricerange'])
                            value_map[d]['price']=constraint['pricerange']
                        if s+']' in restored:
                            restored = restored.replace('[value_%s]'%s, constraint[s])
                            value_map[d][s]=constraint[s]

            if '[value_choice' in restored and mat_ents.get(d):
                restored = restored.replace('[value_choice]', str(len(mat_ents[d])))
                value_map[d]['choice']=str(len(mat_ents[d]))
        if '[value_choice' in restored:
            restored = restored.replace('[value_choice]', str(random.choice([1,2,3,4,5])))


        ent = mat_ents.get(domain[-1], [])
        d=domain[-1]
        if d not in value_map:
            value_map[d]={}
        if ent:
            # handle multiple [value_xxx] tokens first
            restored_split = restored.split()
            token_count = Counter(restored_split)
            for idx, t in enumerate(restored_split):
                if '[value' in t and token_count[t]>1 and token_count[t]<=len(ent):
                    id1=t.index('_')
                    id2=t.index(']')
                    slot = t[id1+1:id2]
                    pattern = r'\['+t[1:-1]+r'\]'
                    for e in ent:
                        if e.get(slot):
                            if domain[-1] == 'hotel' and slot == 'price':
                                slot = 'pricerange'
                            if slot in ['name', 'address']:
                                rep = ' '.join([i.capitalize() if i not in stopwords else i for i in e[slot].split()])
                            elif slot in ['id','postcode']:
                                rep = e[slot].upper()
                            else:
                                rep = e[slot]
                            restored = re.sub(pattern, rep, restored, 1)
                            value_map[d][slot]=rep
                        elif slot == 'price' and  e.get('pricerange'):
                            restored = re.sub(pattern, e['pricerange'], restored, 1)
                            value_map[d][slot]=e['pricerange']

            # handle normal 1 entity case
            ent = ent[0]
            ents_list=self.reader.db.dbs[domain[-1]]
            ref_no=ents_list.index(ent)
            if ref_no>9:
                if '[value_reference]' in restored:
                    restored = restored.replace('[value_reference]', '000000'+str(ref_no))
                    value_map[d]['reference']='000000'+str(ref_no)
            else:
                if '[value_reference]' in restored:
                    restored = restored.replace('[value_reference]', '0000000'+str(ref_no))
                    value_map[d]['reference']='0000000'+str(ref_no)
            for t in restored.split():
                if '[value' in t:
                    id1=t.index('_')
                    id2=t.index(']')
                    slot = t[id1+1:id2]
                    if ent.get(slot):
                        if domain[-1] == 'hotel' and slot == 'price':
                            slot = 'pricerange'
                        if slot in ['name', 'address']:
                            rep = ' '.join([i.capitalize() if i not in stopwords else i for i in ent[slot].split()])
                        elif slot in ['id','postcode']:
                            rep = ent[slot].upper()
                        else:
                            rep = ent[slot]
                        # rep = ent[slot]
                        rep='free' if slot in ['price', 'pricerange'] and rep=='?' else rep
                        if 'total fee' in restored and 'pounds' in rep:
                            price=float(rep.strip('pounds').strip())
                            people=constraint_dict[d].get('people', '1')
                            people=int(people) if people.isdigit() else 1
                            #calculate the total fee, people*price
                            rep = str(round(people*price, 2))+' pounds'
                        restored = restored.replace(t, rep)
                        value_map[d][slot]=rep
                        # restored = restored.replace(t, ent[slot])
                    elif slot == 'price' and  ent.get('pricerange'):
                        rep='free' if ent['pricerange']=='?' else ent['pricerange']
                        restored = restored.replace(t, rep)
                        value_map[d][slot]=rep
                        # else:
                        #     print(restored, domain)       
        #restored = restored.replace('[value_area]', 'centre')
        for t in restored.split():
            if '[value' in t:
                slot=t[7:-1]
                value='UNKNOWN'
                for domain, sv in constraint_dict.items():
                    if isinstance(sv, dict) and slot in sv:
                        value=sv[slot]
                        break
                if value=='UNKNOWN':
                    for domain in mat_ents:
                        if len(mat_ents[domain])==0:
                            continue
                        ent=mat_ents[domain][0]
                        if slot in ent:
                            if slot in ['name', 'address']:
                                value=' '.join([i.capitalize() if i not in stopwords else i for i in ent[slot].split()])
                            elif slot in ['id', 'postcode']:
                                value=ent[slot].upper()
                            else:
                                value=ent[slot]
                            break
                if value!='UNKNOWN':
                    if isinstance(value, str):
                        restored = restored.replace(t, value)
                else:
                    for domain in constraint_dict.keys():
                        temp_ent=self.reader.db.dbs[domain][0]
                        if temp_ent.get(slot, None):
                            value=temp_ent[slot]
                            if isinstance(value, str):
                                restored = restored.replace(t, value)
                                break
        restored = restored.replace('[value_phone]', '01223462354')
        restored = restored.replace('[value_postcode]', 'cb21ab')
        restored = restored.replace('[value_address]', 'regent street')
        restored = restored.replace('[value_people]', 'several')
        restored = restored.replace('[value_day]', 'Saturday')
        restored = restored.replace('[value_time]', '12:00')
        restored = restored.split()
        for idx, w in enumerate(restored):
            if idx>0 and restored[idx-1] in ['.', '?', '!']:
                restored[idx]= restored[idx].capitalize()
        restored = ' '.join(restored)

        return restored.strip()
    
    def init_session(self):
        self.pv_bspn=None
        self.pv_resp=None
        self.pv_aspn=None
        self.dialog=[]


if __name__=='__main__':
    pass