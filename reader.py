from xml import dom
import numpy as np
import os
import csv
import random
import logging
import json
import utils
import ontology
import torch
from copy import deepcopy
from collections import OrderedDict
from db_ops import MultiWozDB
from torch.utils.data import Dataset, DataLoader
import transformers
from config import global_config as cfg
#from config21 import global_config as cfg

class _ReaderBase(object):

    def __init__(self):
        self.train, self.dev, self.test = [], [], []
        self.db = None
        self.set_stats = {}

    def _bucket_by_turn(self, encoded_data):
        turn_bucket = {}
        for dial in encoded_data:
            turn_len = len(dial)
            if turn_len==0:
                continue
            if turn_len not in turn_bucket:
                turn_bucket[turn_len] = []
            turn_bucket[turn_len].append(dial)
        del_l = []
        for k in turn_bucket:
            if k >= 5:
                del_l.append(k)
            logging.debug("bucket %d instance %d" % (k, len(turn_bucket[k])))
        # for k in del_l:
        #    turn_bucket.pop(k)
        return OrderedDict(sorted(turn_bucket.items(), key=lambda i: i[0]))

    def _construct_mini_batch(self, data):
        all_batches = []
        batch = []
        for dial in data:
            batch.append(dial)
            if len(batch) == cfg.batch_size:
                all_batches.append(batch)
                batch = []
        if (len(batch) % len(cfg.cuda_device)) != 0:
            batch = batch[:-(len(batch) % len(cfg.cuda_device))]
        if len(batch)>0:
            all_batches.append(batch)
        '''
        if len(batch) > 0.5 * cfg.batch_size:
            all_batches.append(batch)
        elif len(all_batches):
            all_batches[-1].extend(batch)
        else:
            all_batches.append(batch)
        '''
        return all_batches
    
    def transpose_batch(self, batch):
        dial_batch = []
        turn_num = len(batch[0])
        for turn in range(turn_num):
            turn_l = {}
            for dial in batch:
                this_turn = dial[turn]
                for k in this_turn:
                    if k not in turn_l:
                        turn_l[k] = []
                    turn_l[k].append(this_turn[k])
            dial_batch.append(turn_l)
        return dial_batch
        
    def split_turn_batch(self, turn_batch, batch_size, other_batch=None):
        batches=[]
        other_batches=[]
        B=len(turn_batch['user'])
        for i in range(0, B, batch_size):
            new_turn_batch={}
            if other_batch:
                other_batches.append(other_batch[i:i+batch_size])
            for key in turn_batch:
                new_turn_batch[key]=turn_batch[key][i:i+batch_size]
            batches.append(new_turn_batch)
        if other_batch:
            return batches, other_batches
        else:
            return batches, None


    def inverse_transpose_turn(self, turn_list):
        """
        eval, one dialog at a time
        """
        dialogs = {}
        turn_num = len(turn_list)
        dial_id = turn_list[0]['dial_id']
        dialogs[dial_id] = []
        for turn_idx in range(turn_num):
            dial_turn = {}
            turn = turn_list[turn_idx]
            for key, value in turn.items():
                if key=='dial_id':
                    continue
                if key == 'pointer' and self.db is not None:
                    turn_domain = turn['turn_domain'][-1]
                    value = self.db.pointerBack(value, turn_domain)
                dial_turn[key] = value
            dialogs[dial_id].append(dial_turn)
        return dialogs

    def inverse_transpose_batch(self, turn_batch_list):
        """
        :param turn_batch_list: list of transpose dial batch
        """
        dialogs = []
        total_turn_num = len(turn_batch_list)
        # initialize
        for idx_in_batch, dial_id in enumerate(turn_batch_list[0]['dial_id']):
            dialog = []
            for turn_n in range(total_turn_num):
                dial_turn = {}
                turn_batch = turn_batch_list[turn_n]
                for key, v_list in turn_batch.items():
                    if idx_in_batch>=len(v_list):
                        print('list out of range',key, v_list)
                        continue
                    value = v_list[idx_in_batch]
                    dial_turn[key] = value
                dialog.append(dial_turn)
            dialogs.append(dialog)
        return dialogs
    
    def inverse_transpose_batch0(self, turn_batch_list):
        """
        :param turn_batch_list: list of transpose dial batch
        """
        dialogs = {}
        total_turn_num = len(turn_batch_list)
        # initialize
        for idx_in_batch, dial_id in enumerate(turn_batch_list[0]['dial_id']):
            dialogs[dial_id] = []
            for turn_n in range(total_turn_num):
                dial_turn = {}
                turn_batch = turn_batch_list[turn_n]
                for key, v_list in turn_batch.items():
                    if key == 'dial_id':
                        continue
                    value = v_list[idx_in_batch]
                    if key == 'pointer' and self.db is not None:
                        turn_domain = turn_batch['turn_domain'][idx_in_batch][-1]
                        value = self.db.pointerBack(value, turn_domain)
                    dial_turn[key] = value
                dialogs[dial_id].append(dial_turn)
        return dialogs

    def get_eval_data(self, set_name='dev'):
        name_to_set = {'train': self.train, 'test': self.test, 'dev': self.dev}
        dial = name_to_set[set_name]

        if set_name not in self.set_stats:
            self.set_stats[set_name] = {}
        num_turns = 0
        num_dials = len(dial)
        for d in dial:
            num_turns += len(d)

        self.set_stats[set_name]['num_turns'] = num_turns
        self.set_stats[set_name]['num_dials'] = num_dials

        return dial
        
    def get_batches(self, set_name,data=None):
        """
        compute dataset stats.
        """
        global dia_count
        log_str = ''

        name_to_set = {'train': self.train, 'test': self.test, 'dev': self.dev}
        dial = name_to_set[set_name]
        if data:
            dial=data
        if cfg.low_resource and set_name == 'train':
            # dial = random.sample(dial, int(len(dial)*0.01))
            dial = random.sample(dial, 100)
            print('Low Resource setting, finetuning size: {}'.format(len(dial)))
        turn_bucket = self._bucket_by_turn(dial)
        # self._shuffle_turn_bucket(turn_bucket)
        all_batches = []

        
        if set_name not in self.set_stats:
            self.set_stats[set_name] = {}
        num_training_steps = 0
        num_turns = 0
        num_dials = 0

        for k in turn_bucket:
            if set_name != 'test' and k == 1 or k >= 17:
                continue
            batches = self._construct_mini_batch(turn_bucket[k])
            if len(batches)==0:
                continue
            log_str += "turn num:%d, dial num: %d, batch num: %d last batch len: %d\n" % (
                k, len(turn_bucket[k]), len(batches), len(batches[-1]))
            # print("turn num:%d, dial num:v%d, batch num: %d, "%(k, len(turn_bucket[k]), len(batches)))
            num_training_steps += k * len(batches)
            num_turns += k * len(turn_bucket[k])
            num_dials += len(turn_bucket[k])
            all_batches += batches
        log_str += 'total batch num: %d\n' % len(all_batches)

        # log stats
        # cfg.num_training_steps = num_training_steps * cfg.epoch_num
        self.set_stats[set_name]['num_training_steps_per_epoch'] = num_training_steps
        self.set_stats[set_name]['num_turns'] = num_turns
        self.set_stats[set_name]['num_dials'] = num_dials
        if data is None:
            if set_name == 'train':
                random.shuffle(all_batches)
        return all_batches
    
    def get_nontranspose_data_iterator(self, all_batches):
        for i, batch in enumerate(all_batches):
            yield batch

    def get_data_iterator(self, all_batches):
        for i, batch in enumerate(all_batches):
            yield self.transpose_batch(batch)

    def save_result(self, write_mode, results, field, write_title=False,result_name=None):
        field=list(results[0].keys())
        result_name=result_name if result_name is not None else 'result.csv'
        with open(os.path.join(cfg.eval_load_path,result_name), write_mode) as rf:
            if write_title:
                rf.write(write_title+'\n')
            try:
                writer = csv.DictWriter(rf, fieldnames=field)
                writer.writeheader()
                writer.writerows(results)
            except Exception as e:
                print(e)
        return None
    
    def load_result(self,result_path):
        results=[]
        with open(result_path, 'r') as rf:
            reader=csv.reader(rf)
            is_field=True
            for n,line in enumerate(reader):
                entry={}
                if n==0 and line=='DECODED RESULTS:':
                    continue
                if is_field:
                    field=line
                    is_field=False
                else:
                    for i,key in enumerate(field):
                        entry[key]=line[i]
                    results.append(entry)
        return results,field



class MultiWozReader(_ReaderBase):

    def __init__(self, tokenizer):
        super().__init__()

        self.db = MultiWozDB(cfg.dbs)
        self.tokenizer = tokenizer
        self.add_sepcial_tokens()

        self.domain_files = json.loads(open(cfg.domain_file_path, 'r').read())
        self.slot_value_set = json.loads(
            open(cfg.slot_value_set_path, 'r').read())
        
        test_list = [l.strip().lower()
                     for l in open(cfg.test_list, 'r').readlines()]
        dev_list = [l.strip().lower()
                    for l in open(cfg.dev_list, 'r').readlines()]
        self.test_list=[fn.replace('.json', '') for fn in test_list]
        self.dev_list=[fn.replace('.json', '') for fn in dev_list]

        if 'all' not in cfg.exp_domains:
            domains=self.get_exp_domains(cfg.exp_domains, list(self.domain_files.keys()))
            fn_list=[]
            for d in domains:
                fn_list+=[fn.replace('.json', '') for fn in self.domain_files[d]]
            self.fn_list=fn_list
            self.test_list=[fn for fn in self.test_list if fn in self.fn_list]
            self.dev_list=[fn for fn in self.dev_list if fn in self.fn_list]

        self._load_data()
        self.get_special_ids()
        self.clean_dial_id()
        self.nooffer_options=json.load(open('data/multi-woz-2.1-processed/nooffer_options.json', 'r'))

    def get_exp_domains(self, exp_domains, all_domains_list):
        for domain in ['hotel', 'train', 'attraction', 'restaurant', 'taxi']:
            if domain in exp_domains:
                if 'except' in exp_domains:
                    domains=[d for d in all_domains_list if domain not in d and 'multi' not in d]
                else:
                    domains=[domain+'_single', domain+'_multi']
        
        return domains

    def clean_dial_id(self):
        new_list=[]
        for dial_id in self.dev_list:
            if dial_id in self.data:
                new_list.append(dial_id)
        self.dev_list=new_list

    def add_sepcial_tokens(self):
        special_tokens = []
        for word in ontology.all_domains + ['general']:
            word = '[' + word + ']'
            special_tokens.append(word)
        for word in ontology.all_acts:
            word = '[' + word + ']'
            special_tokens.append(word)
        # for word in ontology.all_slots:
            # to be determine whether slot should be [slot]
            # if slot, tokenizer having trouble decoding.
            # special_tokens.append(word)
        vocab_special_tokens=["[value_name]", "[value_choice]", "[value_area]", "[value_price]",
         "[value_type]", "[value_reference]", "[value_phone]", "[value_address]","[value_food]",
         "[value_leave]", "[value_postcode]", "[value_id]", "[value_arrive]", "[value_stars]",
         "[value_day]", "[value_destination]", "[value_car]", "[value_departure]","[value_time]",
         "[value_people]", "[value_stay]", "[value_pricerange]", "[value_department]", "[value_name]([value_phone]"]
        '''
        for word in self.vocab._word2idx.keys():
            if word.startswith('[value_') and word.endswith(']'):
                special_tokens.append(word)
        '''
        for word in vocab_special_tokens:
            if word!='[value_choice]':
                special_tokens.append(word)
            else:
                if cfg.delex_as_damd:
                    special_tokens.append(word)
        special_tokens.extend(ontology.special_tokens)
        
        if cfg.train_us:
            #special_tokens.extend(['[book]','[fail_book]','[fail_info]','[pre_invalid]','[invalid]','<sos_g>','<eos_g>','<sos_ua>','<eos_ua>'])
            special_tokens.extend(['<sos_g>','<eos_g>','<sos_ua>','<eos_ua>'])
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        print('Added special tokens to gpt tokenizer.')

        cfg.pad_id = self.tokenizer.encode('<pad>')[0]
    
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

    def _load_data(self, save_temp=True):
        """
        load processed data and encode, or load already encoded data
        """
        if cfg.train_us:
            encoded_file = os.path.join(cfg.data_path, 'encoded_us_data.json')
        else:
            encoded_file = os.path.join(cfg.data_path, 'encoded_data.json')
        
        data_path='data/multi-woz-2.1-processed/data_for_rl.json'
        self.data = json.loads(open(data_path, 'r', encoding='utf-8').read().lower())
        self.train_list=[]
        for key in self.data:
            fn=key.replace('.json', '')
            if fn not in self.dev_list and fn not in self.test_list:
                if 'all' not in cfg.exp_domains:
                    if fn in self.fn_list:
                        self.train_list.append(fn)
                else:
                    self.train_list.append(fn)
        print('Reading data from {}'.format(data_path))
        if not cfg.rl_train: # encode data
            if os.path.exists(encoded_file):
                print('Reading encoded data from {}'.format(encoded_file))
                encoded_data = json.loads(open(encoded_file, 'r', encoding='utf-8').read())
                self.train=[dial for dial in encoded_data['train'] if dial[0]['dial_id'] in self.train_list]
                self.dev = [dial for dial in encoded_data['dev'] if dial[0]['dial_id'] in self.dev_list]
                self.test = [dial for dial in encoded_data['test'] if dial[0]['dial_id'] in self.test_list]
            else:
                print('Encoding data now and save the encoded data in {}'.format(encoded_file))
                self.train, self.dev, self.test = [], [], []
                for fn, dial in self.data.items():
                    if '.json' in fn:
                        fn = fn.replace('.json', '')
                    if fn in self.dev_list:
                        self.dev.append(self._get_encoded_data(fn, dial))
                    elif fn in self.test_list:
                        self.test.append(self._get_encoded_data(fn, dial))
                    elif fn in self.train_list:
                        self.train.append(self._get_encoded_data(fn, dial))
                
                # save encoded data
                encoded_data = {'train': self.train, 'dev': self.dev, 'test': self.test}
                json.dump(encoded_data, open(encoded_file, 'w'), indent=2)
                print('Encoded file saved in %s'%encoded_file)


        random.shuffle(self.train)
        print('train size:{}, dev size:{}, test size:{}'.format(len(self.train), len(self.dev), len(self.test)))

    def fix_dialog_state(self, data):
        count=0
        for dial_id in data:
            dial=data[dial_id]['log']
            for turn_id, turn in enumerate(dial):
                cons=turn['constraint']
                if 'name' in cons:
                    cons_dict=self.bspan_to_constraint_dict(cons)
                    for domain in cons_dict:
                        name_value=cons_dict[domain].get('name', None)
                        if name_value and name_value not in turn['user']:# not in the current turn
                            name_in_user=False
                            for i in range(turn_id):
                                if name_value in dial[i]['user']:# in previous turns
                                    name_in_user=True
                                    break
                            if not name_in_user:
                                count+=1
                                cons_dict[domain].pop('name')
                    turn['constraint']=self.cons_dict_to_bspn(cons_dict)
        print(count)
        return data

    def cons_dict_to_bspn(self, cons_dict):
        bs_list=[]
        for domain in cons_dict:
            bs_list.append('['+domain+']')
            for slot in cons_dict[domain]:
                bs_list.append(slot)
                bs_list.append(cons_dict[domain][slot])
        return ' '.join(bs_list)

    def _get_encoded_data(self, fn, dial):
        encoded_dial = []
        for idx, t in enumerate(dial['log']):  # tokenize to list of ids
            enc = {}
            enc['dial_id'] = fn
            for key in t:
                if key in ['turn_domain', 'turn_num']:
                    enc[key]=t[key]
                else:
                    enc[key]=self.modified_encode(t[key])
            encoded_dial.append(enc)
        return encoded_dial
    
    def encode_data(self, data, tokenizer, modular='dst'):
        encoded_data={}
        for set in data:
            encoded_data[set]=[]
            for item in data[set]:
                encoded_data[set].append([self.modified_encode(item[0], tokenizer), self.modified_encode(item[1], tokenizer)])
        return encoded_data

    def update_goal(self, init_goal, final_goal, user_act, sys_act):
        # constraint and sys_act are from last turn
        goal=deepcopy(init_goal)
        for domain in user_act:
            if domain not in goal:
                continue
            if sys_act and domain in sys_act and ('nooffer' in sys_act[domain] or 'nobook' in sys_act[domain])\
             and 'inform' in goal[domain] and domain in self.nooffer_options:
                # handle no offer situation: change one slot value
                intent='nooffer' if 'nooffer' in sys_act[domain] else 'nobook'
                if intent=='nooffer' or (intent=='nobook' and cfg.consider_nobook):
                    for slot in sys_act[domain][intent]:
                        if slot in goal[domain]['inform'] and slot in self.nooffer_options[domain]:
                            origin_value=goal[domain]['inform'][slot]
                            options=deepcopy(self.nooffer_options[domain][slot])
                            if origin_value in options:
                                options.pop(options.index(origin_value))
                            new_value=random.sample(options, 1)[0]
                            # goal change: change the current goal state and final goal
                            goal[domain]['inform'][slot]=new_value
                            final_goal[domain]['inform'][slot]=new_value
                            break
            else:
                # remove user act from goal state
                for intent, sv in user_act[domain].items():
                    if intent=='inform':
                        for slot, value in sv.items():
                            # In user act, price can express both price and pricerange
                            if slot=='price' and intent in goal[domain] and slot not in goal[domain][intent]:
                                slot='pricerange'
                            if  'inform' in goal[domain] and slot in goal[domain]['inform']:
                                if goal[domain]['inform'][slot]==value:
                                    goal[domain]['inform'].pop(slot)
                                if goal[domain]['inform']=={}:
                                    goal[domain].pop('inform')
                            elif 'book' in goal[domain] and slot in goal[domain]['book']:
                                if goal[domain]['book'][slot]==value:
                                    goal[domain]['book'].pop(slot)
                                if goal[domain]['book']=={}:
                                    goal[domain].pop('book')
                    elif intent=='request' and not cfg.same_policy_as_agenda:
                        # if same_policy_as_agenda, we do not delet the requestable slot of user act from goal state
                        for slot in sv:
                            if slot=='price' and intent in goal[domain] and slot not in goal[domain][intent]:
                                slot='pricerange'
                            if 'request' in goal[domain] and slot in goal[domain]['request']:
                                goal[domain]['request'].pop(goal[domain]['request'].index(slot))
                                if goal[domain]['request']==[]:
                                    goal[domain].pop('request')
                    if intent=='inform' and 'inform' in goal[domain] and goal[domain]['inform']=={}:
                        goal[domain].pop('inform')
            
            if goal[domain]=={}:
                goal.pop(domain)
        if sys_act and cfg.same_policy_as_agenda:
            for domain in sys_act:
                if domain not in goal:
                    continue
                for intent, slots in sys_act[domain].items():
                    # if system has inform the slot in last turn then user simulator needn't request
                    if intent=='inform' and 'request' in goal[domain]:
                        for slot in slots:
                            if slot=='pricerange' and slot not in goal[domain]['request']:
                                # there's only price slot in goal span
                                slot='price'
                            if slot in goal[domain]['request']:
                                goal[domain]['request'].pop(goal[domain]['request'].index(slot))
                        if goal[domain]['request']==[]:
                            goal[domain].pop('request')
                    if cfg.consider_offerbooked and intent=='offerbooked' and 'request' in goal[domain]:
                        for slot in slots:
                            if slot=='pricerange' and slot not in goal[domain]['request']:
                                # there's only price slot in goal span
                                slot='price'
                            if slot in goal[domain]['request']:
                                goal[domain]['request'].pop(goal[domain]['request'].index(slot))
                        if goal[domain]['request']==[]:
                            goal[domain].pop('request')
                if goal[domain]=={}:
                    goal.pop(domain)
                
        return goal, final_goal

    def update_bspn(self, pv_bspn, user_act):
        cons_dict={} if pv_bspn==None else self.bspan_to_constraint_dict(pv_bspn)
        user_act_dict=self.aspan_to_act_dict(user_act, side='user')
        for domain in user_act_dict:
            if domain not in cons_dict and 'inform' in user_act_dict[domain]:
                cons_dict[domain]={}
            for intent, sv in user_act_dict[domain].items():
                if intent=='inform':
                    for slot, value in sv.items():
                        if slot=='price' and (domain in ['restaurant', 'hotel'] or value in ['cheap', 'expensive', 'moderate']):
                            slot='pricerange'
                        cons_dict[domain].update({slot:value})
        bspn='<sos_b> ' + self.cons_dict_to_bspn(cons_dict) + ' <eos_b>'
        return bspn

    def correct_act(self, sys_act, user_act):
        domains=list(user_act.keys())
        new_sys_act=deepcopy(sys_act)
        if len(domains)>0:
            for domain in sys_act:
                if domain!='general' and domain not in domains:
                    new_domain=domains[-1]
                    if new_domain not in new_sys_act:
                        new_sys_act[new_domain]=new_sys_act.pop(domain)
        return new_sys_act

    def accumulate_goal(self, pv_goal, user_act):
        goal=deepcopy(pv_goal) if pv_goal else {}
        if not user_act and not pv_goal:
            return goal
        for domain in user_act:
            if domain=='general': # last turn and no specific slot
                continue
            if domain not in goal:
                goal.update({domain:{}})
            for intent, sv in user_act[domain].items():
                if intent=='inform':
                    for slot, value in sv.items():
                        if domain in ontology.book_domains and slot in ontology.book_domains[domain]:
                            intent='book'
                        if intent not in goal[domain]:
                            goal[domain].update({intent:{slot:value}})
                        else:
                            goal[domain][intent].update({slot:value})
                    if sv=={} and intent not in goal[domain] and domain in ['police', 'hospital']:
                        goal[domain].update({intent:{}})
                elif intent=='request':
                    for slot in sv:
                        if slot=='reference':
                            continue
                        if domain not in goal:
                            goal.update({domain:{intent:[slot]}})
                        elif intent not in goal[domain]:
                            goal[domain].update({intent:[slot]})
                        elif slot not in goal[domain][intent]:
                            goal[domain][intent].append(slot)   
                else:
                    print('Unknown intent:', intent)
        return goal


    def goal_to_gpan(self, goal, cur_domain=None):
        if goal=={}:
            return ''
        domain_gpan=[]
        domain_idx=0
        cur_domain_idx=-1
        for domain in goal:
            if domain==cur_domain:
                cur_domain_idx=domain_idx
            domain_idx+=1
            goal_list=[]
            goal_list.append('['+domain+']')
            for intent in goal[domain]:
                goal_list.append('['+intent+']')
                if isinstance(goal[domain][intent],dict):
                    for s, v in goal[domain][intent].items():
                        goal_list.append(s)
                        goal_list.append(v)
                elif isinstance(goal[domain][intent],list):
                    for s in goal[domain][intent]:
                        goal_list.append(s)
            domain_gpan.append(' '.join(goal_list))
        # current domain must be the last
        if cur_domain!='general' and cur_domain_idx>=0:
            domain_gpan[cur_domain_idx], domain_gpan[-1] = domain_gpan[-1], domain_gpan[cur_domain_idx]
        return ' '.join(domain_gpan)

    def goal_norm(self, goal):
        new_goal={}
        for domain in goal:
            new_goal[domain]={}
            for intent in goal[domain]:
                if intent in ['fail_book','fail_info']:
                    continue
                elif intent in ['info', 'book']:
                    new_intent='inform' if intent=='info' else intent
                    new_goal[domain][new_intent]={}
                    for slot, value in goal[domain][intent].items():
                        slot=slot.lower()
                        if slot in ['pre_invalid','invalid']:
                            continue
                        slot=ontology.normlize_slot_names.get(slot, slot)
                        slot='price' if slot=='pricerange' else slot
                        new_goal[domain][new_intent][slot]=value.lower()
                elif intent=='reqt':
                    new_goal[domain]['request']=[]
                    for slot in goal[domain][intent]:
                        slot=slot.lower()
                        slot=ontology.normlize_slot_names.get(slot, slot)
                        slot='price' if slot=='pricerange' else slot
                        new_goal[domain]['request'].append(slot)
                elif intent in ['inform', 'request', 'book']:
                    new_goal[domain][intent]=goal[domain][intent]
        
        return new_goal


    def bspan_to_constraint_dict(self, bspan, bspn_mode='bspn'):
        bspan = bspan.split() if isinstance(bspan, str) else bspan
        constraint_dict = {}
        domain = None
        conslen = len(bspan)
        for idx, cons in enumerate(bspan):
            if cons == '<eos_b>':
                break
            if '[' in cons:
                if cons[1:-1] not in ontology.all_domains:
                    continue
                domain = cons[1:-1]
            elif cons in ontology.get_slot:
                if domain is None:
                    continue
                if cons == 'people':
                    try:
                        ns = bspan[idx+1]
                        if ns == "'s":
                            continue
                    except:
                        continue
                if not constraint_dict.get(domain):
                    constraint_dict[domain] = {}
                if bspn_mode == 'bsdx':
                    constraint_dict[domain][cons] = 1
                    continue
                vidx = idx+1
                if vidx == conslen:
                    break
                vt_collect = []
                vt = bspan[vidx]
                while vidx < conslen and vt != '<eos_b>' and '[' not in vt and vt not in ontology.get_slot:
                    vt_collect.append(vt)
                    vidx += 1
                    if vidx == conslen:
                        break
                    vt = bspan[vidx]
                if vt_collect:
                    constraint_dict[domain][cons] = ' '.join(vt_collect)

        return constraint_dict

    def bspan_to_DBpointer(self, bspan, turn_domain):
        constraint_dict = self.bspan_to_constraint_dict(bspan)
        matnums = self.db.get_match_num(constraint_dict)
        if isinstance(turn_domain, str):
            turn_domain=turn_domain.split()
        match_dom = turn_domain[0] if len(turn_domain) == 1 else turn_domain[1]
        match_dom = match_dom[1:-1] if match_dom.startswith('[') else match_dom
        match = matnums[match_dom]
        # vector = self.db.addDBPointer(match_dom, match)
        vector = self.db.addDBIndicator(match_dom, match)
        return vector
    
    def aspan_to_act_list(self, aspan):
        aspan = aspan.split() if isinstance(aspan, str) else aspan
        acts = []
        domain = None
        conslen = len(aspan)
        for idx, cons in enumerate(aspan):
            if cons in ['<eos_a>', '<eos_ua>', '<eos_g>', '<eos_b>']:
                break
            if '[' in cons and cons[1:-1] in ontology.dialog_acts:
                domain = cons[1:-1]

            elif '[' in cons and cons[1:-1] in ontology.dialog_act_params:
                if domain is None:
                    continue
                vidx = idx+1
                if vidx == conslen:
                    acts.append(domain+'-'+cons[1:-1]+'-none')
                    break
                vt = aspan[vidx]
                no_param_act = True
                while vidx < conslen and vt not in ['<eos_a>', '<eos_ua>', '<eos_g>', '<eos_b>'] and '[' not in vt:
                    no_param_act = False
                    acts.append(domain+'-'+cons[1:-1]+'-'+vt)
                    vidx += 1
                    if vidx == conslen:
                        break
                    vt = aspan[vidx]
                if no_param_act:
                    acts.append(domain+'-'+cons[1:-1]+'-none')
        return acts

    def aspan_to_act_dict(self, aspan, side='sys'):
        assert side in ['sys', 'user'] # sys act or user act
        act_list=self.aspan_to_act_list(aspan)
        act_dict={}
        pv_slot=''
        if side=='sys':
            for act in act_list:
                if act.count('-')!=2:
                    continue
                domain, intent, slot = act.split('-')
                if domain not in act_dict:
                    act_dict[domain]={}
                if intent not in act_dict[domain]:
                    act_dict[domain][intent]=[]
                if slot not in act_dict[domain][intent]:
                    act_dict[domain][intent].append(slot)
        else:
            for act in act_list:
                if act.count('-')!=2:
                    continue
                domain, intent, slot = act.split('-')
                if domain not in act_dict:
                    act_dict[domain]={}
                if intent not in act_dict[domain]:
                    if intent in ['inform','book']:
                        act_dict[domain][intent]={}
                    elif intent=='request':
                        act_dict[domain][intent]=[]
                if intent in ['inform', 'book']:
                    if slot in ontology.all_slots:
                        act_dict[domain][intent][slot]='' 
                        pv_slot=slot                    
                    else:# slot is in fact a value in this condition
                        if pv_slot not in act_dict[domain][intent]:
                            continue
                        if act_dict[domain][intent][pv_slot]=='':
                            act_dict[domain][intent][pv_slot]=slot
                        else:
                            act_dict[domain][intent][pv_slot]+= ' '+slot
                elif intent=='request':
                    if slot not in act_dict[domain][intent]:
                        act_dict[domain][intent].append(slot)
        return act_dict

    def act_dict_to_aspan(self, act_dict):
        aspn=[]
        for domain in act_dict:
            aspn.append('['+domain+']')
            for intent, sv in act_dict[domain].items():
                aspn.append('['+intent+']')
                if isinstance(sv, dict):
                    for slot, value in sv.items():
                        if slot!='none':
                            aspn.extend([slot, value])
                elif isinstance(sv, list):
                    for slot in sv:
                        if slot!='none':
                            aspn.append(slot)
        return ' '.join(aspn)

    def add_reqt_to_bspn(self, bspn, user_act, return_type='span'):
        # both bspn and user_act are sequences
        # return_type: span/dict
        constraint_dict=self.bspan_to_constraint_dict(bspn)
        user_act_dict=self.aspan_to_act_dict(user_act, side='user')
        new_state={}
        for domain, sv in constraint_dict.items():
            new_state[domain]={'inform':sv}
        for domain in user_act_dict:
            if 'request' in user_act_dict[domain]:
                if domain in new_state:
                    new_state[domain]['request']=user_act_dict[domain]['request']
                else:
                    new_state[domain]={'request':user_act_dict[domain]['request']}
        if return_type=='span':
            new_span='<sos_b> '+ self.act_dict_to_aspan(new_state) + ' <eos_b>' if '<sos_b>' in bspn else self.act_dict_to_aspan(new_state)
            return new_span
        elif return_type=='dict':
            return new_state
    
    def delete_reqt_in_bspn(self, bspn, mode='token'):
        if mode=='id':
            bspn=self.tokenizer.decode(bspn)
        intent_dict=self.aspan_to_act_dict(bspn, side='user')
        new_dict={}
        for domain in intent_dict:
            if 'inform' in intent_dict[domain]:
                new_dict[domain]=intent_dict[domain]['inform']
        new_span='<sos_b> ' + self.cons_dict_to_bspn(new_dict) + ' <eos_b>' if '<sos_b>' in bspn else self.cons_dict_to_bspn(new_dict)
        if mode=='id':
            return self.modified_encode(new_span)
        return new_span

    def dspan_to_domain(self, dspan):
        domains = {}
        dspan = dspan.split() if isinstance(dspan, str) else dspan
        for d in dspan:
            if dom != '<eos_d>':
                domains[dom] = 1
            else:
                break
        return domains

    def get_sys_batch(self, data, batch_size=16, mode='train'):
        assert mode in ['train', 'test']
        batches=[]
        batch=[]
        seq_num=0
        for dial in data:
            for turn in dial:
                if mode=='train':
                    batch.append(turn['resp'])
                elif mode=='test':
                    batch.append(turn['resp_gen'])
                if len(batch)>=batch_size:
                    seq_num+=len(batch)
                    batch_np, _ = utils.padSeqs_gpt(batch, cfg.pad_id)
                    batches.append(batch_np)
                    batch=[]
        if batch!=[]:
            seq_num+=len(batch)
            batch_np, _ = utils.padSeqs_gpt(batch, cfg.pad_id)
            batches.append(batch_np)
            batch=[]
        print('Total responses:{}'.format(seq_num))
        return batches, seq_num


    def convert_batch_tokens_to_ids(self, dial_batch, tokenizer):
        new_batch=[]
        for dial in dial_batch:
            if isinstance(dial,list):
                new_dial=[]
                for turn in dial:
                    new_turn={}
                    for key in turn:
                        if key in ['user','bspn','aspn','resp','db', 'usr_act', 'bspn_gen', 'aspn_gen', 
                            'resp_gen', 'db_gen', 'user_gen', 'usr_act_gen', 'gpan', 'pv_aspn', 'goal']:
                            # GPT2Tokenizer of transformers3.5 needs to be modified
                            new_turn[key]=self.modified_encode(turn[key], tokenizer)
                        else:
                            new_turn[key]=turn[key]
                    new_dial.append(new_turn)
                new_batch.append(new_dial)
            elif isinstance(dial,dict):
                new_dial={}
                new_dial['goal']=self.modified_encode(dial['goal'], tokenizer)
                new_dial['log']=[]
                for turn in dial['log']:
                    new_turn={}
                    for key in turn:
                        if key in ['user','usdx','bspn','aspn','resp','bsdx','dspn','db', 'usr_act','bspn_gen', 'aspn_gen', 'resp_gen']:
                            # GPT2Tokenizer of transformers3.5 needs to be modified
                            new_turn[key]=self.modified_encode(turn[key], tokenizer)
                        else:
                            new_turn[key]=turn[key]
                    new_dial['log'].append(new_turn)
                new_batch.append(new_dial)
        return new_batch

    def convert_batch_ids_to_tokens(self, dial_batch):
        new_batch=[]
        for dial in dial_batch:
            new_dial=[]
            for turn in dial:
                new_turn={}
                for key in turn:
                    if isinstance(turn[key], list):
                        try:
                            new_turn[key]=self.tokenizer.decode(turn[key])
                        except Exception as e:
                            new_turn[key]=turn[key]
                new_dial.append(new_turn)
            new_batch.append(new_dial)
        return new_batch

    def transpose_ds_turn_batch(self, batch, rewards):
        turn_batches=[]
        label_batches=[]
        reward_batches=[]
        turn_batch=[]
        label_batch=[]
        reward_batch=[]
        if cfg.transpose_batch:
            p=0
            while(p*cfg.training_batch_size<len(batch)-1):
                batch_part=batch[p*cfg.training_batch_size:(p+1)*cfg.training_batch_size]
                reward_part=rewards[p*cfg.training_batch_size:(p+1)*cfg.training_batch_size]
                p+=1
                turn_id=0
                max_turn_num=max([len(dial) for dial in batch_part])
                for turn_id in range(max_turn_num):
                    if turn_id==0:
                        for dial, reward in zip(batch_part, reward_part):
                            if turn_id>len(dial)-1:
                                continue
                            turn=dial[turn_id]
                            R=reward[turn_id]
                            turn_batch.append(turn['user']+turn['bspn']+turn['db']+turn['aspn']+turn['resp'])
                            if cfg.rl_for_bspn and cfg.rl_for_resp:
                                label_batch.append(
                                    [cfg.pad_id]*len(turn['user'])+
                                    turn['bspn']+turn['db']+turn['aspn']+turn['resp']
                                    )
                            elif cfg.rl_for_resp:
                                label_batch.append(
                                    [cfg.pad_id]*len(turn['user']+turn['bspn']+turn['db'])+
                                    turn['aspn']+turn['resp']
                                    )
                            else:
                                label_batch.append(
                                    [cfg.pad_id]*len(turn['user']+turn['bspn']+turn['db'])+
                                    turn['aspn']+
                                    [cfg.pad_id]*len(turn['resp'])
                                    )
                            reward_batch.append(R)
                    else:
                        for dial, reward in zip(batch_part, reward_part):
                            if turn_id>len(dial)-1:
                                continue
                            turn=dial[turn_id]
                            pv_turn=dial[turn_id-1]
                            R=reward[turn_id]
                            turn_batch.append(pv_turn['bspn']+pv_turn['resp']+turn['user']+turn['bspn']+turn['db']+turn['aspn']+turn['resp'])
                            if cfg.rl_for_bspn and cfg.rl_for_resp:
                                label_batch.append(
                                    [cfg.pad_id]*len(pv_turn['bspn']+pv_turn['resp']+turn['user'])+
                                    turn['bspn']+turn['db']+turn['aspn']+turn['resp']
                                    )
                            elif cfg.rl_for_resp:
                                label_batch.append(
                                    [cfg.pad_id]*len(pv_turn['bspn']+pv_turn['resp']+turn['user']+turn['bspn']+turn['db'])+
                                    turn['aspn']+turn['resp']
                                    )
                            else:
                                label_batch.append(
                                    [cfg.pad_id]*len(pv_turn['bspn']+pv_turn['resp']+turn['user']+turn['bspn']+turn['db'])+
                                    turn['aspn']+
                                    [cfg.pad_id]*len(turn['resp'])
                                    )
                            reward_batch.append(R)
                    
                    if len(turn_batch)>cfg.training_batch_size/2:
                        turn_batch_np, _ = utils.padSeqs_gpt(turn_batch, cfg.pad_id)
                        label_batch_np, _ = utils.padSeqs_gpt(label_batch, cfg.pad_id)
                        turn_batches.append(turn_batch_np)
                        label_batches.append(label_batch_np)
                        reward_batches.append(reward_batch)
                        turn_batch=[]
                        label_batch=[]
                        reward_batch=[]
    
        else:
            for dial, reward in zip(batch, rewards):
                pv_turn=None
                for turn, R in zip(dial, reward):
                    if pv_turn:
                        turn_batch.append(pv_turn['bspn']+pv_turn['resp']+turn['user']+turn['bspn']+turn['db']+turn['aspn']+turn['resp'])
                        if cfg.rl_for_bspn and cfg.rl_for_resp:
                            label_batch.append(
                                [cfg.pad_id]*len(pv_turn['bspn']+pv_turn['resp']+turn['user'])+
                                turn['bspn']+turn['db']+turn['aspn']+turn['resp']
                                )
                        elif cfg.rl_for_resp:
                            label_batch.append(
                                [cfg.pad_id]*len(pv_turn['bspn']+pv_turn['resp']+turn['user']+turn['bspn']+turn['db'])+
                                turn['aspn']+turn['resp']
                                )
                        else:
                            label_batch.append(
                                [cfg.pad_id]*len(pv_turn['bspn']+pv_turn['resp']+turn['user']+turn['bspn']+turn['db'])+
                                turn['aspn']+
                                [cfg.pad_id]*len(turn['resp'])
                                    )
                    else:
                        turn_batch.append(turn['user']+turn['bspn']+turn['db']+turn['aspn']+turn['resp'])
                        if cfg.rl_for_bspn and cfg.rl_for_resp:
                            label_batch.append(
                                [cfg.pad_id]*len(turn['user'])+
                                turn['bspn']+turn['db']+turn['aspn']+turn['resp']
                                )
                        elif cfg.rl_for_resp:
                            label_batch.append(
                                [cfg.pad_id]*len(turn['user']+turn['bspn']+turn['db'])+
                                turn['aspn']+turn['resp']
                                )
                        else:
                            label_batch.append(
                                [cfg.pad_id]*len(turn['user']+turn['bspn']+turn['db'])+
                                turn['aspn']+
                                [cfg.pad_id]*len(turn['resp'])
                                )
                    reward_batch.append(R)
                    if len(turn_batch)==cfg.training_batch_size:
                        turn_batch_np, _ = utils.padSeqs_gpt(turn_batch, cfg.pad_id)
                        label_batch_np, _ = utils.padSeqs_gpt(label_batch, cfg.pad_id)
                        turn_batches.append(turn_batch_np)
                        label_batches.append(label_batch_np)
                        reward_batches.append(reward_batch)
                        turn_batch=[]
                        label_batch=[]
                        reward_batch=[]
                    pv_turn=turn
            if turn_batch!=[]:
                turn_batch_np, _ = utils.padSeqs_gpt(turn_batch, cfg.pad_id)
                label_batch_np, _ = utils.padSeqs_gpt(label_batch, cfg.pad_id)
                turn_batches.append(turn_batch_np)
                label_batches.append(label_batch_np)
                reward_batches.append(reward_batch)
        return turn_batches, label_batches, reward_batches
    
    def transpose_us_turn_batch(self, batch, rewards, tokenizer):
        turn_batches=[]
        label_batches=[]
        reward_batches=[]
        turn_batch=[]
        label_batch=[]
        reward_batch=[]
        if cfg.transpose_batch:
            p=0
            while(p*cfg.training_batch_size<len(batch)-1):
                batch_part=batch[p*cfg.training_batch_size:(p+1)*cfg.training_batch_size]
                reward_part=rewards[p*cfg.training_batch_size:(p+1)*cfg.training_batch_size]
                p+=1
                turn_id=0
                max_turn_num=max([len(dial) for dial in batch_part])
                for turn_id in range(max_turn_num):
                    if turn_id==0:
                        for dial, reward in zip(batch_part, reward_part):
                            if turn_id>len(dial)-1:
                                continue
                            turn=dial[turn_id]
                            R=reward[turn_id]
                            gpan=turn['goal'] if 'goal' in turn else turn['gpan']
                            turn_batch.append(gpan+turn['usr_act']+turn['user'])
                            label_batch.append([cfg.pad_id]*len(gpan)+turn['usr_act']+turn['user'])
                            reward_batch.append(R)
                    else:
                        for dial, reward in zip(batch_part, reward_part):
                            if turn_id>len(dial)-1:
                                continue
                            turn=dial[turn_id]
                            pv_turn=dial[turn_id-1]
                            R=reward[turn_id]
                            pv_aspn=turn['pv_aspn'] if 'pv_aspn' in turn else pv_turn['aspn']
                            #turn_batch.append(pv_turn['resp']+pv_aspn+turn['gpan']+turn['usr_act']+turn['user'])
                            pv_batch=pv_turn['resp']+pv_aspn if cfg.user_nlu else pv_turn['resp']
                            gpan=turn['goal'] if 'goal' in turn else turn['gpan']
                            turn_batch.append(pv_batch+gpan+turn['usr_act']+turn['user'])
                            label_batch.append([cfg.pad_id]*len(pv_batch+gpan)+\
                                    turn['usr_act']+turn['user'])
                            reward_batch.append(R)
                    if len(turn_batch)>cfg.training_batch_size/2:
                        turn_batch_np, _ = utils.padSeqs_gpt(turn_batch, cfg.pad_id)
                        label_batch_np, _ = utils.padSeqs_gpt(label_batch, cfg.pad_id)
                        turn_batches.append(turn_batch_np)
                        label_batches.append(label_batch_np)
                        reward_batches.append(reward_batch)
                        turn_batch=[]
                        label_batch=[]
                        reward_batch=[]
        else:
            for dial, reward in zip(batch, rewards):
                pv_turn=None
                for turn, R in zip(dial, reward):
                    if pv_turn is None:
                        gpan=turn['goal'] if 'goal' in turn else turn['gpan']
                        turn_batch.append(gpan+turn['usr_act']+turn['user'])
                        label_batch.append([cfg.pad_id]*len(gpan)+turn['usr_act']+turn['user'])
                    else:
                        pv_aspn=turn['pv_aspn'] if 'pv_aspn' in turn else pv_turn['aspn']
                        pv_batch=pv_turn['resp']+pv_aspn if cfg.user_nlu else pv_turn['resp']
                        gpan=turn['goal'] if 'goal' in turn else turn['gpan']
                        turn_batch.append(pv_batch+gpan+turn['usr_act']+turn['user'])
                        label_batch.append([cfg.pad_id]*len(pv_batch+gpan)+\
                                turn['usr_act']+turn['user'])
                    reward_batch.append(R)
                    if len(turn_batch)==cfg.training_batch_size:
                        turn_batch_np, _ = utils.padSeqs_gpt(turn_batch, cfg.pad_id)
                        label_batch_np, _ = utils.padSeqs_gpt(label_batch, cfg.pad_id)
                        turn_batches.append(turn_batch_np)
                        label_batches.append(label_batch_np)
                        reward_batches.append(reward_batch)
                        turn_batch=[]
                        label_batch=[]
                        reward_batch=[]
                    pv_turn=turn
            if turn_batch!=[]:
                turn_batch_np, _ = utils.padSeqs_gpt(turn_batch, cfg.pad_id)
                label_batch_np, _ = utils.padSeqs_gpt(label_batch, cfg.pad_id)
                turn_batches.append(turn_batch_np)
                label_batches.append(label_batch_np)
                reward_batches.append(reward_batch)
        return turn_batches, label_batches, reward_batches

    def modified_encode(self, text, tokenizer=None):
        if tokenizer is None:
            tokenizer=self.tokenizer
        if int(transformers.__version__[0])>=3:
            if isinstance(text, str):
                word_list=text.split()
            elif isinstance(text, list):
                word_list=text
            else:             
                raise TypeError(text)
            special_token_pos=[]
            results=[]
            for idx, word in enumerate(word_list):
                if word in tokenizer.additional_special_tokens:
                    special_token_pos.append(idx)
            for j, idx in enumerate(special_token_pos):
                if j<len(special_token_pos)-1:
                    next_idx=special_token_pos[j+1]
                    results+=tokenizer.encode(word_list[idx]) + tokenizer.encode(' '+' '.join(word_list[idx+1:next_idx]))
                else:
                    results+=tokenizer.encode(word_list[idx])
                    if idx<len(word_list)-1:# the last word is not a special token
                        results+=tokenizer.encode(' '+' '.join(word_list[idx+1:]))
            return results

        else:
            return tokenizer.encode(text)

    def batch_align(self,contexts,left_len,return_attn=False):
        max_len=max([len(context) for context in contexts])
        max_len=min(1024-left_len,max_len)
        new_contexts=[]
        attentions=[]
        for id, context in enumerate(contexts):
            if len(context)<max_len:
                new_context=(max_len-len(context))*[cfg.pad_id]+context
                attention=(max_len-len(context))*[0]+len(context)*[1]
            else:
                new_context=context[-max_len:]
                attention=len(new_context)*[1]
            new_contexts.append(new_context)
            attentions.append(attention)
        if return_attn:
            return new_contexts, attentions
        return new_contexts

    def convert_batch_session(self, dial_batch,
        posterior_train=False, only_resp_label=False,bspn_label=False,bspn_pri=False,rl_train=False):
        """
        convert the whole session for training
        concat [U_0, B_0, A_0, R_0, ... , U_n, B_n, A_n, R_n]

        try: [user, bspn, aspn, resp]
        or
        try: [user, bspn, db, aspn, resp]
        """
        inputs = {}
        labels={}
        bspn_labels={}
        contexts = []
        label_contexts=[]
        bspn_label_contexts=[]
        if not posterior_train:
            if cfg.model_act:
                cell_list = ['user', 'bspn', 'db','aspn', 'resp']
                ignore_list= ['user','bspn','db','aspn'] if only_resp_label else ['user']
            else:
                cell_list = ['user', 'bspn', 'db', 'resp']
                ignore_list=['user','bspn','db'] if only_resp_label else ['user','db']
            
        else:
            if cfg.model_act:
                cell_list=['user','resp','bspn','db','aspn']
                ignore_list=['user','resp']
            else:
                cell_list=['user','resp','bspn']
                ignore_list=['user','resp']

        if rl_train:
            cell_list=['user', 'bspn', 'db','aspn', 'resp']
            if cfg.turn_level_reward and not cfg.rl_with_us:#only calculate cross-entropy on response:
                ignor_list=['user','bspn','db','aspn']
            else:
                ignore_list=['user'] if cfg.rl_for_bspn else ['user','bspn','db']
        
        for idx, dial in enumerate(dial_batch):
            context = []
            label_context=[]
            bspn_label_context=[]
            Dial=dial['log'] if isinstance(dial, dict) else dial
    
            for turn_num, turn in enumerate(Dial):
                for cell in cell_list:
                    if cell=='bspn' and bspn_pri and 'bspn_pri' in turn:
                        cell='bspn_pri'
          
                    if cell=='db':
                        if bspn_pri and 'db_pri' in turn:
                            cell='db_pri'
                        else:
                            db_result=self.bspan_to_DBpointer(self.tokenizer.decode(turn['bspn']), turn['turn_domain'])
                            turn[cell] = self.tokenizer.encode('<sos_db> '+ db_result + ' <eos_db>')
                    context.extend(turn[cell])
                    if cell in ignore_list:
                        label_context.extend(len(turn[cell])*[cfg.pad_id])#pad_id在计算损失时被自动忽略
                    else:
                        label_context.extend(turn[cell])
                    if bspn_label:
                        bspn_cell_list=['bspn','db','aspn'] if cfg.model_act else ['bspn']
                        if cell in bspn_cell_list:
                            bspn_label_context.extend(turn[cell])
                        else:
                            bspn_label_context.extend(len(turn[cell])*[cfg.pad_id])
            
            contexts.append(context)
            label_contexts.append(label_context)
            bspn_label_contexts.append(bspn_label_context)

        
        inputs['contexts'] = contexts
        inputs['contexts_np'], inputs['lengths'] = utils.padSeqs_gpt(inputs['contexts'], cfg.pad_id)
        labels['contexts']=label_contexts
        labels['contexts_np'], labels['lengths']=utils.padSeqs_gpt(labels['contexts'], cfg.pad_id)

        if not bspn_label:
            return inputs,labels
        else:
            bspn_labels['contexts']=bspn_label_contexts
            bspn_labels['contexts_np'],bspn_labels['lengths']=utils.padSeqs_gpt(bspn_labels['contexts'], cfg.pad_id)
            return inputs,labels,bspn_labels

    def get_pv_batch(self, pv_batch, user=None, resp=None, bspn=None, aspn=None, goal=None, user_act=None, side='sys'):
        assert side in ['sys', 'user']
        new_pv_batch=[] # pv_batch for next turn
        if side=='sys':
            if pv_batch is None:# first turn
                for u, r, b in zip(user, resp, bspn): 
                    if cfg.input_history:
                        new_pv_batch.append(u+r)
                    elif cfg.input_prev_resp:
                        new_pv_batch.append(b+r)
                    else:
                        new_pv_batch.append(b)
            else:
                for hist, u, r, b in zip(pv_batch,user, resp, bspn):
                    if cfg.input_history:
                        new_pv_batch.append(hist+u+r)
                    elif cfg.input_prev_resp:
                        new_pv_batch.append(b+r)
                    else:
                        new_pv_batch.append(b)
        else:# user's pv batch
            if cfg.gen_goal_state:
                for g, ua, r in zip(goal, user_act, resp):
                    new_pv_batch.append(g+ua+r)
            else:
                for r, a in zip(resp, aspn):     
                    if cfg.user_nlu:
                        new_pv_batch.append(r+a)
                    else:
                        new_pv_batch.append(r)
        return new_pv_batch

    def convert_batch_turn(self, 
        turn_batch, 
        pv_batch, 
        first_turn=False, 
        rl_train=False, 
        mode='oracle', 
        side='sys', 
        posterior=False,
        seg_label=False,
        init_goals=None
        ):
        '''
        Args:
        Returns:
        '''
        assert mode in ['oracle', 'gen']
        assert side in ['sys', 'user']
        inputs = {}
        labels = {}
        contexts = []
        label_contexts = []
        if rl_train:
            rl_labels={}
            rl_label_contexts=[]
        if seg_label:
            seg_labels={}
            seg_contexts=[]
        if side=='sys':
            if first_turn:
                if mode=='oracle':
                    bspn_batch=turn_batch['bspn-ex'] if cfg.sys_nlu else turn_batch['bspn']
                    batch_zipped = zip(turn_batch['user'], bspn_batch, 
                        turn_batch['db'], turn_batch['aspn'], turn_batch['resp'])
                else:
                    bspn_batch=turn_batch['bspn-ex_gen'] if cfg.sys_nlu else turn_batch['bspn_gen']
                    batch_zipped=zip(turn_batch['user'], bspn_batch, 
                        turn_batch['db_gen'], turn_batch['aspn_gen'], turn_batch['resp_gen'])
                    
                for u, b, db, a, r in batch_zipped:
                    if posterior:
                        context=u+r + b+db+a
                        label_context=len(u+r)*[cfg.pad_id] + b+db+a
                    else:
                        context = u+b+db+a+r
                        label_context=len(u)*[cfg.pad_id]+b+db+a+r
                    contexts.append(context)
                    label_contexts.append(label_context)
                    if rl_train:
                        # 1 for belief state, 2 for system act, 3 for response and 0 for others
                        rl_label_context=len(u)*[0]+len(b)*[1]+len(db)*[0]+len(a)*[2]+len(r)*[3]
                        rl_label_contexts.append(rl_label_context)
                    if seg_label:
                        # 1 for hidden state, 2 for response, 0 for others
                        seg_context=len(u)*[0] + len(b+db+a)*[1] + len(r)*[2]
                        seg_contexts.append(seg_context)
            else:
                if mode=='oracle':
                    bspn_batch=turn_batch['bspn-ex'] if cfg.sys_nlu else turn_batch['bspn']
                    batch_zipped = zip(pv_batch,turn_batch['user'], bspn_batch, 
                        turn_batch['db'], turn_batch['aspn'], turn_batch['resp'])
                else:
                    bspn_batch=turn_batch['bspn-ex_gen'] if cfg.sys_nlu else turn_batch['bspn_gen']
                    batch_zipped = zip(pv_batch,turn_batch['user'], bspn_batch, 
                        turn_batch['db_gen'], turn_batch['aspn_gen'], turn_batch['resp_gen'])
                for ur, u, b, db, a, r in batch_zipped:
                    if posterior:
                        context = ur + u + r + b + db + a
                        label_context=len(ur+u+r)*[cfg.pad_id] + b+db+a
                    else:
                        context = ur + u + b + db + a + r
                        label_context=len(ur+u)*[cfg.pad_id]+b+db+a+r
                    contexts.append(context)
                    label_contexts.append(label_context)
                    if rl_train:
                        # 1 for belief state, 2 for system act, 3 for response and 0 for others
                        rl_label_context=len(ur+u)*[0]+len(b)*[1]+len(db)*[0]+len(a)*[2]+len(r)*[3]
                        rl_label_contexts.append(rl_label_context)
                    if seg_label:
                        # 1 for hidden state, 2 for response, 0 for others
                        seg_context=len(ur+u)*[0] + len(b+db+a)*[1] + len(r)*[2]
                        seg_contexts.append(seg_context)
        
        elif side=='user':
            if first_turn:
                if cfg.full_goal:
                    gpan_batch=init_goals
                else:
                    gpan_batch=turn_batch['goal'] if 'goal' in turn_batch else turn_batch['gpan']
                if mode=='oracle':
                    batch_zipped = zip(gpan_batch, turn_batch['usr_act'], turn_batch['user'])
                else:
                    batch_zipped=zip(gpan_batch, turn_batch['usr_act_gen'], turn_batch['user_gen'])                   
                for g, ua, u in batch_zipped:
                    if posterior:
                        context = u + ua
                        label_context = len(u)*[cfg.pad_id] + ua
                    else:
                        context = g + ua + u
                        label_context = len(g)*[cfg.pad_id]+ua+u
                    #context = g + [self.sos_r_id, self.eos_r_id] + ua + u
                    #label_context=(len(g)+2)*[cfg.pad_id]+ua+u
                    contexts.append(context)
                    label_contexts.append(label_context)
            else:
                if cfg.full_goal:
                    gpan_batch=init_goals
                else:
                    gpan_batch=turn_batch['goal'] if 'goal' in turn_batch else turn_batch['gpan']
                if mode=='oracle':
                    batch_zipped = zip(pv_batch, gpan_batch, turn_batch['usr_act'], turn_batch['user'])
                else:
                    batch_zipped=zip(pv_batch, gpan_batch, turn_batch['usr_act_gen'], turn_batch['user_gen'])                   
                for pv, g, ua, u in batch_zipped:
                    if posterior:
                        context = u + ua
                        label_context = len(u)*[cfg.pad_id] + ua
                    else:
                        context = pv + g + ua + u
                        label_context=len(pv+g)*[cfg.pad_id]+ua+u
                    contexts.append(context)
                    label_contexts.append(label_context)

        inputs['contexts'] = contexts
        inputs['contexts_np'], inputs['lengths'] = utils.padSeqs_gpt(inputs['contexts'], cfg.pad_id)
        labels['contexts']=label_contexts
        labels['contexts_np'], labels['lengths']=utils.padSeqs_gpt(labels['contexts'], cfg.pad_id)
        if seg_label and side=='sys':
            seg_labels['contexts']=seg_contexts
            seg_labels['contexts_np'], seg_labels['lengths']=utils.padSeqs_gpt(seg_labels['contexts'], cfg.pad_id)
            return inputs, labels, seg_labels
        if rl_train and side=='sys':
            rl_labels['contexts']=rl_label_contexts
            rl_labels['contexts_np'], rl_labels['lengths']=utils.padSeqs_gpt(rl_labels['contexts'], cfg.pad_id)
            return inputs, labels, rl_labels
        else:
            return inputs, labels


    def convert_eval_batch_turn(self, turn_batch, pv_batch, mode='gen_bspn', bspn_gen=None, db_gen=None, posterior=False):
        eval_batch=[]
        assert mode in ['gen_bspn', 'gen_ar']
        if pv_batch is None:
            if mode=='gen_bspn':
                for u, r in zip(turn_batch['user'], turn_batch['resp']):
                    context=u+r+[self.sos_b_id] if posterior else u+[self.sos_b_id]
                    eval_batch.append(context)
            else:
                for u, b, d, r in zip(turn_batch['user'], bspn_gen, db_gen, turn_batch['resp']):
                    context=u+r+b+d+[self.sos_a_id] if posterior else u+b+d+[self.sos_a_id]
                    eval_batch.append(context)
        else:
            if mode=='gen_bspn':
                for hist, u, r in zip(pv_batch, turn_batch['user'], turn_batch['resp']):
                    context=hist+u+r+[self.sos_b_id] if posterior else hist+u+[self.sos_b_id]
                    eval_batch.append(context)
            else:
                for hist, u, b, d, r in zip(pv_batch, turn_batch['user'], bspn_gen, db_gen, turn_batch['resp']):
                    context=hist+u+r+b+d+[self.sos_a_id] if posterior else hist+u+b+d+[self.sos_a_id]
                    eval_batch.append(context)
        return eval_batch
    
    def convert_eval_batch_turn_us(self, turn_batch, pv_batch, user_act=None, nlu=False, final_goal_batch=None):
        eval_batch=[]
        if nlu:
            assert pv_batch # perform NLU except for the first turn
            for pv_r in pv_batch['resp']:
                eval_batch.append(pv_r + [self.sos_a_id])

        else:
            gpan_batch=turn_batch['gpan'] if not cfg.full_goal else final_goal_batch
            if user_act is None:# generate user act (and utterance)
                if pv_batch==None: # first turn
                    for g in gpan_batch:
                        eval_batch.append(g + [self.sos_ua_id])
                else:
                    for g, pv in zip(gpan_batch, pv_batch):
                        eval_batch.append(pv + g + [self.sos_ua_id])
            else:# generate user utterance
                if pv_batch==None:
                    for g, ua in zip(gpan_batch, user_act):
                        eval_batch.append(g + ua + [self.sos_u_id])
                else:
                    for g, pv, ua in zip(gpan_batch, pv_batch, user_act):
                        eval_batch.append(pv + g + ua + [self.sos_u_id])
        return eval_batch

    def wrap_result_lm(self, result_dict, eos_syntax=None):
        results = []
        eos_syntax = ontology.eos_tokens if not eos_syntax else eos_syntax
        sos_syntax = ontology.sos_tokens
        # ground truth bs, as, ds.. generate response
        field = ['user', 'bspn', 'bspn_gen', 'db', 'db_gen', 'aspn_gen', 'aspn', 'resp_gen', 'resp', 'bspn-ex_gen', 'bspn-ex']
        others = ['dial_id', 'turn_num']

        for dial_id, turns in result_dict.items():
            entry = {'dial_id': dial_id, 'turn_num': len(turns)}
            for f in field:
                entry[f] = '' # ???
            results.append(entry)
            for turn_idx, turn in enumerate(turns):
                entry = {'dial_id': dial_id}
                for key in turn:
                    if key in field:
                        v=self.tokenizer.decode(turn[key])
                        if key in eos_syntax:
                            v=v.split()
                            if eos_syntax[key] in v:
                                v.remove(eos_syntax[key])
                            if sos_syntax[key] in v:
                                v.remove(sos_syntax[key])
                            v = " ".join(v)
                        entry[key]=v
                    elif key in others:
                        v=turn[key]
                        entry[key]=v
                    else:
                        if 'prob' in key:
                            entry[key]=str(turn[key])
                results.append(entry)

        return results, field

class tod_dataset(Dataset):
    def __init__(self, data):
        self.data=data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

def train_collate_fn(batch):
    # item[0]: input text
    # item[1]: target text
    data=[item[0]+item[1] for item in batch]
    label=[[cfg.pad_id]*len(item[0])+item[1] for item in batch]
    data_np, _=utils.padSeqs_gpt(data, cfg.pad_id)
    label_np, _=utils.padSeqs_gpt(label, cfg.pad_id)
    data_tensor=torch.from_numpy(data_np).long()
    label_tensor=torch.from_numpy(label_np).long()
    return [data_tensor, label_tensor]

def test_collate_fn(batch):
    # prediction
    sos_id=batch[0][1][0]
    data=[item[0]+[sos_id] for item in batch]
    label=[item[1] for item in batch]
    return [data, label]

if __name__ == '__main__':
    pass