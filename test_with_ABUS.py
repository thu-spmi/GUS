
# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
from convlab2.nlu.jointBERT.multiwoz import BERTNLU
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.nlg.template.multiwoz import TemplateNLG
from session import turn_level_session
from convlab2.dialog_agent import PipelineAgent
from convlab2.util.analysis_tool.analyzer import Analyzer
import random
import numpy as np
import torch
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cpu')
parser.add_argument('--path', default='none')
args = parser.parse_args()


slot_map={'leave':'leaveAt', 'leaveat':'leaveAt', 'arrive':'arriveBy', 'arriveby':'arriveBy',
          'id':'trainID', 'trainid':'trainID', 'car':'car type'}
def set_seed(r_seed):
    random.seed(r_seed)
    np.random.seed(r_seed)
    torch.manual_seed(r_seed)

def fix_goal(goal):
    for domain in goal:
        for intent, sv in goal[domain].items():
            if isinstance(sv, list):
                for i, slot in enumerate(sv):
                    goal[domain][intent][i]=slot_map.get(slot, slot)
            elif isinstance(sv, dict):
                for key in sv:
                    if key in slot_map:
                        goal[domain][intent][slot_map[key]]=goal[domain][intent].pop(key)
    return goal

def prepare_goal_list():
    data=json.load(open('data/multi-woz-2.1-processed/data_for_damd_fix.json', 'r', encoding='utf-8'))
    test_list = [l.strip().lower()
                     for l in open('data/MultiWOZ_2.1/testListFile.txt', 'r').readlines()]
    goal_list=[]
    for dial_id, dial in data.items():
        if dial_id in test_list:
            goal=dial['goal']
            goal=fix_goal(goal)
            goal_list.append(goal)
    return goal_list

def get_ABUS(cuda_device='cpu'):
    user_nlu = BERTNLU(mode='sys', config_file='multiwoz_sys_context.json', 
        model_file='bert_nlu/bert_multiwoz_sys_context.zip', device=cuda_device)
    user_dst = None
    user_policy = RulePolicy(character='usr')
    user_nlg = TemplateNLG(is_user=True)
    user_agent = PipelineAgent(user_nlu, user_dst, user_policy, user_nlg, name='user')
    analyzer = Analyzer(user_agent=user_agent, dataset='multiwoz')
    return analyzer

def test_end2end():
    print('Model path:', args.path)
    sys_agent=turn_level_session(DS_path=args.path, device1=args.device)
    analyzer = get_ABUS(args.device)

    set_seed(20200202)
    dial_nums=1000
    save_path='analysis/' + args.path.split('/')[-2]+'_{}.json'.format(dial_nums)
    analyzer.comprehensive_analyze(sys_agent=sys_agent, model_name='turn-level-GPT',\
        total_dialog=dial_nums, save_path=save_path, return_dial=False)

def test_policy():
    user_policy = RulePolicy(character='usr')
    dial_num=10
    #goal_seeds = [random.randint(1,100000) for _ in range(dial_num)]
    #print(goal_seeds)
    for i in range(dial_num):
        #set_seed(goal_seeds[i])
        user_policy.init_session()
        goal=user_policy.get_goal()
        print(goal)

if __name__ == '__main__':
    test_end2end()