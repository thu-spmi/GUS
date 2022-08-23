from re import match
from turtle import width
from config import global_config as cfg
from eval import MultiWozEvaluator
from reader import MultiWozReader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import json, random
import torch
import numpy as np
from mwzeval.metrics import Evaluator
import copy, re
from collections import Counter
import argparse
stopwords = ['and','are','as','at','be','been','but','by', 'for','however','if', 
             'not','of','on','or','so','the','there','was','were','whatever','whether','would']

tokenizer=GPT2Tokenizer.from_pretrained('experiments_21/DS-baseline/best_score_model')
reader = MultiWozReader(tokenizer)
evaluator = MultiWozEvaluator(reader)
std_evaluator=Evaluator(bleu=1, success=1, richness=0)


def prepare_for_std_eval(path=None, data=None):
    if path:
        data=json.load(open(path, 'r', encoding='utf-8'))
    new_data={}
    dials=evaluator.pack_dial(data)
    for dial_id in dials:
        new_data[dial_id]=[]
        dial=dials[dial_id]
        for turn in dial:
            if turn['user']=='':
                continue
            entry={}
            entry['response']=turn['resp_gen']
            entry['state']=reader.bspan_to_constraint_dict(turn['bspn_gen'])
            new_data[dial_id].append(entry)
    if path:
        new_path=path[:-5]+'std.json'
        json.dump(new_data, open(new_path, 'w'), indent=2)
    return new_data

def get_metrics_list(path, prepared=False, dial_order=None):
    results=json.load(open(path, 'r'))
    input_data=prepare_for_std_eval(data=results) if not prepared else results
    if dial_order:
        new_data={}
        for dial_id in dial_order:
            if dial_id not in input_data:
                print('No dial id:', dial_id)
                continue
            new_data[dial_id]=input_data[dial_id]
        input_data=new_data
    results, match_list, success_list, bleu_list = std_evaluator.evaluate(input_data, return_all=True)
    print(results)
    return match_list, success_list, bleu_list, list(input_data.keys())

def get_nooffer_slot():
    data=json.load(open('data/multi-woz-2.1-processed/data_for_us.json', 'r'))
    nooffer_dict={}
    for dial in data.values():
        for turn in dial:
            if '[nooffer]' in turn['sys_act']:
                sys_act=reader.aspan_to_act_dict(turn['sys_act'], side='sys')
                for domain in sys_act:
                    if 'nooffer' in sys_act[domain]:
                        if domain not in nooffer_dict:
                            nooffer_dict[domain]=[]
                        for slot in sys_act[domain]['nooffer']:
                            if slot not in nooffer_dict[domain]:
                                nooffer_dict[domain].append(slot)
    print(nooffer_dict)

def compare_init_goal():
    data0=json.load(open('data/multi-woz-2.1-processed/data_for_us0.json', 'r'))
    data=json.load(open('data/multi-woz-2.1-processed/data_for_us.json', 'r'))
    count, total = 0, 0
    total_slots, unequal_slots=0, 0
    for dial_id, dial in data.items():
        init_goal=reader.aspan_to_act_dict(dial[0]['goal'], side='user')
        init_goal0=reader.aspan_to_act_dict(data0[dial_id][0]['goal'], side='user')
        not_equal=False
        total+=1
        for domain in init_goal:
            if domain not in init_goal0:
                not_equal=True
                continue
            for intent in init_goal[domain]:
                if intent not in init_goal0[domain]:
                    not_equal=True
                    continue
                if isinstance(init_goal[domain][intent],dict):
                    if init_goal[domain][intent]!=init_goal0[domain][intent]:
                        not_equal=True
                        continue
                elif isinstance(init_goal[domain][intent], list):
                    if set(init_goal[domain][intent])!=set(init_goal0[domain][intent]):
                        not_equal=True
                        continue
        if not not_equal:
            count+=1
    print('Equal initial goal:', count, 'total:', total)

def analyze_unsuc_dial():
    data=json.load(open('analysis/gen_dials_100_unsuc.json', 'r'))      
    count=0
    for dial in data:
        flag=0
        for turn in dial['log']:
            #if 'not looking to make a booking' in turn['user']:
            if 'guest house'  in turn['user']:
                flag=1
                print()
        if flag:
            count+=1
    print(count)

def evaluate_dialog_with_ABUS(path):
    data=json.load(open(path, 'r'))
    success, match = 0, 0
    turn_num=0
    for dial_id, dial in data.items():
        s, m = evaluator.get_metrics(dial['goal'], dial['log'])
        success+=s
        match+=m
        turn_num+=len(dial['log'])
    print(success/len(data), match/len(data), turn_num/len(data))

def extract_goal_from_ABUS(path):
    data=json.load(open(path, 'r'))
    goal_list=[]
    for dial_id, dial in data.items():
        goal=dial['goal']
        new_goal=reader.unify_goal(goal)
        goal_list.append(new_goal)
    json.dump(goal_list, open('analysis/goal_list.json', 'w'), indent=2)

def get_lex_resp(path):
    result=json.load(open(path, 'r'))
    for dial in result:
        turn_domain=[]
        pv_b=None
        for turn in dial:
            bspn=turn['bspn']
            cons=reader.bspan_to_constraint_dict(bspn)
            cur_domain=list(cons.keys())
            if cur_domain==[]:
                turn_domain=['general']
            else:
                if len(cur_domain)==1:
                    turn_domain=cur_domain
                else:
                    if pv_b is None: # In rare cases, there are more than one domain in the first turn
                        max_slot_num=0 # We choose the domain with most slots as the current domain
                        for domain in cur_domain:
                            if len(cons[domain])>max_slot_num:
                                turn_domain=[domain]
                                max_slot_num=len(cons[domain])
                    else:
                        pv_domain=list(reader.bspan_to_constraint_dict(pv_b).keys())
                        for domain in cur_domain:
                            if domain not in pv_domain: # new domain
                                # if domains are all the same, self.domain will not change
                                turn_domain=[domain]
            pv_b=bspn
            turn['lex_resp']=lex_resp(turn['resp'], bspn, turn['aspn'], turn_domain)
    json.dump(result, open(path, 'w'), indent=2)
        
def lex_resp(resp, bspn, aspn, turn_domain):
    value_map={}
    restored = resp
    restored=restored.replace('<sos_r>','')
    restored=restored.replace('<eos_r>','')
    restored.strip()
    restored = restored.capitalize()
    restored = restored.replace(' -s', 's')
    restored = restored.replace(' -ly', 'ly')
    restored = restored.replace(' -er', 'er')
    constraint_dict=reader.bspan_to_constraint_dict(bspn)#{'hotel': {'stay': '3'}, 'restaurant': {'people': '4'}}
    mat_ents = reader.db.get_match_num(constraint_dict, True)
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
    act_dict=reader.aspan_to_act_dict(aspn)
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
        ents_list=reader.db.dbs[domain[-1]]
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
                    temp_ent=reader.db.dbs[domain][0]
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


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', default='none')
    args = parser.parse_args()
    evaluate_dialog_with_ABUS(path=args.result_path)
    
