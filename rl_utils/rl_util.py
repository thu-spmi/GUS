def fix_act(act_dict):
    reqt_set=set(['phone', 'address', 'postcode'])
    reward=0
    for domain in act_dict:
        for intent in act_dict[domain]:
            if intent=='inform':
                inform_set=set(act_dict[domain]['inform'])
                if len(reqt_set&inform_set)>0:
                    inform_set=inform_set|reqt_set
                    reward=1
                act_dict[domain]['inform']=list(inform_set)
    return act_dict,reward

def repeat_token_reward(sent):
    pv_token=None
    repeat_num=0
    token_num=len(sent.split())
    reward=0
    for token in sent.split():
        if token==pv_token:
            repeat_num+=1
            reward-=repeat_num
        else:
            repeat_num=0
        pv_token=token
    return reward/token_num

def act_compare_reward(act1, act2):
    reward=0
    tp=0
    fp=0
    for domain in act1:
        for intent, slots in act1[domain].items():
            if domain not in act2:
                fp+=len(slots)
                continue
            if intent not in act2[domain]:
                if intent=='inform' and 'recommend' in act2[domain]:
                    intent='recommend'
                elif intent=='recommend' and 'inform' in act2[domain]:
                    intent='inform'
                else:
                    fp+=len(slots)
                    continue
            for slot in slots:
                if slot in act2[domain][intent]:
                    tp+=1
                else:
                    fp+=1
    reward = -0.1*fp
    return reward

def act_cons_compare_reward(act, cons):
    # compare user act with bspn
    reward=0
    tp=0
    fp=0
    for domain in act:
        if 'inform' not in act[domain]:
            continue
        if domain not in cons:
            fp+=len(act[domain]['inform'])
            continue
        for slot, value in act[domain]['inform'].items():
            if slot not in cons[domain]:
                fp+=1
            elif cons[domain][slot]!=value:
                fp+=1
            else:
                tp+=1
    reward = -0.2*fp
    return reward