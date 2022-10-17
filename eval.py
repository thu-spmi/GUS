import math, logging, copy, json
from collections import Counter, OrderedDict
from nltk.util import ngrams
import os

import ontology
from config import global_config as cfg
from clean_dataset import clean_slot_values
from lexical_diversity import lex_div as ld

class BLEUScorer(object):
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def __init__(self):
        pass

    def score(self, parallel_corpus):

        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        
        weights = [0.25, 0.25, 0.25, 0.25]
        p0 = 1e-7

        # accumulate ngram statistics
        for hyps, refs in parallel_corpus:
            hyps = [hyp.split() for hyp in hyps]
            refs = [ref.split() for ref in refs]
            for hyp in hyps:

                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng])) \
                                   for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0: break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)

            


        # computing bleu score
        bp = 1 if c > r else math.exp(1 - float(r) / float(c))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
                for i in range(4)]
        s = math.fsum(w * math.log(p_n) \
                      for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        return bleu * 100


class MultiWozEvaluator(object):

    def __init__(self, reader):
        self.reader = reader
        self.domains = ontology.all_domains
        self.domain_files = self.reader.domain_files
        self.all_data = self.reader.data
        self.test_data = self.reader.test

        self.bleu_scorer = BLEUScorer()

        self.all_info_slot = []
        for d, s_list in ontology.informable_slots.items():
            for s in s_list:
                self.all_info_slot.append(d+'-'+s)
        self.requestables = ['phone', 'address', 'postcode', 'reference', 'id']


    def pack_dial(self, data):
        dials = {}
        for turn in data:
            dial_id = turn['dial_id']
            if dial_id not in dials:
                dials[dial_id] = []
            dials[dial_id].append(turn)
        return dials
    
    def get_metrics(self, final_goal, dial):
        reqs = {}
        goal = {}
        for domain in ontology.all_domains:
            if final_goal.get(domain):
                true_goal = final_goal
                goal = self._parseGoal(goal, true_goal, domain)
        for domain in goal.keys():
            reqs[domain] = goal[domain]['requestable']
        success, match, _ = self._evaluateGeneratedDialogue(dial, goal, reqs)
        return success, match


    def validation_metric(self, data, return_act_acc=False):
        bleu = self.bleu_metric(data)
        if 'test' in cfg.mode:
            self.get_bleu_list(data)
        # accu_single_dom, accu_multi_dom, multi_dom_num = self.domain_eval(data)
        success, match = self.context_to_response_eval(data)
        
        if return_act_acc:
            P, R, F1 = self.resp_eval(data)
            print('resp placeholder P/R/F1:', P, R, F1)
            act_f1,P,R, turn_acc = self.aspn_eval(data)
            return bleu, success, match, turn_acc, P, R, act_f1
        return bleu, success, match

    

    def bleu_metric(self, data, eval_dial_list=None):
        gen, truth = [],[]
        for row in data:
            if eval_dial_list and row['dial_id'] +'.json' not in eval_dial_list:
                continue
            gen.append(row['resp_gen'])
            truth.append(row['resp'])
        wrap_generated = [[_] for _ in gen]
        wrap_truth = [[_] for _ in truth]
        if gen and truth:
            sc = self.bleu_scorer.score(zip(wrap_generated, wrap_truth))
        else:
            sc = 0.0
        return sc
    
    def get_bleu_list(self, data):
        bleu_list=[]
        dials = self.pack_dial(data)
        for dial in dials.values():
            gen, truth=[], []
            for turn in dial:
                gen.append(turn['resp_gen'])
                truth.append(turn['resp'])
            wrap_generated = [[_] for _ in gen]
            wrap_truth = [[_] for _ in truth]
            sc = self.bleu_scorer.score(zip(wrap_generated, wrap_truth))
            bleu_list.append(sc)
        json.dump(bleu_list, open(os.path.join(cfg.eval_load_path, 'bleu_list.json'), 'w'))

    def bleu_metric_us(self, data):
        gen, truth = [],[]
        for dial in data:
            for turn in dial:
                gen.append(turn['user_gen'])
                truth.append(turn['user'])
        wrap_generated = [[_] for _ in gen]
        wrap_truth = [[_] for _ in truth]
        if gen and truth:
            sc = self.bleu_scorer.score(zip(wrap_generated, wrap_truth))
        else:
            sc = 0.0
        return sc

    def diversity_metric_us(self, data):
        gen, truth = [],[]
        for dial in data:
            for turn in dial:
                gen.append(turn['user_gen'])
                truth.append(turn['user'])
        gen_richness=get_richness(gen)
        truth_richness=get_richness(truth)
        return gen_richness, truth_richness

    def value_similar(self, a,b):
        return True if a==b else False

        # the value equal condition used in "Sequicity" is too loose
        if a in b or b in a or a.split()[0] == b.split()[0] or a.split()[-1] == b.split()[-1]:
            return True
        return False

    def _bspn_to_dict(self, bspn, no_name=False, no_book=False, bspn_mode = 'bspn'):
        constraint_dict = self.reader.bspan_to_constraint_dict(bspn, bspn_mode = bspn_mode)
        constraint_dict_flat = {}
        for domain, cons in constraint_dict.items():
            for s,v in cons.items():
                key = domain+'-'+s
                if no_name and s == 'name':
                    continue
                if no_book:
                    if s in ['people', 'stay'] or key in ['hotel-day', 'restaurant-day','restaurant-time'] :
                        continue
                constraint_dict_flat[key] = v
        return constraint_dict_flat

    def _constraint_compare(self, truth_cons, gen_cons, slot_appear_num=None, slot_correct_num=None):
        tp,fp,fn = 0,0,0
        false_slot = []
        for slot in gen_cons:
            v_gen = gen_cons[slot]
            if slot in truth_cons and self.value_similar(v_gen, truth_cons[slot]):  #v_truth = truth_cons[slot]
                tp += 1
                if slot_correct_num is not None:
                    slot_correct_num[slot] = 1 if not slot_correct_num.get(slot) else slot_correct_num.get(slot)+1
            else:
                fp += 1
                false_slot.append(slot)
        for slot in truth_cons:
            v_truth = truth_cons[slot]
            if slot_appear_num is not None:
                slot_appear_num[slot] = 1 if not slot_appear_num.get(slot) else slot_appear_num.get(slot)+1
            if slot not in gen_cons or not self.value_similar(v_truth, gen_cons[slot]):
                fn += 1
                false_slot.append(slot)
        acc = len(self.all_info_slot) - fp - fn
        return tp,fp,fn, acc, list(set(false_slot))

    def domain_eval(self, data, eval_dial_list = None):
        dials = self.pack_dial(data)
        corr_single, total_single, corr_multi, total_multi = 0, 0, 0, 0

        dial_num = 0
        for dial_id in dials:
            if eval_dial_list and dial_id+'.json' not in eval_dial_list:
                continue
            dial_num += 1
            dial = dials[dial_id]
            wrong_pred = []

            prev_constraint_dict = {}
            prev_turn_domain = ['general']

            for turn_num, turn in enumerate(dial):
                if turn_num == 0:
                    continue
                true_domains = self.reader.dspan_to_domain(turn['dspn'])
                if cfg.enable_dspn:
                    pred_domains = self.reader.dspan_to_domain(turn['dspn_gen'])
                else:
                    turn_dom_bs = []
                    if cfg.enable_bspn and not cfg.use_true_bspn_for_ctr_eval and \
                        (cfg.bspn_mode == 'bspn' or cfg.enable_dst):
                        constraint_dict = self.reader.bspan_to_constraint_dict(turn['bspn_gen'])
                    else:
                        constraint_dict = self.reader.bspan_to_constraint_dict(turn['bspn'])
                    for domain in constraint_dict:
                        if domain not in prev_constraint_dict:
                            turn_dom_bs.append(domain)
                        elif prev_constraint_dict[domain] != constraint_dict[domain]:
                            turn_dom_bs.append(domain)
                    aspn = 'aspn' if not cfg.enable_aspn else 'aspn_gen'
                    turn_dom_da = []
                    for a in turn[aspn].split():
                        if a[1:-1] in ontology.all_domains + ['general']:
                            turn_dom_da.append(a[1:-1])

                    # get turn domain
                    turn_domain = turn_dom_bs
                    for dom in turn_dom_da:
                        if dom != 'booking' and dom not in turn_domain:
                            turn_domain.append(dom)
                    if not turn_domain:
                        turn_domain = prev_turn_domain
                    if len(turn_domain) == 2 and 'general' in turn_domain:
                        turn_domain.remove('general')
                    if len(turn_domain) == 2:
                        if len(prev_turn_domain) == 1 and prev_turn_domain[0] == turn_domain[1]:
                            turn_domain = turn_domain[::-1]
                    prev_turn_domain = copy.deepcopy(turn_domain)
                    prev_constraint_dict = copy.deepcopy(constraint_dict)

                    turn['dspn_gen'] = ' '.join(['['+d+']' for d in turn_domain])
                    pred_domains = {}
                    for d in turn_domain:
                        pred_domains['['+d+']'] = 1

                if len(true_domains) == 1:
                    total_single += 1
                    if pred_domains == true_domains:
                        corr_single += 1
                    else:
                        wrong_pred.append(str(turn['turn_num']))
                        turn['wrong_domain'] = 'x'
                else:
                    total_multi += 1
                    if pred_domains == true_domains:
                        corr_multi += 1
                    else:
                        wrong_pred.append(str(turn['turn_num']))
                        turn['wrong_domain'] = 'x'

            # dialog inform metric record
            dial[0]['wrong_domain'] = ' '.join(wrong_pred)
        accu_single = corr_single / (total_single + 1e-10)
        accu_multi = corr_multi / (total_multi + 1e-10)
        return accu_single * 100, accu_multi * 100, total_multi


    def dialog_state_tracking_eval(self, data, eval_dial_list = None, bspn_mode='bspn', no_name=False, no_book=False):
        dials = self.pack_dial(data)
        total_turn, joint_match, total_tp, total_fp, total_fn, total_acc = 0, 0, 0, 0, 0, 0
        slot_appear_num, slot_correct_num = {}, {}
        dial_num = 0
        for dial_id in dials:
            if eval_dial_list and dial_id +'.json' not in eval_dial_list:
                continue
            dial_num += 1
            dial = dials[dial_id]
            missed_jg_turn_id = []
            for turn_num,turn in enumerate(dial):
                if turn_num == 0:
                    continue
                gen_cons = self._bspn_to_dict(turn[bspn_mode+'_gen'], no_name=no_name,
                                                                  no_book=no_book, bspn_mode=bspn_mode)
                truth_cons = self._bspn_to_dict(turn[bspn_mode], no_name=no_name,
                                                                   no_book=no_book, bspn_mode=bspn_mode)

                if truth_cons == gen_cons:
                    joint_match += 1
                else:
                    missed_jg_turn_id.append(str(turn['turn_num']))

                if eval_dial_list is None:
                    tp,fp,fn, acc, false_slots = self._constraint_compare(truth_cons, gen_cons,
                                                                                              slot_appear_num, slot_correct_num)
                else:
                    tp,fp,fn, acc, false_slots = self._constraint_compare(truth_cons, gen_cons,)

                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_acc += acc
                total_turn += 1
                if not no_name and not no_book:
                    turn['wrong_inform'] = '; '.join(false_slots)   # turn inform metric record

            # dialog inform metric record
            if not no_name and not no_book:
                dial[0]['wrong_inform'] = ' '.join(missed_jg_turn_id)

        precision = total_tp / (total_tp + total_fp + 1e-10)
        recall = total_tp / (total_tp + total_fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10) * 100
        accuracy = total_acc / (total_turn * len(self.all_info_slot) + 1e-10) * 100
        joint_goal = joint_match / (total_turn+1e-10) * 100


        return joint_goal, f1, accuracy, slot_appear_num, slot_correct_num

    def resp_eval(self, data):
        def _extract_plh(text):
            plh_list=[]
            for w in text.split():
                if '[value_' in w and w not in plh_list:
                    plh_list.append(w)
            return plh_list

        dials = self.pack_dial(data)
        tp=0
        fp=0
        fn=0
        for dial_id in dials:
            dial=dials[dial_id]
            for turn_num, turn in enumerate(dial):
                if turn_num==0:
                    continue
                plh_list=_extract_plh(turn['resp'])
                plh_list_gen=_extract_plh(turn['resp_gen'])
                for plh_gen in plh_list_gen:
                    if plh_gen in plh_list:
                        tp+=1
                    else:
                        fp+=1
                for plh in plh_list:
                    if plh not in plh_list_gen:
                        fn+=1
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        return precision, recall, f1

    def aspn_eval(self, data, eval_dial_list = None):

        def _get_tp_fp_fn(label_list, pred_list):
            tp = len([t for t in pred_list if t in label_list])
            fp = max(0, len(pred_list) - tp)
            fn = max(0, len(label_list) - tp)
            return tp, fp, fn

        dials = self.pack_dial(data)
        total_tp, total_fp, total_fn = 0, 0, 0

        dial_num = 0
        total_turn=0
        right_turn=0
        for dial_id in dials:
            if eval_dial_list and dial_id+'.json' not in eval_dial_list:
                continue
            dial_num += 1
            dial = dials[dial_id]
            wrong_act = []
            for turn_num, turn in enumerate(dial):
                if turn_num == 0:
                    continue
                total_turn+=1
                if turn['aspn']==turn['aspn_gen']:
                    right_turn+=1
                if cfg.same_eval_act_f1_as_hdsa:
                    pred_acts, true_acts = {}, {}
                    for t in turn['aspn_gen']:
                        pred_acts[t] = 1
                    for t in  turn['aspn']:
                        true_acts[t] = 1
                    tp, fp, fn = _get_tp_fp_fn(true_acts, pred_acts)
                else:
                    pred_acts = self.reader.aspan_to_act_list(turn['aspn_gen'])
                    true_acts = self.reader.aspan_to_act_list(turn['aspn'])
                    tp, fp, fn = _get_tp_fp_fn(true_acts, pred_acts)
                if fp + fn !=0:
                    wrong_act.append(str(turn['turn_num']))
                    turn['wrong_act'] = 'x'

                total_tp += tp
                total_fp += fp
                total_fn += fn

            dial[0]['wrong_act'] = ' '.join(wrong_act)
        precision = total_tp / (total_tp + total_fp + 1e-10)
        recall = total_tp / (total_tp + total_fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        turn_acc=right_turn/total_turn

        return f1, precision, recall, turn_acc

    def multi_act_eval(self, data, eval_dial_list = None):

        dials = self.pack_dial(data)
        total_act_num, total_slot_num = 0, 0

        dial_num = 0
        turn_count = 0
        for dial_id in dials:
            if eval_dial_list and dial_id+'.json' not in eval_dial_list:
                continue
            dial_num += 1
            dial = dials[dial_id]
            for turn_num, turn in enumerate(dial):
                if turn_num == 0:
                    continue
                target = turn['multi_act_gen'] if self.reader.multi_acts_record is not None else turn['aspn_gen']


                # diversity
                act_collect, slot_collect = {}, {}
                act_type_collect = {}
                slot_score = 0
                for act_str in target.split(' | '):
                    pred_acts = self.reader.aspan_to_act_list(act_str)
                    act_type = ''
                    for act in pred_acts:
                        d,a,s = act.split('-')
                        if d + '-' + a not in act_collect:
                            act_collect[d + '-' + a] = {s:1}
                            slot_score += 1
                            act_type += d + '-' + a + ';'
                        elif s not in act_collect:
                            act_collect[d + '-' + a][s] = 1
                            slot_score += 1
                        slot_collect[s] = 1
                    act_type_collect[act_type] = 1
                total_act_num += len(act_collect)
                total_slot_num += len(slot_collect)
                turn_count += 1

        total_act_num = total_act_num/(float(turn_count) + 1e-10)
        total_slot_num = total_slot_num/(float(turn_count) + 1e-10)
        return total_act_num, total_slot_num


    def context_to_response_eval(self, data, eval_dial_list = None):
        dials = self.pack_dial(data)

        dial_num, successes, matches = 0, 0, 0
        success_list=[]
        match_list=[]
        if cfg.col_samples:
            match_sample={}
            mismatch_sample={}
            success_sample={}
            unsuccess_sample={}
        
        for dial_id in dials:
            if eval_dial_list and dial_id +'.json' not in eval_dial_list:
                continue
            dial = dials[dial_id]
            reqs = {}
            goal = {}
            if '.json' not in dial_id and '.json' in list(self.all_data.keys())[0]:
                dial_id = dial_id + '.json'
            for domain in ontology.all_domains:
                if self.all_data[dial_id]['goal'].get(domain):
                    true_goal = self.all_data[dial_id]['goal']
                    goal = self._parseGoal(goal, true_goal, domain)
            # print(goal)
            for domain in goal.keys():
                reqs[domain] = goal[domain]['requestable']

            # print('\n',dial_id)
            success, match, _ = self._evaluateGeneratedDialogue(dial, goal, reqs)
            
            start_idx=0 if dial[0]['user']!='' else 1
            if cfg.col_samples:
                if match>0:
                    match_sample[dial_id]=dial[start_idx:]
                else:
                    mismatch_sample[dial_id]=dial[start_idx:]
                if success>0:
                    success_sample[dial_id]=dial[start_idx:]
                else:
                    unsuccess_sample[dial_id]=dial[start_idx:]

            successes += success
            matches += match
            dial_num += 1
            success_list.append(success)
            match_list.append(match)

        succ_rate = successes/( float(dial_num) + 1e-10) * 100
        match_rate = matches/(float(dial_num) + 1e-10) * 100
        if cfg.col_samples and 'test' in cfg.mode:
            # if cfg.rl_train=True then our validation is online
            mismatch_file='online_mismatch.json' if cfg.rl_train else 'offline_mismatch.json'
            unsuccess_file='online_unsuccess.json' if cfg.rl_train else 'offline_unsuccess.json'
            match_path=os.path.join(cfg.eval_load_path, mismatch_file)
            success_path=os.path.join(cfg.eval_load_path, unsuccess_file)
            #if not os.path.exists(match_path):
            #match_data={'match':match_sample,'mismatch':mismatch_sample}
            match_data=mismatch_sample
            json.dump(match_data, open(match_path, 'w'), indent=2)
            #if not os.path.exists(success_path):
            #success_data={'success':success_sample,'unsuccess':unsuccess_sample}
            success_data=unsuccess_sample
            json.dump(success_data, open(success_path, 'w'), indent=2)
        return succ_rate, match_rate


    def _evaluateGeneratedDialogue(self, dialog, goal, real_requestables, soft_acc=False):
        """Evaluates the dialogue created by the model.
            First we load the user goal of the dialogue, then for each turn
            generated by the system we look for key-words.
            For the Inform rate we look whether the entity was proposed.
            For the Success rate we look for requestables slots"""
        # for computing corpus success

        # CHECK IF MATCH HAPPENED
        provided_requestables = {}
        venue_offered = {}
        domains_in_goal = []
        bspans = {}

        for domain in goal.keys():
            venue_offered[domain] = []
            provided_requestables[domain] = []
            domains_in_goal.append(domain)

        for t, turn in enumerate(dialog):
            if t == 0 and turn['user']=='': 
                continue 
            if 'resp_gen' in turn:
                sent_t = turn['resp_gen']
            else: # evaluate the interaction quality between user simulator and dialog system
                sent_t = turn['resp']
            # sent_t = turn['resp']
            for domain in goal.keys():
                # for computing success
                if '[value_name]' in sent_t or '[value_id]' in sent_t:
                    if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                        # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION
                        if not cfg.use_true_curr_bspn and not cfg.use_true_bspn_for_ctr_eval:
                            if 'bspn_gen' in turn:
                                bspn = turn['bspn_gen']
                            else:# evaluate the interaction quality between user simulator and dialog system
                                bspn = turn['bspn']
                        else:
                            bspn = turn['bspn']
                        # bspn = turn['bspn']

                        constraint_dict = self.reader.bspan_to_constraint_dict(bspn)
                        if constraint_dict.get(domain):
                            venues = self.reader.db.queryJsons(domain, constraint_dict[domain], return_name=True)
                        else:
                            venues = []
                        # if venue has changed
                        if cfg.venue_overwrite:
                            if venues:
                                venue_offered[domain] = venues
                                bspans[domain] = constraint_dict[domain]
                        else:
                            if len(venue_offered[domain]) == 0 and venues:
                                # venue_offered[domain] = random.sample(venues, 1)
                                venue_offered[domain] = venues
                                bspans[domain] = constraint_dict[domain]
                            else:
                                # if 
                                flag = False
                                for ven in venues:
                                    if  ven not in venue_offered[domain]:
                                        flag = True
                                        break
                                # if flag and venues:
                                if flag and venues:  # sometimes there are no results so sample won't work
                                    venue_offered[domain] = venues
                                    bspans[domain] = constraint_dict[domain]
                            
                    else:  # not limited so we can provide one
                        venue_offered[domain] = '[value_name]'

                # ATTENTION: assumption here - we didn't provide phone or address twice! etc
                if cfg.strict_eval:
                    for requestable in ontology.requestable_slots[domain]:
                        if '[value_' + requestable + ']' in sent_t:
                                provided_requestables[domain].append(requestable)
                else:
                    for requestable in self.requestables:
                        if '[value_' + requestable + ']' in sent_t:
                                provided_requestables[domain].append(requestable)

        # if name was given in the task
        for domain in goal.keys():
            # if name was provided for the user, the match is being done automatically
            if 'name' in goal[domain]['informable']:
                venue_offered[domain] = '[value_name]'

            # special domains - entity does not need to be provided
            if domain in ['taxi', 'police', 'hospital']:
                venue_offered[domain] = '[value_name]'

            if domain == 'train':
                if not venue_offered[domain] and 'id' not in goal[domain]['requestable']:
                    venue_offered[domain] = '[value_name]'

        """
        Given all inform and requestable slots
        we go through each domain from the user goal
        and check whether right entity was provided and
        all requestable slots were given to the user.
        The dialogue is successful if that's the case for all domains.
        """
        # HARD EVAL
        stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                 'taxi': [0, 0, 0],
                 'hospital': [0, 0, 0], 'police': [0, 0, 0]}

        match = 0
        success = 0
        match_domain=[]
        success_domain=[]
        # MATCH
        #print('venue offered\n', venue_offered)
        for domain in goal.keys():
            match_stat = 0
            if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                
                goal_venues = self.reader.db.queryJsons(domain, goal[domain]['informable'], return_name=True)
                if cfg.eval_as_simpletod:
                    condition=len(venue_offered[domain]) > 0 and venue_offered[domain][0] in goal_venues
                else:
                    condition=len(venue_offered[domain]) > 0 and len(set(venue_offered[domain])& set(goal_venues))>0
                if type(venue_offered[domain]) is str and '_name' in venue_offered[domain]:
                    match += 1
                    match_stat = 1
                    match_domain.append(domain)
                elif condition:
                    match += 1
                    match_stat = 1
                    match_domain.append(domain)
            else:
                if '_name]' in venue_offered[domain]:
                    match += 1
                    match_stat = 1
                    match_domain.append(domain)

            stats[domain][0] = match_stat
            stats[domain][2] = 1

        if soft_acc:
            match = float(match)/len(goal.keys())
        else:
            if match == len(goal.keys()):
                match = 1.0
            else:
                match = 0.0

        # SUCCESS
        if match == 1.0:
            for domain in domains_in_goal:
                success_stat = 0
                domain_success = 0
                if len(real_requestables[domain]) == 0:

                    success += 1
                    success_stat = 1
                    stats[domain][1] = success_stat
                    success_domain.append(domain)
                    continue
                
                for request in real_requestables[domain]:
                    if request in provided_requestables[domain]:
                        domain_success += 1

                if domain_success >= len(real_requestables[domain]):
                    success += 1
                    success_domain.append(domain)
                    success_stat = 1

                stats[domain][1] = success_stat

            # final eval
            if soft_acc:
                success = float(success)/len(real_requestables)
            else:
                if success >= len(real_requestables):
                    success = 1
                else:
                    success = 0
        return success, match, stats


    def _parseGoal(self, goal, true_goal, domain):
        """Parses user goal into dictionary format."""
        goal[domain] = {}
        goal[domain] = {'informable': {}, 'requestable': [], 'booking': []}
        if 'info' in true_goal[domain]:
            true_goal[domain]['inform']=true_goal[domain].pop('info')
        if 'reqt' in true_goal[domain]:
            true_goal[domain]['request']=true_goal[domain].pop('reqt')

        
        if 'request' in true_goal[domain]:
            for s in true_goal[domain]['request']:
                if cfg.strict_eval:
                    if s in ontology.requestable_slots[domain]:
                        goal[domain]['requestable'].append(s)
                else:
                    if s in self.requestables:
                        goal[domain]['requestable'].append(s)
        if 'book' in true_goal[domain]:
            goal[domain]['requestable'].append('reference')
            goal[domain]["booking"] = true_goal[domain]['book']

        if 'inform' in true_goal[domain]:
            for s, v in true_goal[domain]['inform'].items():
                s_,v_ = clean_slot_values(domain, s,v)
                goal[domain]["informable"][s_] = v_

        return goal

    def evaluate_us(self, data):
        bleu=self.bleu_metric_us(data)
        gen_richness, truth_richness=self.diversity_metric_us(data)
        logging.info('Generate diversity:{}'.format(gen_richness))
        logging.info('Oracle diversity:{}'.format(truth_richness))
        total_turn=0
        total_p=0
        total_r=0
        total_f1=0
        tp=0
        fp=0
        fn=0
        eps=1e-10
        for dial in data:
            for turn in dial:
                total_turn+=1
                ua=self.reader.aspan_to_act_dict(turn['usr_act'], side='user') # this transposing is not enough
                ua_gen=self.reader.aspan_to_act_dict(turn['usr_act_gen'], side='user')
                for domain in ua_gen:
                    for intent in ua_gen[domain]:
                        for slot in ua_gen[domain][intent]:
                            if domain in ua:
                                if intent in ua[domain]:
                                    if slot in ua[domain][intent]:
                                        if intent=='inform':
                                            if ua[domain][intent][slot]==ua_gen[domain][intent][slot]:
                                                tp+=1
                                            else:
                                                fp+=1
                                        else:
                                            tp+=1
                                    else:
                                        fp+=1
                                else:
                                    fp+=1
                            else:
                                fp+=1
                for domain in ua:
                    for intent in ua[domain]:
                        for slot in ua[domain][intent]:
                            if domain in ua_gen:
                                if intent in ua_gen[domain]:
                                    if slot not in ua_gen[domain][intent]:
                                        fn+=1
                                else:
                                    fn+=1
                            else:
                                fn+=1
                '''
                precious=tp/(tp+fp+eps)
                recall=tp/(tp+fn+eps)
                f1=2*precious*recall/(precious+recall+eps)
                total_f1+=f1
                total_p+=precious
                total_r+=recall
                '''
        precious=tp/(tp+fp+eps)
        recall=tp/(tp+fn+eps)
        f1=2*precious*recall/(precious+recall+eps)
        return bleu, precious, recall, f1
        '''
        avg_f1=total_f1/total_turn
        avg_p=total_p/total_turn
        avg_r=total_r/total_turn
        return bleu, avg_p, avg_r, avg_f1
        '''

    def calculate_metrics(self, gen, oracle, modular='dst'):
        eps=1e-10
        total_tp=0
        total_fp=0
        total_fn=0
        joint_acc=0
        if modular=='dst':
            for (gen_bspn, gt_bspn) in zip(gen, oracle):
                gen_cons=self.reader.bspan_to_constraint_dict(gen_bspn)
                gt_cons=self.reader.bspan_to_constraint_dict(gt_bspn)
                tp=0
                fp=0
                fn=0
                for domain in gen_cons:
                    if domain not in gt_cons:
                        fp+=len(gen_cons[domain])
                        continue
                    for slot in gen_cons[domain]:
                        if slot not in gt_cons[domain]:
                            fp+=1
                        elif gt_cons[domain][slot]!=gen_cons[domain][slot]:
                            fp+=1
                        else:
                            tp+=1
                for domain in gt_cons:
                    if domain not in gen_cons:
                        fn+=len(gt_cons[domain])
                        continue
                    for slot in gt_cons[domain]:
                        if slot not in gen_cons[domain]:
                            fn+=1
                total_tp+=tp
                total_fp+=fp
                total_fn+=fn
                if fp==0 and fn==0:
                    joint_acc+=1
            joint_acc/=len(gen)
            P=total_tp/(total_tp+total_fp+eps)
            R=total_tp/(total_tp+total_fn+eps)
            F1=2*P*R/(P+R+eps)
            return joint_acc, (P, R, F1)
        elif modular=='dm':
            for (gen_aspn, gt_aspn) in zip(gen, oracle):
                gen_act=self.reader.aspan_to_act_dict(gen_aspn)
                gt_act=self.reader.aspan_to_act_dict(gt_aspn)
                tp=0
                fp=0
                fn=0
                for domain in gen_act:
                    for intent, slots in gen_act[domain].items():
                        if domain not in gt_act or intent not in gt_act[domain]:
                            fp+=len(slots)
                            continue
                        for slot in slots:
                            if slot not in gt_act[domain][intent]:
                                fp+=1
                            else:
                                tp+=1
                for domain in gt_act:
                    for intent, slots in gt_act[domain].items():
                        if domain not in gen_act or intent not in gen_act[domain]:
                            fn+=len(slots)
                            continue
                        for slot in slots:
                            if slot not in gen_act[domain][intent]:
                                fn+=1
                total_tp+=tp
                total_fp+=fp
                total_fn+=fn
                if fp==0 and fn==0:
                    joint_acc+=1
            joint_acc/=len(gen)
            P=total_tp/(total_tp+total_fp+eps)
            R=total_tp/(total_tp+total_fn+eps)
            F1=2*P*R/(P+R+eps)
            return joint_acc, (P, R, F1)
        elif modular=='nlg':
            wrap_generated = [[sent.replace('<sos_r>', '').replace('<eos_r>', '').strip()] for sent in gen]
            wrap_truth = [[sent.replace('<sos_r>', '').replace('<eos_r>', '').strip()] for sent in oracle]
            sc = self.bleu_scorer.score(zip(wrap_generated, wrap_truth))
            return sc
    
def get_richness(input_data):
    avg_lens, msttr, count = 0, 0, 0
    unique_grams = [Counter() for _ in range(3)]
    all_tokens = []

    for utterance in input_data:
        tokens = ld.tokenize(utterance)
        all_tokens.extend(tokens)
        
        avg_lens  += len(tokens)
        count += 1
        
        unique_grams[0].update(tokens)           
        unique_grams[1].update([(a, b) for a, b in zip(tokens, tokens[1:])])          
        unique_grams[2].update([(a, b, c) for a, b, c in zip(tokens, tokens[1:], tokens[2:])])
            
    avg_lens  /= count
    msttr = ld.msttr(all_tokens, window_length=50)      
    unique_grams_count = [len(c) for c in unique_grams]

    total = sum(v for v in unique_grams[0].values())
    probs = [(u/total) for u in unique_grams[0].values()]
    entropy = -sum(p * math.log(p, 2) for p in probs)
        
    cond = [unique_grams[1][(h, w)]/unique_grams[0][h] for h, w in unique_grams[1]]
    join = [unique_grams[1][(h, w)]/total for h, w in unique_grams[1]]
    cond_entropy = -sum(j * math.log(c, 2) for c, j in zip(cond, join))

    return {
        'entropy'         : entropy,
        'cond_entropy'    : cond_entropy,
        'avg_lengths'     : avg_lens,
        'msttr'           : msttr,
        'num_unigrams'    : unique_grams_count[0],
        'num_bigrams'     : unique_grams_count[1],
        'num_trigrams'    : unique_grams_count[2]
    }
                

if __name__ == '__main__':
    pass
