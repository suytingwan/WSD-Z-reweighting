
import torch
from torch.nn import functional as F
from nltk.corpus import wordnet as wn
import os
import sys
import time
import math
import copy
import random
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset

from wsd_models.util import *


def preprocess_context(tokenizer, text_data, bsz=1, max_len=-1):
    if max_len == -1: assert bsz==1

    data = defaultdict(dict)
    ordered_ids = []

    print('preprocessing data...')
    num_line = 1
    with open(text_data, 'r', encoding='utf8') as f:
        line = f.readline()
        while True:
            if num_line % 1000 == 0:
                print(num_line)
            num_line += 1
            line = f.readline()
            if not line:
                break
            line = line.strip().split('\t')
            
            sentence, target_index_start, target_index_end, target_id, lemma, target_pos, sense_key = line

            c_ids = [torch.tensor([[x]]) for x in tokenizer.encode(tokenizer.cls_token)]
            o_masks = [-1] * len(c_ids)
            
            out_of_bound = False
            for ind, word in enumerate(sentence.strip().split(' ')):
                word_ids = [torch.tensor([[x]]) for x in tokenizer.encode(word.lower())]
                c_ids.extend(word_ids)

                if ind >= int(target_index_start) and ind < int(target_index_end):
                    if max_len > 0 and (len(o_masks) > max_len - 3):
                        out_of_bound = True
                    o_masks.extend([int(target_index_start)]*len(word_ids))
                else:
                    o_masks.extend([-1]*len(word_ids))

            if out_of_bound:
                continue

            target_lemma = generate_key(lemma, target_pos)
            if target_lemma not in data.keys():
                data[target_lemma]['sense'] = []
                data[target_lemma]['sentence'] = []

            c_ids.extend([torch.tensor([[x]]) for x in tokenizer.encode(tokenizer.sep_token)])
            c_attn_mask = [1]*len(c_ids)
            o_masks.extend([-1]*3)

            c_ids, c_attn_masks, o_masks = normalize_length(c_ids, c_attn_mask, o_masks, max_len, pad_id=tokenizer.encode(tokenizer.pad_token)[0])

            data[target_lemma]['sense'].append(sense_key)
            data[target_lemma]['sentence'].append([torch.cat(c_ids,dim=-1), torch.tensor(c_attn_masks).unsqueeze(dim=0), torch.tensor(o_masks).unsqueeze(dim=0), \
                target_lemma, target_id, sense_key])
            ordered_ids.append(target_id)

    # batch data here
    
    keywords = list(data.keys())
    return data, keywords, ordered_ids


class SemDataset(Dataset):

    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.keywords = list(data.keys())
        self.keycounts = []
        for key in self.keywords:
            self.keycounts.append(max(1.0, len(data[key]['sense'])/(batch_size+0.0)))
        self.key_cum_sum = np.cumsum(self.keycounts)
        self.key_sum = np.sum(self.keycounts)

        #generate even sampling mapping
        self.seq_inds = {}
        random.shuffle(self.keywords)
        ind = 0
        for key in self.keywords:
            new_order = np.arange(len(self.data[key]['sense']))
            np.random.shuffle(new_order)
            for i in range(0, len(self.data[key]['sense']), batch_size):
                self.seq_inds[ind] = []
                for j in range(i, min(i+batch_size, len(self.data[key]['sense']))):
                    self.seq_inds[ind].append([key, new_order[j]])
                ind += 1
        self.length = ind

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        keywords_num = int(np.searchsorted(self.key_cum_sum, np.random.rand(1)*self.key_sum))
        senses = self.data[self.keywords[keywords_num]]['sense']
        sense_ids = np.random.choice(len(senses), min(self.batch_size, len(senses)), replace=False)

        bz_data = [self.data[self.keywords[keywords_num]]['sentence'][ind_] for ind_ in sense_ids]

        context_ids = torch.cat([x for x, _, _, _, _, _ in bz_data], dim=0)
        context_attn_mask = torch.cat([x for _, x, _, _, _, _ in bz_data], dim=0)
        context_output_mask = torch.cat([x for _, _, x, _, _, _ in bz_data], dim=0)
        example_keys = []
        instances = []
        labels = []
        for _, _, _, key, instance, label in bz_data:
            example_keys.append(key)
            instances.append(instance)
            labels.append(label)
        
        return context_ids, context_attn_mask, context_output_mask, example_keys, instances, labels

        
class EvalDataset(Dataset):

    def __init__(self, data, ordered_ids, batch_size=1):
        self.data = {}
        len_sen = 0
        for key in data.keys():
            for sen in data[key]['sentence']:
                context_ids, context_attn_mask, context_output_mask, example_keys, instances, labels = sen 
                self.data[instances] = sen
                len_sen += 1
        self.len_sen = len_sen
        self.ordered_ids = ordered_ids
        assert len(self.ordered_ids) == self.len_sen, 'length of instances does not match length of test sentence'

    def __len__(self):
        return self.len_sen

    def __getitem__(self, idx):
        return self.data[self.ordered_ids[idx]]


def tokenize_glosses(gloss_arr, tokenizer, max_len):
    glosses = []
    masks = []
    for gloss_text in gloss_arr:
        g_ids = [torch.tensor([[x]]) for x in tokenizer.encode(tokenizer.cls_token)+tokenizer.encode(gloss_text)+tokenizer.encode(tokenizer.sep_token)]
        g_attn_mask = [1] * len(g_ids)
        g_fake_mask = [-1] * len(g_ids)
        g_ids, g_attn_mask, _ = normalize_length(g_ids, g_attn_mask, g_fake_mask, max_len, pad_id=tokenizer.encode(tokenizer.pad_token)[0])
        g_ids = torch.cat(g_ids, dim=-1)
        g_attn_mask = torch.tensor(g_attn_mask)
        glosses.append(g_ids)
        masks.append(g_attn_mask)
    return glosses, masks

def load_and_preprocess_glosses(lemma_words, tokenizer, wn_senses, max_len=-1):
    sense_glosses = {}
    
    for target_key in lemma_words:
        if target_key not in sense_glosses:
            sensekey_arr = wn_senses[target_key]
            gloss_arr = [wn.lemma_from_key(s).synset().definition() for s in sensekey_arr]
            gloss_ids, gloss_masks = tokenize_glosses(gloss_arr, tokenizer, max_len)
            gloss_ids = torch.cat(gloss_ids, dim=0)
            gloss_masks = torch.stack(gloss_masks, dim=0)
            sense_glosses[target_key] = (gloss_ids, gloss_masks, sensekey_arr)

    return sense_glosses


if __name__ == "__main__":
    text_data = './preprocess/semeval2007.csv'
    tokenizer = load_tokenizer('bert-base')
    batched_data, keywords, ordered_ids = preprocess_context(tokenizer, text_data, bsz=4, max_len=128)
    wn_senses = load_wn_senses('/home/ysuay/codes/LMMS/external/wsd_eval/WSD_Evaluation_Framework/Data_Validation/candidatesWN30.txt')
    load_and_preprocess_glosses(keywords, tokenizer, wn_senses, max_len=32)
    SemDataset(batched_data, 4)

