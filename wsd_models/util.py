import os
import re
import torch
import subprocess
from transformers import *
import random

pos_converter = {'NOUN':'n', 'PROPN':'n', 'VERB':'v', 'AUX':'v', 'ADJ':'a', 'ADV':'r'}

def generate_key(lemma, pos):
	if pos in pos_converter.keys():
		pos = pos_converter[pos]
	key = '{}+{}'.format(lemma, pos)
	return key

def load_pretrained_model(name):
    if name == 'roberta-base':
        model = RobertaModel.from_pretrained('roberta-base')
        hdim = 768
    elif name == 'roberta-large':
        model = RobertaModel.from_pretrained('roberta-large')
        hdim = 1024
    elif name == 'bert-large':
        model = BertModel.from_pretrained('bert-large-uncased')
        hdim = 1024
    elif name == 'bert-base': #bert base
        model = BertModel.from_pretrained('bert-base-uncased')
        hdim = 768

    return model, hdim

def load_tokenizer(name):
	if name == 'roberta-base':
		tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
	elif name == 'roberta-large':
		tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
	elif name == 'bert-large':
		tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
	elif name == 'bert-base': #bert base
		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	return tokenizer

def load_wn_senses(path):
	wn_senses = {}
	with open(path, 'r', encoding="utf8") as f:
		for line in f:
			line = line.strip().split('\t')
			lemma = line[0]
			pos = line[1]
			senses = line[2:]

			key = generate_key(lemma, pos)
			wn_senses[key] = senses
	return wn_senses

def get_label_space(data):
	#get set of labels from dataset
	labels = set()
	
	for sent in data:
		for _, _, _, _, label in sent:
			if label != -1:
				labels.add(label)

	labels = list(labels)
	labels.sort()
	labels.append('n/a')

	label_map = {}
	for sent in data:
		for _, lemma, pos, _, label in sent:
			if label != -1:
				key = generate_key(lemma, pos)
				label_idx = labels.index(label)
				if key not in label_map: label_map[key] = set()
				label_map[key].add(label_idx)

	return labels, label_map

def process_encoder_outputs(output, mask, as_tensor=False):
	combined_outputs = []
	position = -1
	avg_arr = []
	for idx, rep in zip(mask, torch.split(output, 1, dim=0)):
		#ignore unlabeled words
		if idx == -1: continue
		#average representations for units in same example
		elif position < idx: 
			position=idx
			if len(avg_arr) > 0: combined_outputs.append(torch.mean(torch.stack(avg_arr, dim=-1), dim=-1))
			avg_arr = [rep]
		else:
			assert position == idx 
			avg_arr.append(rep)
	#get last example from avg_arr
	if len(avg_arr) > 0: combined_outputs.append(torch.mean(torch.stack(avg_arr, dim=-1), dim=-1))
	if as_tensor: return torch.cat(combined_outputs, dim=0)
	else: return combined_outputs

#run WSD Evaluation Framework scorer within python
def evaluate_output(scorer_path, gold_filepath, out_filepath):
	eval_cmd = ['java','-cp', scorer_path, 'Scorer', gold_filepath, out_filepath]
	print(eval_cmd)
	#output = subprocess.Popen(eval_cmd, stdout=subprocess.PIPE ).communicate()[0]
	output = subprocess.check_output(eval_cmd)
	output = [x.decode("utf-8") for x in output.splitlines()]
	p,r,f1 =  [float(output[i].split('=')[-1].strip()[:-1]) for i in range(3)]
	return p, r, f1

#normalize ids list, masks to whatever the passed in length is
def normalize_length(ids, attn_mask, o_mask, max_len, pad_id):
	if max_len == -1:
		return ids, attn_mask, o_mask
	else:
		if len(ids) < max_len:
			while len(ids) < max_len:
				ids.append(torch.tensor([[pad_id]]))
				attn_mask.append(0)
				o_mask.append(-1)
		else:
			ids = ids[:max_len-1]+[ids[-1]]
			attn_mask = attn_mask[:max_len]
			o_mask = o_mask[:max_len]

		assert len(ids) == max_len
		assert len(attn_mask) == max_len
		assert len(o_mask) == max_len

		return ids, attn_mask, o_mask

