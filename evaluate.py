import torch
from torch.nn import functional as F
from nltk.corpus import wordnet as wn
import os
import sys
import time
import math
import copy
import argparse
from tqdm import tqdm
import pickle

import random
import numpy as np
from torch.utils.data import DataLoader

from dataset_random_group import *
from wsd_models.util import *
from wsd_models.models import BiEncoderModel


context_device = "cuda:0"
gloss_device = "cuda:1"

def _eval(eval_loader, model, gloss_dict, multigpu=False):
    model.eval()
    eval_preds = []
    for i, (context_ids, context_attn_mask, context_output_mask, example_keys, insts, _) in enumerate(eval_loader):
        with torch.no_grad():
            context_ids = context_ids.squeeze(dim=0)
            context_attn_mask = context_attn_mask.squeeze(dim=0)
            context_output_mask = context_output_mask.squeeze(dim=0)
            if multigpu:
                context_ids = context_ids.to(context_device)
                context_attn_mask = context_attn_mask.to(context_device)
            else:
                context_ids = context_ids.cuda()
                context_attn_mask = context_attn_mask.cuda()

            context_output = model.context_forward(context_ids, context_attn_mask, context_output_mask)

            for output, key, inst in zip(context_output.split(1, dim=0), example_keys, insts):
                gloss_ids, gloss_attn_mask, sense_keys = gloss_dict[key]
                if multigpu:
                    gloss_ids = gloss_ids.to(gloss_device)
                    gloss_attn_mask = gloss_attn_mask.to(gloss_device)
                else:
                    gloss_ids = gloss_ids.cuda()
                    gloss_attn_mask = gloss_attn_mask.cuda()
                gloss_output = model.gloss_forward(gloss_ids, gloss_attn_mask)
                gloss_output = gloss_output.transpose(0,1)

                if multigpu:
                    output = output.cpu()
                    gloss_output = gloss_output.cpu()
                output = torch.mm(output, gloss_output)
                pred_idx = output.topk(1, dim=-1)[1].squeeze().item()
                pred_label = sense_keys[pred_idx]
                eval_preds.append((inst, pred_label))

    return eval_preds


def evaluate_model(args):
    print('Evaluating WSD model on {}...'.format(args.split))
    
    model = BiEncoderModel(args.encoder_name, freeze_gloss=False, freeze_context=False)
    model_path = os.path.join(args.ckpt, 'best_model.ckpt')
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()

    tokenizer = load_tokenizer(args.encoder_name)

    eval_path = os.path.join(args.postprocess_data_path, '{}.csv'.format(args.split))
    eval_data, eval_keywords, eval_ordered_ids = preprocess_context(tokenizer, eval_path, max_len=-1)
    eval_dataset = EvalDataset(eval_data, eval_ordered_ids)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    wn_path = os.path.join(args.data_path, 'WSD_Evaluation_Framework/Data_Validation/candidatesWN30.txt')
    wn_senses = load_wn_senses(wn_path)
    gloss_dict = load_and_preprocess_glosses(eval_data, tokenizer, wn_senses, max_len=32)

    eval_preds = _eval(eval_loader, model, gloss_dict, multigpu=False)

    pred_filepath = os.path.join(args.ckpt, './{}_predictions.txt'.format(args.split))
    with open(pred_filepath, 'w') as f:
        for inst, prediction in eval_preds:
            f.write('{} {}\n'.format(inst, prediction))

    gold_filepath = os.path.join(args.data_path, 'WSD_Evaluation_Framework/Evaluation_Datasets/{}/{}.gold.key.txt'.format(args.split, args.split))
    scorer_path = os.path.join(args.data_path, 'WSD_Evaluation_Framework/Evaluation_Datasets')
    p, r, f1 = evaluate_output(scorer_path, gold_filepath, pred_filepath)
    print('f1 of BERT probe on {} test set = {}'.format(args.split, f1))

    return


