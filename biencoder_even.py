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
import json
from transformers import *

import random
import numpy as np
from torch.utils.data import DataLoader

from wsd_models.util import *
from wsd_models.models import BiEncoderModel
from dataset_even import *
from evaluate import _eval, evaluate_model

#uses these two gpus if training in multi-gpu

context_device = "cuda:0"
gloss_device = "cuda:1"

def _train(train_loader, model, gloss_dict, optim, schedule, criterion, max_grad_norm=1.0, \
           grad_step_size=1, multigpu=False, silent=False):
    model.train()
    total_loss = 0.

    start_time = time.time()

    train_data = enumerate(train_loader)
    if not silent: train_data = tqdm(list(train_data))

    context_grad_size = 0
    losses = []
    for i, (context_ids, context_attn_mask, context_output_mask, example_keys, _, labels) in train_data:

        model.zero_grad()
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

        #each batch has same sensekey, calculate gloss once here

        key = example_keys[0][0]
        gloss_ids, gloss_attn_mask, sense_keys = gloss_dict[key]
        if multigpu:
            gloss_ids = gloss_ids.to(gloss_device)
            gloss_attn_mask = gloss_attn_mask.to(gloss_device)
        else:
            gloss_ids = gloss_ids.cuda()
            gloss_attn_mask = gloss_attn_mask.cuda()

        gloss_output = model.gloss_forward(gloss_ids, gloss_attn_mask)
        gloss_output = gloss_output.transpose(0, 1)

        if multigpu:
            context_output = context_output.cpu()
            gloss_output = gloss_output.cpu()

        output = torch.mm(context_output, gloss_output)

        label_inds = []
        for j, label in enumerate(labels):
            idx = sense_keys.index(label[0])
            label_inds.append(idx)

        label_tensor = torch.tensor(label_inds)
        loss = criterion(output, label_tensor)

        context_grad_size += context_output.shape[0]
        losses.append(loss)
        if context_grad_size >= grad_step_size:
            batch_loss = 0.0
            for loss_ in losses:
                batch_loss += torch.sum(loss_)
            batch_loss = batch_loss / context_grad_size
            total_loss += batch_loss.item()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), max_grad_norm)
            optim.step()
            schedule.step()
            optim.zero_grad()
            context_grad_size = 0
            losses = []

    return model, optim, schedule, loss


def train_model(args):
    print('Training WSD bi-encoder model...')

    if not os.path.exists(args.ckpt): os.mkdir(args.ckpt)

    print('loading data and preprocessing data...')
    sys.stdout.flush()

    tokenizer = load_tokenizer(args.encoder_name)

    train_path = os.path.join(args.postprocess_data_path, 'semcor.csv')
    train_data, train_keywords, train_ordered_ids = preprocess_context(tokenizer, train_path, max_len=args.context_max_length)
    train_dataset = SemDataset(train_data, batch_size=4)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True) 

    semeval2007_path = os.path.join(args.postprocess_data_path, 'semeval2007.csv')
    semeval2007_data, eval_keywords, eval_ordered_ids = preprocess_context(tokenizer, semeval2007_path, max_len=-1)
    eval_dataset = EvalDataset(semeval2007_data, eval_ordered_ids)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
 
    wn_senses = load_wn_senses(os.path.join(args.data_path, 'WSD_Evaluation_Framework/Data_Validation/candidatesWN30.txt'))

    train_gloss_dict = load_and_preprocess_glosses(train_keywords, tokenizer, wn_senses, max_len=args.gloss_max_length)

    semeval2007_gloss_dict = load_and_preprocess_glosses(eval_keywords, tokenizer, wn_senses, max_len=args.gloss_max_length)

    model = BiEncoderModel(args.encoder_name, freeze_gloss=False, freeze_context=False, tie_encoders=False)
    if args.multigpu:
        model.gloss_encoder = model.gloss_encoder.to(gloss_device)
        model.context_encoder = model.context_encoder.to(context_device)
    else:
        model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss(size_average=False, reduction=None, reduce=False)

    weight_decay = 0.0
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    adam_epsilon = 1e-8
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=adam_epsilon)
    epochs = args.epochs
    t_total = 226037 * epochs
    schedule = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup, num_training_steps=t_total)


    best_dev_f1 = 0.
    print('Training probe...')
    sys.stdout.flush()

    for epoch in range(1, epochs+1):
        model, optimizer, schedule, train_loss = _train(train_loader, model, train_gloss_dict, optimizer, schedule, criterion, \
                                                            grad_step_size=args.gradient_step_size, max_grad_norm=args.grad_norm, silent=args.silent, multigpu=args.multigpu)
        eval_preds = _eval(eval_loader, model, semeval2007_gloss_dict, multigpu=args.multigpu)

        pred_filepath = os.path.join(args.ckpt, 'tmp_predictions.txt')
        score_filepath = os.path.join(args.ckpt, 'score.txt')
        with open(pred_filepath, 'w') as f:
            for inst, prediction in eval_preds:
                f.write('{} {}\n'.format(inst, prediction))

        gold_filepath = os.path.join(args.data_path, 'WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt')
        scorer_path = os.path.join(args.data_path, 'WSD_Evaluation_Framework/Evaluation_Datasets')
        _, _, dev_f1 = evaluate_output(scorer_path, gold_filepath, pred_filepath)
        print('Dev f1 after {} epochs = {}'.format(epoch, dev_f1))
        with open(score_filepath, 'a+') as fr:
            fr.write('Dev f1 after {} epochs = {}\n'.format(epoch, dev_f1))
        sys.stdout.flush() 

        if dev_f1 >= best_dev_f1:
            print('updating best model at epoch {}...'.format(epoch))
            sys.stdout.flush() 
            best_dev_f1 = dev_f1
            #save to file if best probe so far on dev set
            model_fname = os.path.join(args.ckpt, 'best_model.ckpt')
            with open(model_fname, 'wb') as f:
                torch.save(model.state_dict(), f)
            sys.stdout.flush()
    return


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Need available GPU(s) to run this model...")
        quit()

    parser = argparse.ArgumentParser(description='Gloss Informed Bi-encoder for WSD')

    #training arguments
    parser.add_argument('--rand_seed', type=int, default=42)
    parser.add_argument('--grad-norm', type=float, default=1.0)
    parser.add_argument('--silent', action='store_true',
	help='Flag to supress training progress bar for each epoch')
    parser.add_argument('--multigpu', action='store_true')
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--context-max-length', type=int, default=128)
    parser.add_argument('--gloss-max-length', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--gradient-step-size', type=int, default=36, 
        help='gradient update step according to context size')
    parser.add_argument('--mini-batch-size', type=int, default=4,
        help='batch size for context sentences') 
    parser.add_argument('--encoder-name', type=str, default='bert-base',
	choices=['bert-base', 'bert-large', 'roberta-base', 'roberta-large'])
    parser.add_argument('--ckpt', type=str, required=True,
	help='filepath at which to save best probing model (on dev set)')
    parser.add_argument('--data-path', type=str, required=True,
    	help='Location of top-level directory for the Unified WSD Framework')
    parser.add_argument('--postprocess-data-path', type=str, required=True,
        help='Location of training and evaluating data')

    #evaluation arguments
    parser.add_argument('--eval', action='store_true',
	help='Flag to set script to evaluate probe (rather than train)')
    parser.add_argument('--split', type=str, default='semeval2007',
	choices=['semeval2007', 'senseval2', 'senseval3', 'semeval2013', 'semeval2015', 'ALL'],
	help='Which evaluation split on which to evaluate probe')


    args = parser.parse_args()
    print(args)
    torch.manual_seed(args.rand_seed)
    os.environ['PYTHONHASHSEED'] = str(args.rand_seed)
    torch.cuda.manual_seed(args.rand_seed)
    torch.cuda.manual_seed_all(args.rand_seed)
    np.random.seed(args.rand_seed)
    random.seed(args.rand_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic=True


    if args.eval: evaluate_model(args)
    else: train_model(args)

