import numpy as np
from collections import defaultdict, OrderedDict
from nltk.corpus import wordnet as wn
import subprocess

def load_dict(mfsfile, lfsfile):
    mfs_words = []
    for line in open(mfsfile):
        info = line.strip().split('\t')
        mfs_words.append(info[1])

    lfs_words = []
    for line in open(lfsfile):
        info = line.strip().split('\t')
        lfs_words.append(info[1])

    return set(mfs_words), set(lfs_words)


def wn_first_sense(key, postag=None):

    tag = wn.lemma_from_key(key).synset().pos()

    lemma = key.split('%')[0]
    first_synset = wn.synsets(lemma, pos=tag)[0]
    found = False
    for lem in first_synset.lemmas():
        first_key = lem.key()
        if first_key.startswith('{}%'.format(lemma)) and first_key == key:
            found = True
            break
    return found    


def eval_output(score_path, gold_filepath, out_filepath):

    eval_cmd = ['java', '-cp', scorer_path, 'Scorer', gold_filepath, out_filepath]
    print(eval_cmd)

    output = subprocess.check_output(eval_cmd)
    output = [x.decode('utf-8') for x in output.splitlines()]

    p, r, f1 = [float(output[i].split('=')[-1].strip()[:-1]) for i in range(3)]

    return p, r, f1


def split_mcs_lcs(gold_filepath, mcs_gold_out, lcs_gold_out, zero_gold_out, mfs, lfs):

    fw1 = open(mcs_gold_out, 'w')
    fw2 = open(lcs_gold_out, 'w')
    fw3 = open(zero_gold_out, 'w')

    for line in open(gold_filepath):
        info = line.strip().split(' ')

        if wn_first_sense(info[1]):
            fw1.write(line)

        else:
            fw2.write(line)

        unseen = True
        for i in range(len(info)-1):
            if (info[i+1] in mfs) or (info[i+1] in lfs):
                unseen = False
                break

        if unseen:
            fw3.write(line)


    fw1.close()
    fw2.close()
    fw3.close()

def split(pred_filepath, gold_filepath, mcs_pred_out, lcs_pred_out, zero_pred_out, mfs, lfs):

    fw1 = open(mcs_pred_out, 'w')
    fw2 = open(lcs_pred_out, 'w')
    fw3 = open(zero_pred_out, 'w')

    gold_sense = {}
    for line in open(gold_filepath):
        info = line.strip().split(' ')
        gold_sense[info[0]] = info[1:]

    for line in open(pred_filepath):
        info = line.strip().split(' ')
        if wn_first_sense(gold_sense[info[0]][0]):
            fw1.write(line)
        else:
            fw2.write(line)

        unseen = True
        for i in range(len(gold_sense[info[0]])):
            if (gold_sense[info[0]][i] in mfs) or (gold_sense[info[0]][i] in lfs):
                unseen = False
                break

        if unseen:
            fw3.write(line)

    fw1.close()
    fw2.close()
    fw3.close() 


if __name__ == "__main__":

    gold_filepath = '../data/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.gold.key.txt'

    mfs, lfs = load_dict('mfs.txt', 'lfs.txt')
    split_mcs_lcs(gold_filepath, 'gold_mcs.txt', 'gold_lcs.txt', 'gold_zero.txt', mfs, lfs)

    path = 'semcor_reweight_synset_V11'

    pred_filepath = '../ckpts_hyperparameters/{}/ALL_predictions.txt'.format(path)
    split(pred_filepath, gold_filepath, 'pred_mcs.txt', 'pred_lcs.txt', 'pred_zero.txt', mfs, lfs)

    scorer_path = '../data/WSD_Evaluation_Framework/Evaluation_Datasets/' 

    gold_filepath = ['./gold_mcs.txt', './gold_lcs.txt', './gold_zero.txt']
    pred_filepath = ['./pred_mcs.txt', './pred_lcs.txt', './pred_zero.txt']

    for gold, pred in zip(gold_filepath, pred_filepath):
        p, r, f1 = eval_output(scorer_path, gold, pred)
        print(p, r, f1)
