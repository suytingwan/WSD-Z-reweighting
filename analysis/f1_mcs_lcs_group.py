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


def split_mcs_lcs(gold_filepath, mcs_gold_out, lcs_gold_out, zero_gold_out, mfs, lfs, keys):

    fw1 = open(mcs_gold_out, 'w')
    fw2 = open(lcs_gold_out, 'w')
    fw3 = open(zero_gold_out, 'w')

    for line in open(gold_filepath):
        info = line.strip().split(' ')

        if info[0] not in keys:
            continue

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

def split(pred_filepath, gold_filepath, mcs_pred_out, lcs_pred_out, zero_pred_out, mfs, lfs, keys):

    fw1 = open(mcs_pred_out, 'w')
    fw2 = open(lcs_pred_out, 'w')
    fw3 = open(zero_pred_out, 'w')

    gold_sense = {}
    for line in open(gold_filepath):
        info = line.strip().split(' ')
        if info[0] not in keys:
            continue
        gold_sense[info[0]] = info[1:]


    for line in open(pred_filepath):
        info = line.strip().split(' ')
        if info[0] not in keys:
            continue
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

    lemma_words_file = '/home/ysuay/codes/WSD-margin/preprocess/semcor_count.txt'

    groups = [300, 600, 900, 1800, 3900, 22436]
    dicts_synset = {}
    for j in range(len(groups)+1):
        dicts_synset[j] = []
    for i, line in enumerate(open(lemma_words_file)):
        info = line.strip().split('\t')
        if i < 300:
            dicts_synset[0].append(info[1])
        elif i < 600:
            dicts_synset[1].append(info[1])
        elif i < 900:
            dicts_synset[2].append(info[1])
        elif i < 1800:
            dicts_synset[3].append(info[1])
        elif i < 3900:
            dicts_synset[4].append(info[1])
        else:
            dicts_synset[5].append(info[1])


    dicts_gold_synset = {}
    for j in range(len(groups)+1):
        dicts_gold_synset[j] = []

    for line in open(gold_filepath):
        info = line.strip().split(' ')
        lemma = wn.lemma_from_key(info[1])
        word = lemma.name().lower()
        postag = lemma.synset().pos()
        lemma_pos = word + '+' + postag
        find = False
        for i in range(len(groups)+1):
            if lemma_pos in set(dicts_synset[i]):
                dicts_gold_synset[i].append(info[0])
                find = True
                break
        if find == False:
            dicts_gold_synset[len(groups)].append(info[0])

    keys = dicts_gold_synset[1] 

    mfs, lfs = load_dict('mfs.txt', 'lfs.txt')

    for i in range(len(groups)+1):
        print('calculating group: ', i)
        keys = dicts_gold_synset[i]
        split_mcs_lcs(gold_filepath, 'gold_mcs_group.txt', 'gold_lcs_group.txt', 'gold_zero_group.txt', mfs, lfs, set(keys))
    
        pred_filepath = '../replicate2/ALL_predictions.txt'

        split(pred_filepath, gold_filepath, 'pred_mcs_group.txt', 'pred_lcs_group.txt', 'pred_zero_group.txt', mfs, lfs, set(keys))

        scorer_path = '../data/WSD_Evaluation_Framework/Evaluation_Datasets/' 

        gold_filepath_eval = ['./gold_mcs_group.txt', './gold_lcs_group.txt', './gold_zero_group.txt']
        pred_filepath_eval = ['./pred_mcs_group.txt', './pred_lcs_group.txt', './pred_zero_group.txt']

        for gold, pred in zip(gold_filepath_eval, pred_filepath_eval):
            p, r, f1 = eval_output(scorer_path, gold, pred)
            print(p, r, f1)
