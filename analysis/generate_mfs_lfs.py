import random
import numpy as np
import os
import re
from collections import defaultdict, OrderedDict
from nltk.corpus import wordnet as wn

def load_data(datapath, name):
    text_path = os.path.join(datapath, '{}.data.xml'.format(name))
    gold_path = os.path.join(datapath, '{}.gold.key.txt'.format(name))

    #load gold labels 
    gold_labels = {}
    with open(gold_path, 'r', encoding="utf8") as f:
        for line in f:
            line = line.strip().split(' ')
            instance = line[0]
            #this means we are ignoring other senses if labeled with more than one 
            #(happens at least in SemCor data)
            key = line[1]
            gold_labels[instance] = key

    #load train examples + annotate sense instances with gold labels
    sentences = []
    s = []
    with open(text_path, 'r', encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line == '</sentence>':
                sentences.append(s)
                s=[]
                
            elif line.startswith('<instance') or line.startswith('<wf'):
                word = re.search('>(.+?)<', line).group(1)
                lemma = re.search('lemma="(.+?)"', line).group(1) 
                pos =  re.search('pos="(.+?)"', line).group(1)

                #clean up data
                word = re.sub('&apos;', '\'', word)
                lemma = re.sub('&apos;', '\'', lemma)

                sense_inst = -1
                sense_label = -1
                if line.startswith('<instance'):
                    sense_inst = re.search('instance id="(.+?)"', line).group(1)
                    #annotate sense instance with gold label
                    sense_label = gold_labels[sense_inst]
                s.append((word, lemma, pos, sense_inst, sense_label))

    return sentences



def mfs_senses(text_data, mfs_out, lfs_out):
    sense_dict = defaultdict(list)

    for sent in text_data:
        for form, lemma, pos, inst, sense in sent:
            if sense == -1:
                continue
    
            elif lemma not in sense_dict.keys():
                sense_dict[lemma] = defaultdict(dict)
                sense_dict[lemma][sense] = 1
            elif sense not in sense_dict[lemma].keys():
                sense_dict[lemma][sense] = 1
            else:
                sense_dict[lemma][sense] += 1

    fw_mfs = open(mfs_out, 'w')
    fw_lfs = open(lfs_out, 'w')
    for lemma in sense_dict.keys():
        vd = sorted(sense_dict[lemma].items(), key=lambda t:t[1], reverse=True)
        for i in range(0, len(vd)):
            if vd[i][1] == vd[0][1]:
                fw_mfs.write('{}\t{}\t{}\n'.format(lemma, vd[i][0], vd[i][1]))
            else:
                fw_lfs.write('{}\t{}\t{}\n'.format(lemma, vd[i][0], vd[i][1]))
    fw_mfs.close()
    fw_lfs.close()


if __name__ == "__main__":

    train_path = '../data/WSD_Evaluation_Framework/Training_Corpora/SemCor/'
    train_data = load_data(train_path, 'semcor')

    mfs_senses(train_data, 'mfs.txt', 'lfs.txt')
