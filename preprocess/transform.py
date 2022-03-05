import os
import re
import subprocess
import random
from collections import defaultdict
import xml.etree.ElementTree as ET
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn


pos_converter = {'NOUN':'n', 'PROPN':'n', 'VERB':'v', 'AUX':'v', 'ADJ':'a', 'ADV':'r'}

def generate_key(lemma, pos):
    if pos in pos_converter.keys():
        pos = pos_converter[pos]
    key = '{}+{}'.format(lemma, pos)
    return key


def load_data(datapath, name):
    text_path = os.path.join(datapath, '{}.data.xml'.format(name))
    gold_path = os.path.join(datapath, '{}.gold.key.txt'.format(name))

    #load train examples + annotate sense instances with gold labels
    tree = ET.ElementTree(file=text_path)
    root = tree.getroot()

    sentences = []
    poss = []
    targets = []
    targets_index_start = []
    targets_index_end = []
    lemmas = []


    for doc in root:
        for sent in doc:
            sentence = []
            pos = []
            target = []
            target_index_start = []
            target_index_end = []
            lemma = []
            for token in sent:
                assert token.tag == 'wf' or token.tag == 'instance'
                if token.tag == 'wf':
                    for i in token.text.split(' '):
                        sentence.append(i)
                        pos.append(token.attrib['pos'])
                        target.append('X')
                        lemma.append(token.attrib['lemma'])
                if token.tag == 'instance':
                    target_start = len(sentence)
                    for i in token.text.split(' '):
                        sentence.append(i)
                        pos.append(token.attrib['pos'])
                        target.append(token.attrib['id'])
                        lemma.append(token.attrib['lemma'])
                    target_end = len(sentence)
                    assert ' '.join(sentence[target_start:target_end]) == token.text
                    target_index_start.append(target_start)
                    target_index_end.append(target_end)
            sentences.append(sentence)
            poss.append(pos)
            targets.append(target)
            targets_index_start.append(target_index_start)
            targets_index_end.append(target_index_end)
            lemmas.append(lemma)

    gold_keys = []
    with open(gold_path, 'r', encoding="utf8") as m:
        key = m.readline().strip().split()
        while key:
            gold_keys.append(key[1])
            key = m.readline().strip().split()

    #outfile = 'semcor.csv'
    outfile = name + '.csv'
    with open(outfile, 'w', encoding="utf8") as g:
        g.write('sentence\ttarget_index_start\ttarget_index_end\ttarget_id\ttarget_lemma\ttarget_pos\tsense_key\n')
        num = 0
        for i in range(len(sentences)):
            for j in range(len(targets_index_start[i])):
                sentence = ' '.join(sentences[i])
                target_start = targets_index_start[i][j]
                target_end = targets_index_end[i][j]
                target_id = targets[i][target_start]
                target_lemma = lemmas[i][target_start]
                target_pos = poss[i][target_start]
                sense_key = gold_keys[num]
                num += 1
                g.write('\t'.join((sentence, str(target_start), str(target_end), target_id, target_lemma, target_pos, sense_key)))
                g.write('\n')

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
            
# test
if __name__ == "__main__":
    #data_path = '../data/WSD_Evaluation_Framework/Training_Corpora/SemCor'
    #name = 'semcor'

    data_path = '../data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/'
    name = 'semeval2007'

    load_data(data_path, name)
