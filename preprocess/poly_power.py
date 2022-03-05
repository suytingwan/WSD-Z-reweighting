import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json

pos_converter = {'NOUN':'n', 'PROPN':'n', 'VERB':'v', 'AUX':'v', 'ADJ':'a', 'ADV':'r'}

def generate_key(lemma, pos):
    if pos in pos_converter.keys():
        pos = pos_converter[pos]
    key = '{}+{}'.format(lemma, pos)
    return key


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


def read_numlist(filein, wn_senses, fileout_json, fileout_txt):
    '''
    calculate polysemy distribution on word level, and training instances for senses
    '''
    fread = open(filein)
    fread.readline()

    sense_count_dict = defaultdict(list)
    synset_count_dict = defaultdict(int)

    for line in fread:
        info = line.strip().split('\t')
        lemma, pos, sensekey = info[4], info[5], info[6]

        target_key = generate_key(lemma, pos)
        senses = wn_senses[target_key]

        synset_count_dict[target_key] += 1
        senseind = senses.index(sensekey)
        if target_key not in sense_count_dict.keys():
            sense_count_dict[target_key] = [0] * len(senses)
        sense_count_dict[target_key][senseind] += 1

    fread.close()
    json.dump(sense_count_dict, open(fileout_json, 'w'))

    sort_synset = sorted(synset_count_dict.items(), key=lambda x: x[1], reverse=True)

    fw = open(fileout_txt, 'w')
    for j, (key, count) in enumerate(sort_synset):
        new_line = str(count) + '\t' + key + '\t'
        for i in range(len(sense_count_dict[key])):
            new_line += str(sense_count_dict[key][i]) + '_'
        fw.write(new_line + '\n')

    fw.close()


def get_poly_dist(filein, K=300):
    '''
    smooth the polysemy distribution by group number K
    K = 50, 100, 200, 300, 400
    '''

    fread = open(filein)

    training_case_num = []
    mfs_case_num = []
    lfs_case_num = []
    polysemy_num = []

    tmp_case = []
    tmp_mfs_case = []
    tmp_lfs_case = []
    tmp_polysemy = []

    mfs_sense_count = 0

    for i, line in enumerate(fread):
        if i % K == 0 and i !=0:
            training_case_num.append(sum(tmp_case))
            mfs_case_num.append(sum(tmp_mfs_case))
            lfs_case_num.append(sum(tmp_lfs_case))
            polysemy_num.append(sum(tmp_polysemy)/(K+0.0))

            tmp_case = []
            tmp_mfs_case = []
            tmp_lfs_case = []
            tmp_polysemy = []

        info = line.strip().split('\t')
        tmp_case.append(int(info[0]))
        tmp_mfs_case.append(int(info[2].split('_')[0]))
        if int(info[2].split('_')[0]) != 0:
            mfs_sense_count += 1

        tmp_lfs_case.append(int(info[0]) - int(info[2].split('_')[0]))
        tmp_polysemy.append(len(info[2].split('_')) - 1)

    np.save(open('./semcor_polysemy_K_{}.npy'.format(K), 'wb'), np.array(polysemy_num))

    fread.close()

if __name__ == "__main__":

    wn_senses = load_wn_senses('../data/WSD_Evaluation_Framework/Data_Validation/candidatesWN30.txt')
    read_numlist('./semcor.csv', wn_senses, './semcor_sense_count.json', './semcor_synset_count.txt')

    get_poly_dist('./semcor_synset_count.txt', K=300)
