import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from poly_power import generate_key, load_wn_senses
from collections import defaultdict
import json

def func(x, a, b, c):
    return a * np.log(x+c) + b

def get_poly_dist(filein):
    '''
    fit the smoothed polysemy distribution by power law function
    '''
    semcor_polysemy = np.load(open(filein, 'rb'))

    x = np.arange(len(semcor_polysemy))

    popt, pconv = curve_fit(func, x, semcor_polysemy)
    y_fit = func(x, *popt)
    weight = np.round(y_fit/max(y_fit), decimals=1)

    threads = []
    for i in range(10, 0, -1):
        vals = np.where(weight == i/10.0)
        if vals[0].shape[0] > 0:
            threads.append([vals[0][0], i/10.0])

    return threads

def find_group(threads, ind, K, gamma):

    for i in range(len(threads) - 1):
        if ind >= threads[i][0] * K and ind < threads[i+1][0] * K:
            return np.round(np.power(threads[i][1], gamma)+1e-4, decimals=1)
        
    return np.round(np.power(threads[-1][1], gamma)+1e-4, decimals=1)
            
    
def assign_weight(filein, threads, fileout, K=100, gamma=1):
    '''
    assign weight to words according to fitting value
    gamma = 1, 2
    '''

    fread = open(filein)
    fread.readline()
    lemma_words = []

    synset_count_dict = defaultdict(int)
    synset_weight_dict = defaultdict(float)

    for line in fread:
        info = line.strip().split('\t')
        lemma, pos, sensekey = info[4], info[5], info[6]
        target_key = generate_key(lemma, pos)
        synset_count_dict[target_key] += 1

    fread.close()

    sort_synset = sorted(synset_count_dict.items(), key=lambda x: x[1], reverse=True)

    for j, (key, count) in enumerate(sort_synset):
        reg_weight = find_group(threads, j, K, gamma)
        if j % 100 == 0:
            print(reg_weight, j)
        if reg_weight < 0.1:
            reg_weight = 0.1
        synset_weight_dict[key] = reg_weight

    json.dump(synset_weight_dict, open(fileout, 'w'))


if __name__ == "__main__":

    threads = get_poly_dist('./semcor_polysemy_K_300.npy')
    K = 300
    gamma = 2
    assign_weight('./semcor.csv', threads, './semcor_synset_weight_{}_{}.json'.format(K, gamma), K, gamma)
