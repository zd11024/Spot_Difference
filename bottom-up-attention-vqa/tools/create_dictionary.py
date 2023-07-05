from __future__ import print_function
import os
import sys
import json
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import Dictionary
import pickle
import argparse

def create_dictionary(dataroot):
    dictionary = Dictionary()
    questions = []
    files = [
        'v2_OpenEnded_mscoco_train2014_questions.json',
        'v2_OpenEnded_mscoco_val2014_questions.json',
        'v2_OpenEnded_mscoco_test2015_questions.json',
        'v2_OpenEnded_mscoco_test-dev2015_questions.json'
    ]
    for path in files:
        question_path = os.path.join(dataroot, path)
        qs = json.load(open(question_path))['questions']
        for q in qs:
            dictionary.tokenize(q['question'], True)
    return dictionary

def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = [float(val) for val in vals[1:]]
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


def create_dictionary_spot_diff(dataroot):
    dictionary = Dictionary()
    files = [
        'spot_diff_train.json',
        'spot_diff_val.json',
        'spot_diff_test.json'
    ]
    for path in files:
        filepath = os.path.join(dataroot, path)
        with open(filepath) as f:
            dialogs = json.load(f)
        
        for dialog in dialogs:
            turn = len(dialog['questions'])
            for t in range(turn):
                ques = dialog['questions'][t]
                ans = dialog['answers'][t]
                dictionary.tokenize(ques, True)
                dictionary.tokenize(ans, True)

    return dictionary


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='../data/0206')
    args = parser.parse_args()
    d = create_dictionary_spot_diff(args.dataroot)
    d.dump_to_file(os.path.join(args.dataroot, 'dictionary.pkl'))

    d = Dictionary.load_from_file(os.path.join(args.dataroot, 'dictionary.pkl'))
    emb_dim = 300
    glove_file = 'data/glove/glove.6B.%dd.txt' % emb_dim
    weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)
    np.save(os.path.join(args.dataroot, 'glove6b_init_%dd.npy' %emb_dim), weights)