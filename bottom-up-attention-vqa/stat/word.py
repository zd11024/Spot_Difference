import json
import torch

def main():
    with open('../data/VQA/v2_OpenEnded_mscoco_train2014_questions.json') as f:
        questions = json.load(f)['questions']
    
    from collections import Counter

    pre_word = []
    suf_word = []
    ng2 = []
    ques_len = []

    for entry in questions:
        q = entry['question']
        q = q.lower()
        q = q.replace('?', '')
        tokens = q.split(' ')

        pre_word += [tokens[0]]
        suf_word += [tokens[-1]]
        for i in range(len(tokens)-1):
            ng2 += [(tokens[i], tokens[i+1])]
        
        ques_len += [len(tokens)]

    pre_word_cnt = Counter(pre_word)
    suf_word_cnt = Counter(suf_word)
    ng2_cnt = Counter(ng2)

    k = 20

    for w, f in pre_word_cnt.most_common(k):
        print(w, f)
    
    pre_word_freq = []
    for w, f in pre_word_cnt.most_common():
        pre_word_freq += [f]
    print('vocab:', len(pre_word_freq))
    print('mean:',torch.tensor(pre_word_freq).float().mean().item())

    print('====================================')

    for w, f in suf_word_cnt.most_common(k):
        print(w, f)
    
    suf_word_freq = []
    for w, f in suf_word_cnt.most_common():
        suf_word_freq += [f]
    print('vocab:', len(suf_word_freq))
    print('mean:', torch.tensor(suf_word_freq).float().mean().item())

    print(max(ques_len))
    print(torch.tensor(ques_len).float().mean().item())


if __name__=='__main__':
    main()