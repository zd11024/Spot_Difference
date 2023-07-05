import os
import json
import pickle

def get_answer_dict(dialogs):
    ret = {}
    for dialog in dialogs:
        turn = len(dialog['questions'])
        for t in range(turn):
            ret[dialog['answers'][t]] = ret.get(dialog['answers'][t], 0) + 1
    return ret

def create_ans2label(occurence, name, cache_root='data/cache'):
    """Note that this will also create label2ans.pkl at the same time
    occurence: dict {answer -> whatever}
    name: prefix of the output file
    cache_root: str
    """

    ans2label = {}
    label2ans = []
    label = 0
    for answer in occurence:
        label2ans.append(answer)
        ans2label[answer] = label
        label += 1


    cache_file = os.path.join(cache_root, name+'_ans2label.pkl')
    pickle.dump(ans2label, open(cache_file, 'wb'))
    cache_file = os.path.join(cache_root, name+'_label2ans.pkl')
    pickle.dump(label2ans, open(cache_file, 'wb'))
    return ans2label


if __name__=='__main__':
    dataroot = '../data/0206/'
    train_file = os.path.join(dataroot, 'spot_diff_train.json')
    val_file = os.path.join(dataroot, 'spot_diff_val.json')
    cache_dir = os.path.join(dataroot, 'cache')

    with open(train_file) as f:
        train_data = json.load(f)

    with open(val_file) as f:
        val_data = json.load(f)
    
    train_ans_dict = get_answer_dict(train_data)
    val_ans_dict = get_answer_dict(val_data)

    ans_list = list(set(list(train_ans_dict.keys()) + list(val_ans_dict.keys())))
    ans2label = create_ans2label(ans_list, 'trainval', cache_root=cache_dir)
    
