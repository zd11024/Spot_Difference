import os
import json
import argparse

def main(args):
    dialogs = []
    with open(os.path.join(args.dataroot, 'spot_diff_train.json')) as f:
        d = json.load(f)
        dialogs.extend(d)
    with open(os.path.join(args.dataroot, 'spot_diff_val.json')) as f:
        d = json.load(f)
        dialogs.extend(d)
    with open(os.path.join(args.dataroot, 'spot_diff_test.json')) as f:
        d = json.load(f)
        dialogs.extend(d)

    qtype_cnt = {}
    for d in dialogs:
        turn = len(d['questions'])
        for t in range(turn):
            qtype = d['questions_type'][t]
            qtype_cnt[qtype] = qtype_cnt.get(qtype, 0) + 1
    
    print(qtype_cnt)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='../data/1013')
    args = parser.parse_args()
    main(args)