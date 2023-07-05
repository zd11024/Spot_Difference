import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', default=0, type=int)
args = parser.parse_args()

with open('../data/0206/spot_diff_test.json') as f:
    dialogs = json.load(f)

img2dialog = {}
for d in dialogs:
    img = int(d['img1'].split('/')[-1].split('.')[0][4:])
    img2dialog[img] = d
    if img==args.i:
        x = d
        for t in range(len(x['questions'])):
            ques = x['questions'][t]
            ans = x['answers'][t]
            print(ques, ans)
        print('==========================')
        print(x)