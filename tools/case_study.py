import argparse
import json

def solve(args):
    img2dialog = {}
    with open(args.gen_dialog_file) as f:
        dialogs = json.load(f)
    for d in dialogs:
        img = d['img1']
        img2dialog[img] = d
        if img==args.i:
            print(d['img1'], d['img2'])
            for t in range(len(d['questions'])):
                ques = d['questions'][t]
                ans = d['answers'][t]
                print(ques, ans)
            print(d['prediction'], d['target'])
            print('==================================')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spot_diff_file', type=str, default='../data/0206/spot_diff_test.json')
    parser.add_argument('--gen_dialog_file', type=str, default='gen.json')
    parser.add_argument('-i', type=int, default=0)
    args = parser.parse_args()

    solve(args)
