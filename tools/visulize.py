import re
import copy
import json
import argparse
from typing import DefaultDict
import matplotlib.pyplot as plt

super_cate = ['home appliance', 'large household appliance', 'small household appliance', 'furniture', 'table', 'toy', 'animal toy', 'toy model', 'food', 'fruit', 'drink', 'baked food', 'meat product', 'sporting goods', 'ball', 'sports equipment', 'kitchenware', 'tableware', 'office supply', 'stationery', 'paper product', 'office equipment', 'computer', 'decorative', 'fashion item', 'fashion accessory', 'shoes', 'hat']
color = ['red', 'blue', 'gray', 'yellow', 'brown', 'black', 'white', 'green', 'pink', 'silver', 'golden', 'purple']
material = ['wooden', 'plastic', 'metal', 'ceramic', 'leather', 'cloth', 'paper', 'marble']


with open('../dialog_generation/asset_graph.json') as f:
    obj2graph = json.load(f)
    for o in obj2graph:
        g = obj2graph[o]
        floyd = copy.deepcopy(g['edges'])
        nodes = []
        for x in g['nodes']:
            nodes.append(x['ref'])
        n = len(g['nodes'])
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if floyd[i][k]==1 and floyd[k][j]==1:
                        floyd[i][j] = 1
        g['nodes'] = nodes
        g['floyd'] = floyd


def get_tran_type(node1, node2):
    g = None
    for o in obj2graph:
        if node2 in obj2graph[o]['nodes']:
            g = obj2graph[o]
            break
    if node1 not in g['nodes']:
        return 'transferred'

    i = g['nodes'].index(node1)
    j = g['nodes'].index(node2)
    if g['floyd'][j][i]==1:
        return 'progressive'
    return 'transferred'


def solve1(args):
    from cate_score import QuestionParser
    F = QuestionParser(spot_diff_file=args.spot_diff_file)

    with open(args.input_file) as f:
        dialogs = json.load(f)
    
    total = {}
    correct = {}
    
    from collections import defaultdict
    tran = defaultdict(list)
    tran2 = defaultdict(list)


    for i, d in enumerate(dialogs):
        cnt = 0
        nodes = []
        for q in d['questions']:
            x = F.parse_question(q)
            cur_node = None
            flg1, flg2 = False, False
            if x and x['type'] in ['count1', 'count2']:
                objs = F.get_objs_with_node(d['img1'], x['node'])
                objs = set([re.sub(r'[0-9]*', '', o) for o in objs])
                if len(objs)>1:
                    flg1 = True
                if flg1:
                    cnt += 1
                cur_node = x['node']
            nodes.append(cur_node)
        
        for i in range(4):
            if nodes[i] is None or nodes[i+1] is None:
                continue
            tran[(nodes[i], nodes[i+1])].append( (d['target']==d['prediction']) )
            tran2[get_tran_type(nodes[i], nodes[i+1])].append( (d['target']==d['prediction']) )

        total[cnt] = total.get(cnt, 0) + 1
        if d['target']==d['prediction']:
            correct[cnt] = correct.get(cnt, 0) + 1

    X, Y = [], []
    for k in sorted(total.keys()):
        print('[K: %d][Total: %d][Correct: %d][Acc: %.4f]' % (k, total[k], correct[k], correct[k] / total[k]))
        X.append(k)
        Y.append(correct[k]/total[k])

    plt.plot(X, Y)
    plt.xlabel('count')
    plt.ylabel('succ')
    plt.show()

    L = [(k[0], k[1], sum(v), sum(v)/len(v)) for k, v in tran.items()]
    L2 = sorted(L, key=lambda x: x[2], reverse=True)
    for x in  L2:
        print(x[0], x[1], x[2], x[3])
    
    for k, v in tran2.items():
        print(k, sum(v) / len(v), len(v))

# acc
def solve2(args):
    from cate_score import QuestionParser
    F = QuestionParser(spot_diff_file=args.spot_diff_file)

    with open(args.input_file) as f:
        dialogs = json.load(f)
    
    from collections import defaultdict
    cate_ques_acc = defaultdict(list)

    for i,d in enumerate(dialogs):
        cate_ques_total = 0
        cate_ques_correct = 0

        for q in d['questions']:
            x = F.parse_question(q)
            flg1 = False
            if x and (x['type'] in ['count1', 'count2']):
                objs = F.get_objs_with_node(d['img1'], x['node'])
                cnt_real = len(objs)
                objs = set([re.sub(r'[0-9]*', '', o) for o in objs])
                if len(objs)>1:
                    flg1 = True
                if flg1:
                    cate_ques_total += 1
                    if ('cnt' not in x) or (cnt_real==x['cnt']):
                        cate_ques_correct += 1                

        if cate_ques_total > 0:
            acc = cate_ques_correct/cate_ques_total
            cate_ques_acc[acc].append(d['target']==d['prediction'])
        
        # if cate_ques_total>=3 and cate_ques_correct==cate_ques_total and d['target']!=d['prediction']:
        #     for t in range(len(d['questions'])):
        #         ques = d['questions'][t]
        #         ans = d['answers'][t]
        #         print(ques, ans)

    for k, v in cate_ques_acc.items():
        print(k, sum(v)/len(v), len(v))

    X, Y = [], []
    for k, v in sorted(cate_ques_acc.items(), key=lambda x: x[0]):
        if len(v)<200: continue
        X.append(k)
        Y.append(sum(v)/len(v))
        print(k, sum(v)/len(v))
    plt.plot(X, Y)
    plt.xlabel('accu')
    plt.ylabel('succ')
    plt.show()

# f1
def solve3(args):
    from cate_score import QuestionParser
    F = QuestionParser(spot_diff_file='../data/1013/spot_diff_val.json')

    from collections import defaultdict
    img_to_cate = defaultdict(list)
    with open('../data/1013/spot_diff_val.json') as f:
        gt_dialogs = json.load(f)
    
    for d in gt_dialogs:
        imgid = int(d['img1'].split('/')[-1].split('.')[0][4:])
        for a in d['question_action']:
            if a=='null': continue
            img_to_cate[imgid].append(a)


    with open(args.input_file) as f:
        dialogs = json.load(f)
    
    cate_ques_acc = defaultdict(list)
    cate_ques_recall = defaultdict(list)
    cate_ques_f1 = defaultdict(list)

    for i,d in enumerate(dialogs):
        cate_ques_total = 0
        cate_ques_correct = 0
        actions = []

        for q in d['questions']:
            x = F.parse_question(q)
            flg1 = False
            if x and (x['type'] in ['count1', 'count2']):
                objs = F.get_objs_with_node(d['img1'], x['node'])
                cnt_real = len(objs)
                objs = set([re.sub(r'[0-9]*', '', o) for o in objs])
                if len(objs)>1:
                    flg1 = True
                if flg1:
                    cate_ques_total += 1
                    if ('cnt' not in x) or (cnt_real==x['cnt']):
                        cate_ques_correct += 1
                    actions.append(x['node'])

        if cate_ques_total>0:
            acc = cate_ques_correct / cate_ques_total
            recall = len(set(actions) & set(img_to_cate[d['img1']])) / len(set(img_to_cate[d['img1']]))
            if acc+recall==0:
                f1 = 0
            else:
                f1 = 2 * acc * recall / (acc + recall)
        else:
            f1 = 0            
        cate_ques_f1[f1].append(d['target']==d['prediction'])

    for k, v in cate_ques_f1.items():
        print(k, sum(v)/len(v), len(v))

    X, Y = [], []
    for k, v in sorted(cate_ques_f1.items(), key=lambda x: x[0]):
        if len(v)<500: continue
        X.append(k)
        Y.append(sum(v)/len(v))
    plt.plot(X, Y)
    plt.xlabel('f1')
    plt.ylabel('succ')
    plt.show()

# recall
def solve4(args):
    from cate_score import QuestionParser
    F = QuestionParser(spot_diff_file=args.spot_diff_file)

    from collections import defaultdict
    img_to_cate = defaultdict(list)
    with open(args.spot_diff_file) as f:
        gt_dialogs = json.load(f)
    
    for d in gt_dialogs:
        imgid = int(d['img1'].split('/')[-1].split('.')[0][4:])
        for a in d['question_action']:
            if a=='null': continue
            img_to_cate[imgid].append(a)


    with open(args.input_file) as f:
        dialogs = json.load(f)
    
    cate_ques_acc = defaultdict(list)
    cate_ques_recall = defaultdict(list)
    cate_ques_f1 = defaultdict(list)

    for i,d in enumerate(dialogs):
        cate_ques_total = 0
        cate_ques_correct = 0
        actions = []

        for q in d['questions']:
            x = F.parse_question(q)
            flg1 = False
            if x and (x['type'] in ['count1', 'count2']):
                objs = F.get_objs_with_node(d['img1'], x['node'])
                cnt_real = len(objs)
                objs = set([re.sub(r'[0-9]*', '', o) for o in objs])
                if len(objs)>1:
                    flg1 = True
                if flg1:
                    cate_ques_total += 1
                    if ('cnt' not in x) or (cnt_real==x['cnt']):
                        cate_ques_correct += 1
                    actions.append(x['node'])

        if cate_ques_total>0:
            acc = cate_ques_correct / cate_ques_total
            recall = len(set(actions) & set(img_to_cate[d['img1']])) / len(set(img_to_cate[d['img1']]))
            if acc+recall==0:
                f1 = 0
            else:
                f1 = 2 * acc * recall / (acc + recall)
        else:
            f1 = 0

        if len(set(img_to_cate[d['img1']]))>0:
            recall = len(set(actions) & set(img_to_cate[d['img1']])) / len(set(img_to_cate[d['img1']]))
            cate_ques_recall[recall].append(d['target']==d['prediction'])         

    for k, v in cate_ques_f1.items():
        print(k, sum(v)/len(v), len(v))

    X, Y = [], []
    for k, v in sorted(cate_ques_recall.items(), key=lambda x: x[0]):
        if len(v)<500: continue
        X.append(k)
        Y.append(sum(v)/len(v))
        print(k, sum(v)/len(v))
    plt.plot(X, Y)
    plt.xlabel('recall')
    plt.ylabel('succ')
    plt.show()


# count && acc
def solve5(args):
    from cate_score import QuestionParser
    F = QuestionParser(spot_diff_file='../data/1013/spot_diff_val.json')

    with open(args.input_file) as f:
        dialogs = json.load(f)
    
    from collections import defaultdict
    cate_ques_acc = defaultdict(list)

    for i,d in enumerate(dialogs):
        cate_ques_total = 0
        cate_ques_correct = 0

        for q in d['questions']:
            x = F.parse_question(q)
            flg1 = False
            if x and (x['type'] in ['count1', 'count2']):
                objs = F.get_objs_with_node(d['img1'], x['node'])
                cnt_real = len(objs)
                objs = set([re.sub(r'[0-9]*', '', o) for o in objs])
                if len(objs)>1:
                    flg1 = True
                if flg1:
                    cate_ques_total += 1
                    if ('cnt' not in x) or (cnt_real==x['cnt']):
                        cate_ques_correct += 1                

        for t in range(len(d['questions'])):
            cate_ques_acc[(cate_ques_total, t+1)].append(d['guess_every_round'][t]==d['target'])

    X = [i for i in range(1, 6)]
    for k in range(6):
        Y = []
        for r in range(1, 6):
            acc = sum(cate_ques_acc[(k, r)]) / len(cate_ques_acc[(k, r)])
            print(k, r, len(cate_ques_acc[k, r]))
            Y.append(acc)
        plt.plot(X, Y, label=str(k))
    
    plt.legend()
    plt.ylabel('succ')
    plt.xlabel('round')
    plt.show()

# count--acc
def solve6(args):
    from cate_score import QuestionParser
    F = QuestionParser(spot_diff_file='../data/1013/spot_diff_val.json')

    with open(args.input_file) as f:
        dialogs = json.load(f)
    
    from collections import defaultdict
    cate_ques_acc = defaultdict(list)

    for i,d in enumerate(dialogs):
        cate_ques_total = 0
        cate_ques_correct = 0

        for q in d['questions']:
            x = F.parse_question(q)
            flg1 = False
            if x and (x['type'] in ['count1', 'count2']):
                objs = F.get_objs_with_node(d['img1'], x['node'])
                cnt_real = len(objs)
                objs = set([re.sub(r'[0-9]*', '', o) for o in objs])
                if len(objs)>1:
                    flg1 = True
                if flg1:
                    cate_ques_total += 1
                    if ('cnt' not in x) or (cnt_real==x['cnt']):
                        cate_ques_correct += 1                

        for t in range(len(d['questions'])):
            cate_ques_acc[cate_ques_total].append(d['prediction']==d['target'])

    X, Y = [], []
    for k, v in sorted(cate_ques_acc.items(), key=lambda x: x[0]):
        if len(v)<200: continue
        X.append(k)
        Y.append(sum(v)/len(v))
    plt.plot(X, Y)
    plt.xlabel('count')
    plt.ylabel('succ')
    plt.show()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='vqg_gen_test.json', type=str)
    parser.add_argument('--spot_diff_file', default='../data/1013/spot_diff_test.json', type=str)
    args = parser.parse_args()
    solve1(args)
    # solve2(args)
    # solve3(args)
    # solve4(args)
    # solve5(args)
    # solve6(args)