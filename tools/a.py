import os
import json
import copy
import argparse
from tkinter import dialog
import matplotlib.pyplot as plt
simulator_data_path=os.path.join(os.path.abspath(os.getcwd()), 'data', 'simulator')

with open(os.path.join(simulator_data_path, 'asset_graph.json')) as f:
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
        return 'convert'

    i = g['nodes'].index(node1)
    j = g['nodes'].index(node2)
    if g['floyd'][j][i]==1:
        return 'progressive'
    return 'convert'

def vis_tran(args):
    from simulator import QuestionParser
    q_parser = QuestionParser()
    
    from simulator import AnswerSimulator
    answer_simulator = AnswerSimulator()


    from collections import defaultdict
    tran = defaultdict(list)
    tran2 = defaultdict(list)

    with open(args.input_file) as f:
        dialogs = json.load(f)

    for i, d in enumerate(dialogs):
        nodes = []
        for q in d['questions']:
            ques_item = q_parser.parse_question(q)
            cur_node = None
            if ques_item and ques_item['qtype'] in ['count-hint', 'count-nohint']:
                ans_item = answer_simulator.answer(str(d['img1']), ques_item)
                if ans_item['cnt']>1:
                    cur_node = ques_item['ref']
            nodes.append(cur_node)
        
        for i in range(4):
            if nodes[i] is None or nodes[i+1] is None: 
                continue
            if get_tran_type(nodes[i], nodes[i+1])=='progressive':
                tran[(nodes[i], nodes[i+1])].append( d['target']==d['prediction'])
            else:                   
                tran2[(nodes[i], nodes[i+1])].append( (d['target']==d['prediction']) )
        
    L = [(k[0], k[1], len(v), sum(v)/len(v)) for k, v in tran.items()]
    L2 = sorted(L, key=lambda x: x[2], reverse=True)
    for x in L2:
        print(x[0], x[1], x[2], x[3])
    print('====================================')
    L = [(k[0], k[1], len(v), sum(v)/len(v)) for k, v in tran2.items()]
    L2 = sorted(L, key=lambda x: x[2], reverse=True)
    for x in L2:
        print(x[0], x[1], x[2], x[3])

    progressive_total = sum([len(v) for k, v in tran.items()])
    progressive_correct = sum([sum(v) for k, v in tran.items()])
    progressive_accu = progressive_correct / progressive_total
    print('progressive:', progressive_total, progressive_accu)

    convert_total = sum([len(v) for k, v in tran2.items()])
    convert_correct = sum([sum(v) for k, v in tran2.items()])
    convert_accu = convert_correct / convert_total
    print('convert:', convert_total, convert_accu)

def vis_acc(args):
    from simulator import QuestionParser
    q_parser = QuestionParser()

    from simulator import AnswerSimulator
    answer_simulator = AnswerSimulator()

    from collections import defaultdict
    cate_ques_acc = defaultdict(list)

    with open(args.input_file) as f:
        dialogs = json.load(f)
    

    for i, d in enumerate(dialogs):
        cate_ques_total = 0
        cate_ques_correct = 0
        
        for q in d['questions']:
            ques_item = q_parser.parse_question(q)
            if ques_item and ques_item['qtype'] in ['count-hint']:
                ans_item = answer_simulator.answer(str(d['img1']), ques_item)
                if ans_item and ans_item['cnt']>1:
                    cate_ques_total += 1
                    if ans_item['cnt']==ques_item['q_ans_cnt']:
                        cate_ques_correct += 1

        if cate_ques_total > 0:
            acc = cate_ques_correct / cate_ques_total
            cate_ques_acc[acc].append(d['target']==d['prediction'])


    for k, v in cate_ques_acc.items():
        print(k, sum(v)/len(v), len(v))

    X, Y = [], []
    for k, v in sorted(cate_ques_acc.items(), key=lambda x: x[0]):
        if len(v)<100: continue
        X.append(k)
        Y.append(sum(v)/len(v))
        print(k, sum(v)/len(v))
    plt.plot(X, Y)
    plt.xlabel('accu')
    plt.ylabel('succ')
    plt.show()


def vis_recall(args):
    from simulator import QuestionParser
    q_parser = QuestionParser()

    from simulator import AnswerSimulator
    answer_simulator = AnswerSimulator()

    from collections import defaultdict
    cate_ques_recall = defaultdict(list)

    with open(args.input_file) as f:
        dialogs = json.load(f)
    
    with open(args.spot_diff_file) as f:
        gt_dialogs = json.load(f)
        img2cateq = defaultdict(list)
        for d in gt_dialogs:
            imgid = int(d['img1'].split('/')[-1].split('.')[0][4:])
            for a in d['actions']:
                if a[0]['op']=='count':
                    img2cateq[imgid].append(a[0]['ref'])
        

    for i, d in enumerate(dialogs):
        
        cateq_list = []
        for q in d['questions']:
            ques_item = q_parser.parse_question(q)
            if ques_item and ques_item['qtype'] in ['count-hint', 'count-nohint']:
                ans_item = answer_simulator.answer(str(d['img1']), ques_item)
                if ans_item and ans_item['cnt']>1:
                    cateq_list.append(ques_item['ref'])
        
        if len(set(img2cateq[d['img1']]))>0:
            recall = len(set(cateq_list) & set(img2cateq[d['img1']])) / len(set(img2cateq[d['img1']]))
            cate_ques_recall[recall].append(d['target']==d['prediction'])        

    for k, v in cate_ques_recall.items():
        print(k, sum(v)/len(v), len(v))
    
    print('=================================')

    X, Y = [], []
    for k, v in sorted(cate_ques_recall.items(), key=lambda x: x[0]):
        if len(v)<300: continue
        X.append(k)
        Y.append(sum(v)/len(v))
        print(k, sum(v)/len(v))
    plt.plot(X, Y)
    plt.xlabel('recall')
    plt.ylabel('succ')
    plt.show()

if __name__=='__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='tools/gen.json')
    parser.add_argument('--spot_diff_file', type=str, default='data/0206/spot_diff_test.json')
    args = parser.parse_args()

    # vis_tran(args)
    # vis_acc(args)
    vis_recall(args)
    