from logging import exception
import os
from platform import node
import re
import json
from copy import deepcopy
from collections import defaultdict
simulator_data_path=os.path.join(os.path.abspath(os.path.join(__file__, '../..')), 'data', 'simulator')


class QuestionParser:
    def __init__(self):
        self.templates = defaultdict(list)
        with open(os.path.join(simulator_data_path, 'question_template.json')) as f:
            templates = json.load(f)
            for k in templates:
                for v in templates[k]:
                    s = re.sub(r'[?]', '[?]', v)
                    s = s.replace('%(copula)s', '(is|are)')
                    s = re.sub(r'%[(].*?[)]s', '(.*?)', s)
                    keys = re.findall(r'(?<=%[(]).*?(?=[)]s)', v)
                    L = [s] + [x.strip() for x in keys]
                    self.templates[k].append(L)
        
        self.c_types = ['same', 'diff', 'more', 'less']
        self.q_types = list(set(self.templates.keys()) - set(self.c_types))
        self.eng2num = {'a': 1, 'an': 1, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}
        
        with open(os.path.join(simulator_data_path, 'ref_text.json')) as f:
            nodes = json.load(f)
            self.nodes = nodes
            self.ref2node = {}
            for x in nodes:
                if 'mapping' in nodes[x]:
                    continue
                for k in nodes[x]['singular']:
                    if nodes[x]['activation']==1 or k not in self.ref2node:
                        self.ref2node[k.lower()] = x
                for k in nodes[x]['plural']:
                    if nodes[x]['activation']==1 or k not in self.ref2node:
                        self.ref2node[k.lower()] = x

            # print(len(self.ref2node))

    def parse_question(self, ques):
        ques = deepcopy(ques)
        # remove conjunction
        for k in self.c_types:
            for l in self.templates[k]:
                ques = ques.replace(l[0], '')
                ques = ques.strip()
        ques = ques.replace(' - ', '-')

        # parse question
        ques_item = None
        for k in self.q_types:
            for l in self.templates[k]:
                results = re.search(l[0], ques)
                if results:
                    flg = True
                    try:
                        tmp = {'qtype': k}
                        for i in range(1, len(l)):
                            k = l[i]
                            v = results.group(i)
                            if 'ref' in k:
                                v = self.ref2node[v]
                            if k=='q_ans_cnt':
                                v = self.eng2num[v]
                            tmp[k] = v
                    except Exception as e:
                        # print(e)
                        flg = False
                    if flg:
                        ques_item = tmp
        
        return ques_item


class AnswerSimulator:
    def __init__(self):
        with open(os.path.join(simulator_data_path, 'ref_text.json')) as f:
            self.ref_text = json.load(f)
        with open(os.path.join(simulator_data_path, 'scene_graph.json')) as f:
            self.scene_graph = json.load(f)
        with open(os.path.join(simulator_data_path, 'asset_graph.json')) as f:
            self.cate2nodes = {}
            asset_graph = json.load(f)
            for k in asset_graph:
                L = []
                for x in asset_graph[k]['nodes']:
                    L += [x['ref']]
                self.cate2nodes[k] = L
        with open(os.path.join(simulator_data_path, 'annotation_definitions.json')) as f:
            self.id2object = {}
            d = json.load(f)
            for item in d['annotation_definitions'][0]['spec']:
                self.id2object[item['label_id']] = item['label_name']

        self.num2eng = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}
        self.colors = ['blue', 'brown', 'green', 'yellow', 'purple', 'gray', 'gold', 'black', 'pink', 'silver', 'white', 'red']
        self.materials = ['plastic', 'leather', 'glass', 'paper', 'wooden', 'ceramic', 'rubber', 'marble', 'cloth', 'metal']
        self.categories = ['desk lamp', 'milk', 'paperbox', 'bus model', 'plate', 'cola', 'dish decorative', 'palette', 'sandals', 'notebook', 'plug plate', 'airplane model', 'bowl', 'baseball cap', 'cloth tree', 'floor lamp', 'television', 'cabinet', 'pizza', 'dining table', 'pencil', 'keyboard', 'chair', 'fork', 'frame', 'rabbit chocolate', 'trash can', 'study table', 'hammer', 'bunny toy', 'desktop', 'dinosaur toy', 'baseball bat', 'headphone', 'sofa', 'apple', 'guitar', 'washing machine', 'bow', 'coffee machine', 'cotton cap', 'beer', 'basketball', 'dumbbel', 'boots', 'scissors', 'car model', 'spoon', 'soccer', 'tea table', 'phone', 'bowling pin', 'plant', 'toilet paper', 'skateboard', 'carpet', 'top hat', 'teddy bear', 'glasses', 'baby bed', 'giraffe toy', 'backpack', 'bed', 'watermelon', 'cup', 'cut board', 'chicken nugget', 'laptop', 'book', 'canvas shoes', 'fridge', 'archery target', 'tea', 'chicken leg', 'elephant toy', 'mouse', 'kettle', 'folder', 'bike model', 'bench', 'bread', 'cat toy', 'tennis', 'nightstand', 'vase', 'banana']

    def answer(self, imgid, ques, last_ques=None):
        g = self.scene_graph[imgid]
        ans_item = None
        # count questions
        if ques['qtype'] in ['count-hint', 'count-nohint']:
            cnt = 0
            for x in g['category']:
                c = self.id2object[x]
                if ques['ref'] in self.cate2nodes[c]:
                    cnt += 1
            ans_item = {'qtype': ques['qtype'], 'cnt': cnt}
        
        # ref questions
        if ques['qtype'] in ['refer-it', 'refer-them']:
            L = []
            for x in g['category']:
                c = self.id2object[x]
                if last_ques['ref'] in self.cate2nodes[c]:
                    L += [c]
            if len(L)==1:
                return {'qtype': ques['qtype'], 'ref': L[0]}
            else:
                ref_dict = {}
                for c in L:
                    x = re.sub(r'\d', '', c.replace('_', ' '))
                    ref_dict[x] = ref_dict.get(x, 0)+1
                return {'qtype': ques['qtype'], 'ref_dict': ref_dict}

        # query questions
        if ques['qtype'] in ['query-color', 'query-color-hint', 'query-material', 'query-material-hint']:
            L = []
            for x in g['category']:
                c = self.id2object[x]
                if ques['obj'] in self.cate2nodes[c]:
                    L += [c]
            if len(L)==0:
                ans_item = {'qtype': ques['qtype'], 'exception': 'no reference'}
            elif len(L)>1:
                ans_item = {'qtype': ques['qtype'], 'exception': 'multiple references'}
            else:
                keys = self.colors if ques['qtype']=='query-color' else self.materials
                for x in self.cate2nodes[L[0]]:
                    if x in keys:
                        ans_item = {'qtype': ques['qtype'], 'val': x}
                        break
        
        # extreme questions
        if ques['qtype'].startswith('extreme'):
            if ques['qtype'] in ['extreme-pic', 'extreme-pic-hint']:
                L = [d[ques['loc1']] for d in g['position']]
                if ques['loc1'] in ['left', 'front']:
                    ix = L.index(min(L))
                else:
                    ix = L.index(max(L))
            else:
                L1 = [i for i, x in enumerate(g['category']) if re.sub(r'\d+', '', self.id2object[x]).replace('_', ' ')==ques['ref']]
                assert (ques['qtype'] in ['extreme-obj', 'extreme-obj-hint'] and len(L1)==1) or ques['qtype'] in ['extreme-obj2', 'extreme-obj2-hint'], len(L1)
                if ques['qtype'] in ['extreme-obj2', 'extreme-obj2-hint']:
                    # L2 = [d[ques['loc2']] for d in g['position']]
                    L2 = [g['position'][i][ques['loc2']] for i in L1]
                    if ques['loc2'] in ['left', 'front']:
                        iy = L2.index(min(L2))
                    else:
                        iy = L2.index(max(L2))
                    iy = L1[iy]
                else:
                    iy = L1[0]

                L3 = [i for i, x in enumerate(g['place_rel']) if x==iy]
                L4 = [g['position'][i][ques['loc1']] for i in L3]
                if ques['loc1'] in ['left', 'front']:
                    ix = L4.index(min(L4))
                else:
                    ix = L4.index(max(L4))
                ix = L3[ix]

                
            cate = self.id2object[g['category'][ix]]
            ans_item = {'qtype': ques['qtype'], 'object': cate}
            
                    

        return ans_item

    def transform_to_language(self, ans_item):
        def get_article(x):
            if x[0] in 'aeiou': return 'an'
            return 'a'
        if ans_item['qtype'] in ['count-hint', 'count-nohint']:
            return self.num2eng[ans_item['cnt']]
        if ans_item['qtype'] in ['refer-it', 'refer-them']:
            if 'ref' in ans_item:
                return self.ref_text[ans_item['ref']]['singular'][0]
            else:
                ref_dict = ans_item['ref_dict']
                kvs = sorted(ref_dict.items())
                splits = []
                for k, v in kvs:
                    t = 'singular' if v==1 else 'plural'
                    ref = self.ref_text[k][t][0]
                    if v==1:
                        ref = get_article(ref) + ' ' + ref
                    else:
                        ref = self.num2eng[v] + ' ' + ref
                    splits.append(ref)
                if len(splits)<=2:
                    answer = ' and '.join(splits)
                else:
                    answer = ', '.join(splits[:-1]) + ' and ' + splits[-1]
                return answer
        if ans_item['qtype'].startswith('extreme'):
            return self.ref_text[ans_item['object']]['singular'][0]
        if ans_item['qtype'] in ['query-color', 'query-material']:
            if 'exception' in ans_item:
                return ans_item['exception']
            else:
                return ans_item['val']



if __name__=='__main__':

    # count
    # parser = QuestionParser()
    # s = 'there is a brown square table on the far front of the image, and you?'
    # imgid = '143227'
    # ques_item = parser.parse_question(s)
    # print(ques_item)
    # answerer_simulator = AnswerSimulator()
    # ans_item = answerer_simulator.answer(imgid, ques_item)
    # print(answerer_simulator.transform_to_language(ans_item))
    # exit()

    # ref2
    # last_ques = 'mine is different from yours. i can see two white three-drawer nightstands, and you?'
    # ques = 'this is different from mine. what are they?'
    # imgid = '130004'
    # ques_item = parser.parse_question(ques)
    # last_ques_item = parser.parse_question(last_ques)
    # print(ques_item)
    # answerer_simulator = AnswerSimulator()
    # ans_item = answerer_simulator.answer(imgid, ques_item, last_ques_item)
    # print(ans_item)
    # print(answerer_simulator.transform_to_language(ans_item))

    # query
    # parser = QuestionParser()
    # imgid = '143242'
    # ques = 'yes. what is the frontmost thing on the leftmost nightstand?'
    # ques_item = parser.parse_question(ques)
    # print(ques_item)
    # answerer_simulator = AnswerSimulator()
    # ans_item = answerer_simulator.answer(imgid, ques_item)
    # print(ans_item)
    # print(answerer_simulator.transform_to_language(ans_item))


    parser = QuestionParser()
    answer_simulator = AnswerSimulator()

    with open('data/0206/spot_diff_test.json') as f:
        dialogs = json.load(f)
    tot = 0
    correct = 0
    for d in dialogs:
        turn = len(d['questions'])
        img2 = d['img2'].split('/')[-1].split('.')[0][4:]
        last_ques = None
        for t in range(turn):
            try:
                ques_item = parser.parse_question(d['questions'][t].lower())
                ans_item = answer_simulator.answer(img2, ques_item, last_ques=last_ques)
                last_ques = ques_item
                ans = answer_simulator.transform_to_language(ans_item)
                if ans==d['answers'][t]:
                    correct += 1
                tot += 1
            except Exception as e:
                print(d['questions'][t].lower(), img2, e)
                pass
        
    print(correct/tot, correct, tot)