import re
import json
import copy

remove_template = ['so do i.', 'as same as you.', 'yes.', 'ok.', "this is different from mine.", "mine is different from yours.", "there are some differences.",  'i have one more than you.', 'mine is more than yours.', 'mine is less than yours.']
count1_template = [r'how many (.*) can you see?', r'how many (.*) are there?', r'how many (.*) do you have?', r'can you tell me how many %(ref)s in your image?", "i want to know the number of (.*) in your picture.']
count2_template = [r'i have (.*), how about you?', r'i can see (.*), and you?', r'there is (.*) in my picture, what about you?', r'there are (.*) in my picture, what about you?']
eng2num = {'one':1, 'a': 1, 'an': 1, 'two': 2, 'three':3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}
num2eng = {0: 'zero', 1: 'a', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}

# get img id
def get_imgid(x):
    if isinstance(x, int):
        return x
    return int(x.split('/')[-1].split('.')[0][4:])

# get ref_to_node
def get_ref_to_node(filename):
    with open(filename) as f:
        d = json.load(f)
    
    ret = {}
    for k, v in d.items():
        for x in v['singular']:
            ret[x.lower()] = k
        for x in v['plural']:
            ret[x.lower()] = k
    return ret

def get_oname(ann_file):
    id2object = {}
    with open(ann_file) as f:
        d = json.load(f)
        for item in d['annotation_definitions'][0]['spec']:
            id2object[item['label_id']] = item['label_name']
    return id2object

# get imgid_to_dialog
def get_imgid_to_dialog(spot_diff_file):
    with open(spot_diff_file) as f:
        dialogs = json.load(f)

    imgid_to_dialog = {}
    for d in dialogs:
        imgid = get_imgid(d['img1'])
        imgid_to_dialog[imgid] = {
            'candidate_ids': d['candidate_ids'],
            'boxes': d['boxes'],
            'gt_index': d['gt_index']
        }
    return imgid_to_dialog


class QuestionParser:
    def __init__(self, 
        ann_file='data/simulator/annotation_definitions.json', 
        spot_diff_file='data/0123/spot_diff_test.json', 
        ref_text_file='data/simulator/ref_text.json',
        asset_graph_file='data/simulator/asset_graph.json'
    ):
        self.oname = get_oname(ann_file)
        self.ref_to_node = get_ref_to_node(ref_text_file)
        self.imgid_to_dialog = get_imgid_to_dialog(spot_diff_file)
        with open(asset_graph_file) as f:
            self.obj_to_graph = json.load(f)

    def get_objs_with_node(self, imgid, node):
        objs = self.imgid_to_dialog[imgid]
        objs = [self.oname[o] for o in objs]
        ret = []
        for o in objs:
            g = self.obj_to_graph[o]
            flg = False
            for x in g['nodes']:
                if x['ref']==node:
                    flg = True
            if flg:
                ret.append(o)
        return ret

    def parse_question(self, ques):
        for p in remove_template:
            ques = ques.replace(p, '')
            ques = ques.strip()
        ques = ques.replace(' - ', '-')
        for p in count2_template:
            search_obj = re.search(p, ques)
            if search_obj:
                x = search_obj.group(1)
                search_obj2 = re.search(r'(a|an|one|two|three|four|five|six|seven|eight|nine) (.*)', x)
                if search_obj2 and search_obj2.span()[0]==0:
                    n = search_obj2.group(1)
                    e = search_obj2.group(2)
                    n = eng2num[n]
                    if e not in self.ref_to_node:
                        # print(ques)
                        return None
                    e = self.ref_to_node[e]

                    return {
                        'type': 'count2',
                        'cnt': n,
                        'node': e
                    }

        for p in count1_template:
            search_obj = re.search(p, ques)
            if search_obj:
                e = search_obj.group(1)
                if e not in self.ref_to_node:
                    return None
                e = self.ref_to_node[e]
                return {
                    'type': 'count1',
                    'node': e
                }
        return None

    def get_objs_with_node(self, imgid, node):
        cand = self.imgid_to_dialog[imgid]['candidate_ids']
        cand = [self.oname[o] for o in cand]
        objs = []
        for o in cand:
            g = self.obj_to_graph[o]
            flg = False
            for x in g['nodes']:
                if x['ref']==node:
                    flg = True
            if flg:
                objs.append(o)
        return objs

    def fix_question(self, ques, imgid):
        out = self.parse_question(ques)
        if out and out['type']=='count2':
            objs = self.get_objs_with_node(imgid, out['node'])
            gt_ques = ques
            if len(objs)!=out['cnt']:
                for x in eng2num:
                    p = ' ' + x + ' '
                    if p in ques:
                        gt_ques = ques.replace(p, ' '+num2eng[len(objs)]+' ')
            return {
                'type': out['type'],
                'node': out['node'],
                'objs': objs,
                'cnt1': out['cnt'],
                'cnt2': len(objs),
                'ques_correct': gt_ques
            }

        return {
            'type': out['type'] if out else 'none',
            'ques_correct': ques,
        }


if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='vqg_gen.json', type=str)
    args = parser.parse_args()

    with open(args.input_file) as f:
        dialogs = json.load(f)

    total = {}
    correct = {}
    dialogs_fix = []
    F = QuestionParser(spot_diff_file='../data/1013/spot_diff_val.json')

    error_cnt = {}

    for i, d in enumerate(dialogs):
        img1 = get_imgid(d['img1'])
        img2 = get_imgid(d['img2'])

        flg = False

        questions_fix = []
        for q in d['questions']:
            out = F.fix_question(q, img1) 
            ques_correct = out['ques_correct']
            if out['type']=='count2':
                e = out['node']
                n = out['cnt1']
                cnt = out['cnt2']
                total[e] = total.get(e, 0) + 1
                if n==cnt:
                    correct[e] = correct.get(e, 0) + 1
                else:
                    error_cnt[str(n) + ' ' + str(cnt)] = error_cnt.get(str(n) + ' ' + str(cnt), 0)+1
                    # print(out['ques_origin'], ques_correct)
            questions_fix.append(ques_correct)

        candidate_ids = F.imgid_to_dialog[img1]['candidate_ids']
        boxes = F.imgid_to_dialog[img1]['boxes']
        gt_index = F.imgid_to_dialog[img1]['gt_index']
    
        dialogs_fix.append({
            'img1': img1,
            'img2': img2,
            'questions': questions_fix,
            'answers': d['answers'],
            'candidate_ids': candidate_ids,
            'boxes': boxes,
            'gt_index': gt_index,
        })
    
    for x in total:
        acc = correct.get(x, 0) / total.get(x, 0)
        print('[%s][Total: %d][Correct: %d][Acc: %.4f]' % (x, total.get(x, 0), correct.get(x, 0), acc))

    sum_total = sum(list(total.values()))
    sum_correct = sum(list(correct.values()))
    print('[Total: %d][Correct: %d][Acc: %.4f]' % (sum_total, sum_correct, sum_correct/sum_total))

    for k, v in error_cnt.items():
        print(k, v)

    # with open(args.output_file, 'w') as f:
    #     json.dump(dialogs_fix, f)
