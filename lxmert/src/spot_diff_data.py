# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle
import copy
import h5py
import numpy as np
import torch
from torch.functional import split
from torch.utils.data import Dataset

# from param import args
from utils import load_obj_tsv

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
# TINY_IMG_NUM = 512
# FAST_IMG_NUM = 5000

# The path to data and image features.
# VQA_DATA_ROOT = 'data/vqa/'
# MSCOCO_IMGFEAT_ROOT = 'data/mscoco_imgfeat/'
# SPLIT2NAME = {
#     'train': 'train2014',
#     'valid': 'val2014',
#     'minival': 'val2014',
#     'nominival': 'val2014',
#     'test': 'test2015',
# }
SPLIT2NAME = {
    'train': 'spot_diff_train',
    'valid': 'spot_diff_val',
    'test': 'spot_diff_test'
}


class ImageFeaturesHdfReader(object):
    """
    A reader for HDF files containing pre-extracted image features. A typical
    HDF file is expected to have a column named "image_id", and another column
    named "features".
    Example of an HDF file:
    ```
    visdial_train_faster_rcnn_bottomup_features.h5
       |--- "image_id" [shape: (num_images, )]
       |--- "features" [shape: (num_images, num_proposals, feature_size)]
       +--- .attrs ("split", "train")
    ```
    Refer ``$PROJECT_ROOT/data/extract_bottomup.py`` script for more details
    about HDF structure.
    Parameters
    ----------
    features_hdfpath : str
        Path to an HDF file containing VisDial v1.0 train, val or test split
        image features.
    in_memory : bool
        Whether to load the whole HDF file in memory. Beware, these files are
        sometimes tens of GBs in size. Set this to true if you have sufficient
        RAM - trade-off between speed and memory.
    """

    def __init__(self, features_hdfpath: str, in_memory: bool = False):
        self.features_hdfpath = features_hdfpath
        self._in_memory = in_memory
        print(self.features_hdfpath)
        with h5py.File(self.features_hdfpath, 'r') as features_hdf:
            # self._split = features_hdf.attrs["split"]
            self._image_id_list = list(features_hdf["img_id"])
            # "features" is List[np.ndarray] if the dataset is loaded in-memory
            # If not loaded in memory, then list of None.
            self.features = [None] * len(self._image_id_list)
            self.boxes = [None] * len(self._image_id_list)
            self.classes = [None] * len(self._image_id_list)
            self.scores = [None] * len(self._image_id_list)
            self.img_ws = [None] * len(self._image_id_list)
            self.img_hs = [None] * len(self._image_id_list)
            self.num_boxes_list = [None] * len(self._image_id_list)

    def __len__(self):
        return len(self._image_id_list)

    def __getitem__(self, image_id: int):
        index = self._image_id_list.index(image_id)
        if self._in_memory:
            # Load features during first epoch, all not loaded together as it
            # has a slow start.
            if self.features[index] is not None:
                image_id_features = self.features[index]
                boxes = self.boxes[index]
                single_class = self.classes[index]
                single_score = self.scores[index]
                img_w = self.img_ws[index]
                img_h = self.img_hs[index]
                num_boxes = self.num_boxes_list[index]

            else:
                with h5py.File(self.features_hdfpath, "r") as features_hdf:
                    image_id_features = features_hdf["features"][index]
                    boxes = features_hdf["boxes"][index]
                    single_class = features_hdf["objects_id"][index]
                    single_score = features_hdf["objects_conf"][index]
                    img_w = features_hdf['img_w'][index]
                    img_h = features_hdf['img_h'][index]
                    num_boxes = features_hdf['num_boxes'][index]

                    self.features[index] = image_id_features
                    self.boxes[index] = boxes
                    self.classes[index] = single_class
                    self.scores[index] = single_score
                    self.img_ws[index] = img_w
                    self.img_hs[index] = img_h
                    self.num_boxes_list[index] = num_boxes
 
        else:
            # Read chunk from file everytime if not loaded in memory.
            with h5py.File(self.features_hdfpath, "r") as features_hdf:
                image_id_features = features_hdf["features"][index]
                boxes = features_hdf["boxes"][index]
                single_class = features_hdf["objects_id"][index]
                single_score = features_hdf["objects_conf"][index]
                img_w = features_hdf['img_w'][index]
                img_h = features_hdf['img_h'][index]
                num_boxes = features_hdf['num_boxes'][index]
        
        boxes = torch.from_numpy(boxes)
        rel_boxes = boxes.clone()
        rel_boxes[:, [0, 2]] /= img_w
        rel_boxes[:, [1, 3]] /= img_h

        visual_attention_mask = (torch.arange(boxes.shape[0]) < num_boxes).long()

        return {
            'features': torch.from_numpy(image_id_features),
            'rel_boxes': rel_boxes,
            'visual_attention_mask': visual_attention_mask
        }
    
    def get_batch_img_info(self, img_id_list):
        batch = [self.__getitem__(i) for i in img_id_list]
        mergedBatch = {k: [d[k] for d in batch] for k in batch[0]}
        out = {}
        for key in mergedBatch:
            if isinstance(mergedBatch[key][0], int):
                out[key] = torch.tensor(mergedBatch[key])
            else:
                out[key] = torch.stack(mergedBatch[key])
        return out

class VQADataset:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
    """
    def __init__(self, splits, args):
        self.name = splits
        self.splits = splits.split(',')
        # Loading datasets
        self.data = []
        for split in self.splits:
            path = os.path.join(args.dataroot, '%s.json' % SPLIT2NAME[split])
            with open(path) as f:
                self.data.extend(json.load(f))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # # Convert list to dict (for evaluation)
        # self.id2datum = {
        #     datum['question_id']: datum
        #     for datum in self.data
        # }
        question_id = 0
        self.id2datum = {}
        data_list = []
        for dialog in self.data:
            turn = len(dialog['questions'])
            img_id = dialog['img2'].split('/')[-1]
            img_id = int(img_id.split('.')[0][4:])
            for t in range(turn):
                if t==0:
                    ques = dialog['questions'][0]
                else:
                    ques = dialog['questions'][t-1] + dialog['answers'][t-1] + ' ' + dialog['questions'][t]
                ans = dialog['answers'][t]
                ques_type = dialog['questions_type'][t]
                self.id2datum[question_id] = {
                    'question_id': question_id,
                    'img_id': img_id,
                    'sent': ques,
                    'label': {ans: 1.0},
                    'ques_type': ques_type
                }
                data_list.append(self.id2datum[question_id])
                question_id += 1
        self.data = data_list

        # Answers
        # self.ans2label = json.load(open("data/vqa/trainval_ans2label.json"))
        # self.label2ans = json.load(open("data/vqa/trainval_label2ans.json"))
        ans2label_path = os.path.join(args.dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(args.dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        assert len(self.ans2label) == len(self.label2ans)

        self.img_reader = ImageFeaturesHdfReader(args.img_feat_file, in_memory=False)


    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
class VQATorchDataset(Dataset):
    def __init__(self, dataset: VQADataset):
        super().__init__()
        self.raw_dataset = dataset

        # if args.tiny:
        #     topk = TINY_IMG_NUM
        # elif args.fast:
        #     topk = FAST_IMG_NUM
        # else:
        #     topk = None

        # Loading detection features to img_data
        # img_data = []
        # for split in dataset.splits:
        #     # Minival is 5K images in MS COCO, which is used in evaluating VQA/LXMERT-pre-training.
        #     # It is saved as the top 5K features in val2014_***.tsv
        #     load_topk = 5000 if (split == 'minival' and topk is None) else topk
        #     img_data.extend(load_obj_tsv(
        #         os.path.join(MSCOCO_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME[split])),
        #         topk=load_topk))

        # Convert img list to dict
        # self.imgid2img = {}
        # for img_datum in img_data:
        #     self.imgid2img[img_datum['img_id']] = img_datum

        # # Only kept the data with loaded image features
        # self.data = []
        # for datum in self.raw_dataset.data:
        #     if datum['img_id'] in self.imgid2img:
        #         self.data.append(datum)
        
        # append all data
        self.data = []
        for datum in self.raw_dataset.data:
            self.data.append(datum)
        

        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']

        # # Get image info
        # img_info = self.imgid2img[img_id]
        img_info = self.raw_dataset.img_reader[img_id]
        # obj_num = img_info['num_boxes']
        # feats = img_info['features'].copy()
        # boxes = img_info['boxes'].copy()
        # num_boxes = img_info['num_boxes']
        # visual_attention_mask = (torch.arange(boxes.shape[0])<num_boxes).long()
        # assert len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        # img_h, img_w = img_info['img_h'], img_info['img_w']
        # boxes = boxes.copy()
        # boxes[:, (0, 2)] /= img_w
        # boxes[:, (1, 3)] /= img_h
        # np.testing.assert_array_less(boxes, 1+1e-5)
        # np.testing.assert_array_less(-boxes, 0+1e-5)
        feats, boxes, visual_attention_mask = img_info['features'], img_info['rel_boxes'], img_info['visual_attention_mask']

        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, visual_attention_mask, target
        else:
            return ques_id, feats, boxes, ques, visual_attention_mask



class VQAEvaluator:
    def __init__(self, dataset: VQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        ques_type_total = {}
        ques_type_correct = {}
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
            t = datum['ques_type']
            ques_type_total[t] = ques_type_total.get(t, 0) + 1
            ques_type_correct[t] = ques_type_correct.get(t, 0) + label.get(ans, 0)

        print('===========================================')
        for k in ques_type_total:
            acc = 1.0 * ques_type_correct[k] / ques_type_total[k]
            print('[qtype: %s][total: %d][correct: %d][acc: %.4f]' % (k, ques_type_total[k], ques_type_correct[k], acc))

        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)


# process question type
colors = ['yellow', 'red', 'rose gold', 'purple', 'blue', 'black', 'gray', 'white', 'green', 'silver', 'pink', 'brown']
materials = ['paper', 'marble', 'leather', 'plastic', 'glass', 'ceramic', 'cloth', 'metal', 'rubber', 'wooden']
def parse_action(x):
    flg = False
    for d in '0123456789':
        if d in x:
            flg = True
    if flg:
        return {'detail': x}
    
    ret = {}
    for c in colors:
        if c in x:
            ret['color'] = c
            x = x.replace(c, '').strip()
    for m in materials:
        if m in x:
            ret['material'] = m
            x = x.replace(m, '').strip()
    if x!='':
        ret['category'] = x

    return ret
def question_type_mapping(cur_action, last_action=None):

    if cur_action[0] in ['ref1', 'ref2']:
        return 'ref'
    elif cur_action[0] in ['extreme1', 'extreme2', 'extreme3']:
        return 'extreme'
    else:
        x = parse_action(cur_action[1])
        if last_action is None:
            L = ['count'] + list(x.keys())
            return ' '.join(L)

# process_action
qtype_map = {'count-ques': 'count', 'count-desc': 'count', 'extreme1': 'extreme1', 'extreme2': 'extreme2', 'extreme3': 'extreme3', 'ref1': 'ref1', 'ref2': 'ref2'}
def action_mapping(qtype, qact):
    ret = qtype_map[qtype]
    if qact!='null':
        ret += ' ' + qact
    return ret

   

class VQG(Dataset):
    def __init__(self, split, tokenizer, args):
        super().__init__()
        self.tokenizer = tokenizer
        self.decoder_tokenizer = tokenizer
        if 'gpt2' in args.decoder_model:
            from transformers import AutoTokenizer
            self.decoder_tokenizer = AutoTokenizer.from_pretrained(args.decoder_model)
            for x in ['cls', 'sep', 'pad']:
                setattr(self.decoder_tokenizer, '%s_token'%x, self.decoder_tokenizer.bos_token)


        self.img_reader = ImageFeaturesHdfReader(args.img_feat_file, in_memory=False) 
        self.max_hist_len = args.max_hist_len
        self.max_target_len = args.max_target_len
        self.max_instance_num = 9
        
        # tasks
        self.with_question_type_cls = args.with_question_type_cls or args.with_decoder_question_type_cls
        self.with_object_cls = args.with_object_cls or args.with_decoder_object_cls
        self.with_spot_diff_cls = args.with_spot_diff_cls
        self.with_action_cls = args.with_action_cls

        # action
        self.generate_action = args.generate_action
        self.include_action = args.include_action

        self.reverse_target = args.reverse_target
        print('reverse_target:', self.reverse_target)

        filepath = os.path.join(args.dataroot, '%s.json' %SPLIT2NAME[split])
        with open(filepath) as f:
            dialogs = json.load(f)
        
        self.entries = []
        for dialog in dialogs:
            img_id = dialog['img1'].split('/')[-1]
            img_id = int(img_id.split('.')[0][4:])
            ques_tokens = [self.tokenizer.tokenize(ques) for ques in dialog['questions']]
            ans_tokens = [self.tokenizer.tokenize(ans) for ans in dialog['answers']]
            ques_tokens_dec = ques_tokens
            if self.tokenizer!=self.decoder_tokenizer:
                ques_tokens_dec = [self.decoder_tokenizer.tokenize(ques) for ques in dialog['questions']]

            info = {
                'img_id': img_id,
                'ques_tokens': ques_tokens,
                'ans_tokens': ans_tokens,
                'ques_tokens_dec': ques_tokens_dec,
                'candidate_ids': dialog['candidate_ids'],
                'gt_boxes': dialog['boxes'],
                'q_objects': dialog['q_objects'],
                'qtype': dialog['questions_type'],
                # 'qact': dialog['question_action']
            }

            if self.with_spot_diff_cls:
                spot_diff = [0]
                turn = len(dialog['questions'])
                for t in range(turn-1):
                    flg = 0
                    l1 = len(dialog['q_objects'][t])
                    l2 = len(dialog['a_objects'][t])
                    if (dialog['questions_type'][t] in ['count-ques', 'count-desc']) and (l1!=l2):
                        flg = 1
                    spot_diff.append(flg)
                info.update({
                    'spot_diff': spot_diff
                })
                assert len(spot_diff)==turn

            self.entries.append(info)

        if self.with_question_type_cls:
            # self.idx2qtype = ['count-ques', 'count-desc', 'extreme1', 'extreme2', 'extreme3', 'ref1', 'ref2']
            self.idx2qtype = ['count color category', 'ref', 'count material', 'count color material', 'extreme', 'count detail', 'count material category', 'count color', 'count category', 'count color material category']
            self.qtype2idx = {w:i for i, w in enumerate(self.idx2qtype)}

        if self.with_action_cls:
            import pickle
            with open(os.path.join(args.dataroot, 'cache', 'label2act.pkl'), 'rb') as f:
                self.label2act = pickle.load(f)
            with open(os.path.join(args.dataroot, 'cache', 'act2label.pkl'), 'rb') as f:
                self.act2label = pickle.load(f)


    
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, index):
        entry = self.entries[index]


        img_id = entry['img_id']
        img_info = self.img_reader[img_id]
        # obj_num = img_info['num_boxes']
        # feats = img_info['features'].copy()
        # boxes = img_info['boxes'].copy()
        # assert len(boxes)==len(feats)


        # # Normalize the boxes (to 0 ~ 1)
        # img_h, img_w = img_info['img_h'], img_info['img_w']
        # boxes = boxes.copy()
        # boxes[:, (0, 2)] /= img_w
        # boxes[:, (1, 3)] /= img_h
        # np.testing.assert_array_less(boxes, 1+1e-5)
        # np.testing.assert_array_less(-boxes, 0+1e-5)
        # feats = torch.tensor(feats)
        # boxes = torch.tensor(boxes)
        # num_boxes = img_info['num_boxes']
        # visual_attention_mask = (torch.arange(boxes.size(0))<num_boxes).long()
        feats, boxes, visual_attention_mask = img_info['features'], img_info['rel_boxes'], img_info['visual_attention_mask']

        ques_tokens = entry['ques_tokens']
        ans_tokens = entry['ans_tokens']
        ques_tokens_dec = entry['ques_tokens_dec']
        turn = len(entry['ques_tokens'])
        hist_tokens = []
        target_tokens = []
        for t in range(turn):
            cur_hist_tokens = [x for _t in range(t) for x in ques_tokens[_t]+ans_tokens[_t]]
            if self.include_action and t > 0:
                last_type = entry['qtype'][t-1]
                last_act = entry['qact'][t-1]
                if last_act!='null':
                    action = last_type + ' ' + last_act
                else:
                    action = last_type
                action_tokens = self.tokenizer.tokenize(action)
                cur_hist_tokens = cur_hist_tokens + ['[SEP]'] + action_tokens

            hist_tokens.append(
                ['[CLS]'] + cur_hist_tokens[:self.max_hist_len-2] + ['[SEP]']
            )
            cur_target_tokens = copy.deepcopy(ques_tokens_dec[t])

            if self.generate_action:
                cur_type = entry['qtype'][t]
                cur_act = entry['qact'][t]
                if cur_act!='null':
                    action = cur_type + ' ' + cur_act
                else:
                    action = cur_type
                action_tokens = self.tokenizer.tokenize(action)
                cur_target_tokens = action_tokens + ['$'] + cur_target_tokens

            cur_target_tokens = cur_target_tokens[:self.max_target_len-2]
            if self.reverse_target:
                cur_target_tokens.reverse()
            
            target_tokens.append(
                [self.decoder_tokenizer.cls_token] + cur_target_tokens + [self.decoder_tokenizer.sep_token]
            )

        # max_seq_len = max([len(x) for x in hist_tokens])
        input_ids = []
        attention_mask = []
        for t in range(turn):
            tokens = hist_tokens[t] + ['[PAD]'] * (self.max_hist_len - len(hist_tokens[t]))
            input_ids.append(
                self.tokenizer.convert_tokens_to_ids(tokens)
            )
            attention_mask.append(
                [1] * len(hist_tokens[t]) + [0] * (self.max_hist_len - len(hist_tokens[t]))
            )

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        token_type_ids = torch.zeros_like(attention_mask)

        # pad target with decoder_tokenizer
        decoder_input_ids = []
        decoder_attention_mask = []
        labels = []
        for t in range(turn):
            pad_len = self.max_target_len - (len(target_tokens[t]) - 1)
            decoder_input_ids.append(
                self.decoder_tokenizer.convert_tokens_to_ids(target_tokens[t][:-1] + [self.decoder_tokenizer.pad_token] * pad_len )
            )
            decoder_attention_mask.append(
                [1] * (len(target_tokens[t]) - 1) + [0] * pad_len
            )
            labels.append(
                self.decoder_tokenizer.convert_tokens_to_ids( target_tokens[t][1:]) + [-100] * pad_len
            )
            assert len(decoder_input_ids[t])==len(decoder_attention_mask[t])==len(labels[t])

        decoder_input_ids = torch.tensor(decoder_input_ids)
        decoder_attention_mask = torch.tensor(decoder_attention_mask)
        labels = torch.tensor(labels)

        ret = {
            'img_id': entry['img_id'],
            'features': feats,
            'boxes': boxes,
            'visual_attention_mask': visual_attention_mask,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'decoder_input_ids': decoder_input_ids,
            'labels': labels,
            'dialog_turn': turn,
            'dialog_turn_mask': torch.ones(turn, dtype=torch.long)
        }

        # question type classification
        if self.with_question_type_cls:
            qtype = []
            for t in range(turn):
                qt = question_type_mapping((entry['qtype'][t], entry['qact'][t]), None)
                qtype.append(self.qtype2idx[qt])
            ret.update({
                'qtype': torch.tensor(qtype)
            })
        
        if self.with_action_cls:
            actions = []
            for t in range(turn):
                act = action_mapping(entry['qtype'][t], entry['qact'][t])
                actions.append(self.act2label[act])
            ret.update({
                'action': torch.tensor(actions)
            })

        # object classification
        # if self.with_object_cls:
        #     pad_len = self.max_instance_num - len(entry['candidate_ids'])
        #     gt_objects_id = torch.tensor(entry['candidate_ids'] + [0] * pad_len)
        #     gt_objects_mask = torch.tensor([1] * len(entry['candidate_ids']) + [0] * pad_len)
        #     gt_boxes = torch.tensor(entry['gt_boxes'] + [[0,0,0,0]] * pad_len)
        #     gt_boxes[:, [0,2]] /= 1366
        #     gt_boxes[:, [1,3]] /= 768

        #     q_attention_objects = torch.zeros([turn, self.max_instance_num], dtype=torch.long)
        #     for t in range(turn):
        #         for i in entry['q_objects'][t]:
        #             q_attention_objects[t, i]=1
            
        #     ret.update({
        #         'gt_objects_id': gt_objects_id,
        #         'gt_objects_mask': gt_objects_mask,
        #         'gt_boxes': gt_boxes,
        #         'q_attention_objects': q_attention_objects
        #     })

        if self.with_spot_diff_cls:
            ret.update({
                'spot_diff': torch.tensor(entry['spot_diff'])
            })

        return ret

def collate_batch(batch):
    # pad to max turn
    max_turn = max([x['input_ids'].size(0) for x in batch])
    for i, x in enumerate(batch):
        cur_turn = x['input_ids'].size(0)
        # encoder
        batch[i]['input_ids'] = pad_sequence(x['input_ids'], max_turn)
        batch[i]['attention_mask'] = pad_sequence(x['attention_mask'], max_turn)
        batch[i]['token_type_ids'] = pad_sequence(x['token_type_ids'], max_turn)

        # decoder
        batch[i]['decoder_input_ids'] = pad_sequence(x['decoder_input_ids'], max_turn)       
        batch[i]['labels'] = pad_sequence(x['labels'], max_turn, y=-100)
        
        # dialog turn
        batch[i]['dialog_turn_mask'] = pad_sequence(x['dialog_turn_mask'], max_turn)


        if 'qtype' in batch[0]:
            batch[i]['qtype'] = pad_sequence(x['qtype'], max_turn)

        if 'q_attention_objects' in batch[0]:
            batch[i]['q_attention_objects'] = pad_sequence(x['q_attention_objects'], max_turn)
    
        if 'spot_diff' in batch[0]:
            batch[i]['spot_diff'] = pad_sequence(x['spot_diff'], max_turn)

        if 'action' in batch[0]:
            batch[i]['action'] = pad_sequence(x['action'], max_turn)

    
    out = {}
    mergedBatch = {key: [d[key] for d in batch] for key in batch[0]}

    for key in mergedBatch:
        if isinstance(mergedBatch[key][0], int):
            out[key] = torch.tensor(mergedBatch[key])
        else:
            out[key] = torch.stack(mergedBatch[key])
        
    return out

def pad_sequence(x, max_sequence_len, y=0, dtype=torch.long):
    cur_sequence_len = x.size(0)
    constants = torch.zeros([max_sequence_len - cur_sequence_len] + list(x.size()[1:]), dtype=dtype)
    constants.fill_(y)
    x = torch.cat([x, constants], dim=0)
    return x
