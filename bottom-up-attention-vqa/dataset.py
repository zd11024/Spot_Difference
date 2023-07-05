import os
import json
import pickle
import numpy as np
import utils
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Dict, List, Union
from timeit import default_timer as timer


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

    def __init__(self, features_hdfpath: str, in_memory: bool = False, vd_bert=False):
        self.features_hdfpath = features_hdfpath
        self._in_memory = in_memory
        self.vd_bert = vd_bert
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

            else:
                with h5py.File(self.features_hdfpath, "r") as features_hdf:
                    image_id_features = features_hdf["features"][index]
                    boxes = features_hdf["boxes"][index]
                    single_class = features_hdf["objects_id"][index]
                    single_score = features_hdf["objects_conf"][index]
                    img_w = features_hdf['img_w'][index]
                    img_h = features_hdf['img_h'][index]

                    self.features[index] = image_id_features
                    self.boxes[index] = boxes
                    self.classes[index] = single_class
                    self.scores[index] = single_score
                    self.img_ws[index] = img_w
                    self.img_hs[index] = img_h
 
        else:
            # Read chunk from file everytime if not loaded in memory.
            with h5py.File(self.features_hdfpath, "r") as features_hdf:
                image_id_features = features_hdf["features"][index]
                boxes = features_hdf["boxes"][index]
                single_class = features_hdf["objects_id"][index]
                single_score = features_hdf["objects_conf"][index]
                img_w = features_hdf['img_w'][index]
                img_h = features_hdf['img_h'][index]
        
        return image_id_features

class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', ' ').replace('?', ' ').replace('.', ' ').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                if w in self.word2idx:
                    tokens.append(self.word2idx[w])
                else:
                    tokens.append(self.padding_idx)
        return tokens

    def dump_to_file(self, path):
        pickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = pickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer}
    return entry


def _load_dataset(dataroot, name, img_id2val):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    question_path = os.path.join(
        dataroot, 'VQA', 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = pickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])

    utils.assert_eq(len(questions), len(answers))
    entries = []
    for question, answer in zip(questions, answers):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        entries.append(_create_entry(img_id2val[img_id], question, answer))

    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, img_feature_reader, dataroot='data', add_special=False, reverse=False):
        super(VQAFeatureDataset, self).__init__()
        self.img_feature_reader=img_feature_reader
        self.img_id2idx=self.img_feature_reader.img_id2idx
        assert name in ['train', 'val']

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)
        self.add_special=add_special
        self.reverse=reverse

        self.dictionary = dictionary

        # self.img_id2idx = pickle.load(
        #     open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % name)))
        # print('loading features from h5 file')
        # h5_path = os.path.join(dataroot, '%s36.hdf5' % name)
        # with h5py.File(h5_path, 'r') as hf:
        #     # self.features = np.array(hf.get('image_features'))
        #     # self.spatials = np.array(hf.get('spatial_features'))
        #     self.features = np.array(hf.get('features'))
        #     self.image_id_list = list(hf.get(["image_id"]))

        # self.img_id2idx = {img_id: i for i, img_id in enumerate(self.image_id_list)}
        self.entries = _load_dataset(dataroot, name, self.img_id2idx)

        self.tokenize()
        self.tensorize()
        self.v_dim = self.img_feature_reader.features.size(2)
        # self.v_dim = self.features.size(2)
        # self.s_dim = self.spatials.size(2)

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            if self.reverse:
                tokens.reverse()
            tokens = tokens[:max_length]
            if self.add_special:
                tokens = [self.dictionary.bos_idx] + tokens + [self.dictionary.eos_idx]
                if len(tokens) < max_length+2:
                    padding = [self.dictionary.padding_idx] * (max_length + 2 - len(tokens))
                    tokens = padding + tokens
            else:
                if len(tokens) < max_length:
                    # Note here we pad in front of the sentence
                    padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                    tokens = padding + tokens
                utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        # self.features = torch.from_numpy(self.features)
        # self.spatials = torch.from_numpy(self.spatials)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        image_id = entry['image_id']
        # features = self.features[entry['image']]
        features = self.img_feature_reader[entry['image_id']]
        spatials = torch.zeros(36, 6)
        # spatials = self.spatials[entry['image']]

        question = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)

        ques_len = (question!=self.dictionary.padding_idx).sum().item()

        return image_id, features, spatials, question, ques_len, target

    def __len__(self):
        return len(self.entries)



def _load_dataset_spot_diff(dataroot, name, ans2label):
    filename = os.path.join(dataroot, 'spot_diff_%s.json' %name)
    with open(filename) as f:
        dialogs = json.load(f)
    ret = []
    for dialog in dialogs:
        turn = len(dialog['questions'])
        img2_id = dialog['img2'].split('/')[-1]
        img2_id = int(img2_id.split('.')[0][4:])
        for t in range(turn):
            if t==0:
                ques = dialog['questions'][t]
            else:
                ques = dialog['questions'][t-1] + dialog['answers'][t-1] + ' ' + dialog['questions'][t]
            ans = dialog['answers'][t]
            ques_type = dialog['questions_type'][t]
            ret.append({
                'image_id': img2_id,
                'question': ques,
                'answer': ans2label[ans],
                'ques_type': ques_type
            })
    return ret

# for SpotDiff Dataset
class SpotDiffDataset(Dataset):
    def __init__(self, name, dictionary, img_feature_reader, dataroot='data', add_special=False, reverse=False):
        super(SpotDiffDataset, self).__init__()
        self.img_feature_reader=img_feature_reader
        assert name in ['train', 'val']

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)
        self.add_special=add_special
        self.reverse=reverse

        self.dictionary = dictionary

        self.entries = _load_dataset_spot_diff(dataroot, name, self.ans2label)

        self.tokenize()
        self.tensorize()
        # self.v_dim = self.img_feature_reader.features.size(2)
        self.v_dim = 2048

   
    def tokenize(self, max_length=50):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            if self.reverse:
                tokens.reverse()
            tokens = tokens[:max_length]
            if self.add_special:
                tokens = [self.dictionary.bos_idx] + tokens + [self.dictionary.eos_idx]
                if len(tokens) < max_length+2:
                    padding = [self.dictionary.padding_idx] * (max_length + 2 - len(tokens))
                    tokens = padding + tokens
            else:
                if len(tokens) < max_length:
                    # Note here we pad in front of the sentence
                    padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                    tokens = padding + tokens
                utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            entry['answer'] = torch.tensor(entry['answer'])
            # labels = np.array(answer['labels'])
            # scores = np.array(answer['scores'], dtype=np.float32)
            # if len(labels):
            #     labels = torch.from_numpy(labels)
            #     scores = torch.from_numpy(scores)
            #     entry['answer']['labels'] = labels
            #     entry['answer']['scores'] = scores
            # else:
            #     entry['answer']['labels'] = None
            #     entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        image_id = entry['image_id']
        # features = self.features[entry['image']]
        features = torch.tensor(self.img_feature_reader[entry['image_id']])
        spatials = torch.zeros(36, 6)
        # spatials = self.spatials[entry['image']]

        question = torch.tensor(entry['q_token'])
        # answer = entry['answer']
        # labels = answer['labels']
        # scores = answer['scores']
        # target = torch.zeros(self.num_ans_candidates)
        # if labels is not None:
            # target.scatter_(0, labels, scores)

        answer = entry['answer']
        target = torch.zeros(self.num_ans_candidates)
        if answer is not None:
            target.scatter_(0, answer, 1)

        ques_len = (question!=self.dictionary.padding_idx).sum().item()

        # return image_id, features, spatials, question, ques_len, target
        return {
            'image_id': image_id,
            'features': features,
            'spatials': spatials,
            'question': question,
            'ques_len': ques_len,
            'target': target,
            'ques_type': entry['ques_type']
        }

    def __len__(self):
        return len(self.entries)

def collate_fn(batch):
    out = {}
    mergedBatch = {key: [d[key] for d in batch] for key in batch[0]}
    for key in mergedBatch:
        if isinstance(mergedBatch[key][0], int):
            out[key] = torch.tensor(mergedBatch[key])
        elif isinstance(mergedBatch[key][0], str):
            out[key] = mergedBatch[key]
        else:
            out[key] = torch.stack(mergedBatch[key])
    return out