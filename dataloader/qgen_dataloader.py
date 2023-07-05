import json
import copy
import h5py
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from transformers import (
    PreTrainedTokenizer,
    AutoTokenizer
)

from dataloader.loader_utils import ImageFeaturesHdfReader, pad_sequence

class SpotDiffDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path, args):
        # super(Dataset, self).__init__()
        self.mode = args.mode
        assert self.mode in ['questioner', 'answerer']
        self.tokenizer = tokenizer
        self.block_size = args.block_size
        
        self.img_reader = ImageFeaturesHdfReader(args.img_feat_file, in_memory=False) 
        with open(file_path) as f:
            dialogs = json.load(f)
        
        self.data = []
        for dialog in tqdm(dialogs):
            ques_tokens = [tokenizer.tokenize(ques) for ques in dialog['questions']]
            ans_tokens = [tokenizer.tokenize(ans) for ans in dialog['answers']]
            
            img1_id = dialog['img1'].split('/')[-1]
            img1_id = int(img1_id.split('.')[0][4:])
            img2_id = dialog['img2'].split('/')[-1]
            img2_id = int(img2_id.split('.')[0][4:])

            self.data.append({'ques_tokens': ques_tokens, 'ans_tokens': ans_tokens, 'img1': img1_id,  'img2': img2_id}) 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):

        item = self.data[i]
        img_id = item['img1'] if self.mode in ['questioner'] else item['img2']
        vis_feats_dict = self.img_reader[img_id]
        vis_feats = torch.cat([vis_feats_dict['features'], vis_feats_dict['rel_boxes']], dim=1)

        input_ids = []
        token_type_ids = []
        labels = []
        dialog_turn = len(item['ques_tokens'])
        questions = []
        answers = []

        for turn_t in range(dialog_turn):
            ques_tokens = item['ques_tokens'][turn_t]
            ans_tokens = item['ans_tokens'][turn_t]
            img_tokens = ['[image]'] * 36 + ['[SEP]']

            # append history
            hist_tokens = []
            for i in range(turn_t):
                hist_tokens.extend(item['ques_tokens'][i] + item['ans_tokens'][i] + ['[SEP]'])

            if self.mode=='questioner':
                res_tokens = ['[BOS]'] + ques_tokens
            elif self.mode=='answerer':
                res_tokens = ['[BOS]'] + ans_tokens
                hist_tokens.extend(item['ques_tokens'][turn_t] + ['[SEP]'])

            # prepare input
            cur_input_ids = self.tokenizer.convert_tokens_to_ids(img_tokens + hist_tokens + res_tokens)
            cur_token_type_ids = self.tokenizer.convert_tokens_to_ids(['[image]'] * len(img_tokens) + ['[context]'] * len(hist_tokens) + ['[target]'] * len(res_tokens))
            cur_labels = [-100] * (len(img_tokens) + len(hist_tokens)) + self.tokenizer.convert_tokens_to_ids(res_tokens[1:] + ['[EOS]'])

            assert len(cur_input_ids)==len(cur_token_type_ids)==len(cur_labels)

            # pad
            token_len_t = len(cur_input_ids)
            cur_input_ids.extend([self.tokenizer.pad_token_id] * (self.block_size - token_len_t))
            cur_token_type_ids.extend([self.tokenizer.pad_token_id] * (self.block_size - token_len_t))
            cur_labels.extend([-100] * (self.block_size - token_len_t))

            input_ids.append(cur_input_ids)
            token_type_ids.append(cur_token_type_ids)
            labels.append(cur_labels)

            # questions && answers
            questions.append(item['ques_tokens'][turn_t])
            answers.append(item['ans_tokens'][turn_t])

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return {
            'vis_feats': vis_feats,
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'labels': labels,
            'dialog_turn': dialog_turn,
            'dialog_turn_mask': torch.ones(dialog_turn),
            'questions': questions,
            'answers': answers,
        }


def collate_batch(batch):
    # pad to max turn
    max_turn = max([x['dialog_turn'] for x in batch])
    for i, x in enumerate(batch):
        cur_turn = x['input_ids'].size(0)
        batch[i]['input_ids'] = pad_sequence(x['input_ids'], max_turn, y=0)
        batch[i]['token_type_ids'] = pad_sequence(x['token_type_ids'], max_turn, y=0)
        batch[i]['labels'] = pad_sequence(x['labels'], max_turn, y=-100)
        batch[i]['dialog_turn_mask'] = pad_sequence(x['dialog_turn_mask'], max_turn, y=0, dtype=torch.float)
    
    out = {}
    mergedBatch = {key: [d[key] for d in batch] for key in batch[0]}

    for key in mergedBatch:
        if key in ['questions', 'answers']:
            continue
        if isinstance(mergedBatch[key][0], int):
            out[key] = torch.tensor(mergedBatch[key])
        else:
            out[key] = torch.stack(mergedBatch[key])
        
    return out
 


def test():
    import argparse
    from torch.utils.data import DataLoader, RandomSampler
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='../data/spot_diff_train.json')
    parser.add_argument('--img_feat_file', type=str, default='../data/img_feat.h5')
    parser.add_argument('--model_name_or_path', type=str, default='gpt2')
    parser.add_argument('--cache_dir', type=str, default='../checkpoints/gpt2')
    parser.add_argument('--block_size', type=int, default=512)
    args = parser.parse_args()

    if args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, cache_dir=args.cache_dir
        )
        special_tokens_dict = {'bos_token': '[BOS]', 'eos_token': '[EOS]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'sep_token': '[SEP]', 'unk_token': '[UNK]'}
        tokenizer.add_special_tokens(special_tokens_dict)
        tokens_dict = ['[image]', '[context]', '[target]']
        tokenizer.add_tokens(tokens_dict)
    
    dataset = SpotDiffDataset(tokenizer, args.file_path, args)
    item = dataset[0]

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=2,
        collate_fn=collate_batch,
    )

    for batch in dataloader:
        vis_feats = batch['vis_feats']
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        labels = batch['labels']
        dialog_turn = batch['dialog_turn']

        for k in batch:
            print(k, batch[k].size())
        break


if __name__=='__main__':
    test()