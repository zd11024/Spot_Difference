import json
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from dataloader.loader_utils import pad_sequence, ImageFeaturesHdfReader


class SpotDiffDataset4Guesser(Dataset):
    def __init__(self, tokenizer, file_path, args, block_size=512, with_img_feat=False, id_list=None):

        self.tokenizer = tokenizer
        self.block_size = block_size
        self.with_img_feat = with_img_feat
        if self.with_img_feat:
            self.img_reader = ImageFeaturesHdfReader(args.img_feat_file, in_memory=False) 


        with open(file_path) as f:
            dialogs = json.load(f)
        
        self.data = []
        for i, dialog in enumerate(dialogs):
            img1 = dialog['img1'] # image id
            candidate_ids = dialog['candidate_ids']
            boxes = dialog['boxes']
            labels = dialog['gt_index']
            ques_tokens = [tokenizer.tokenize(ques) for ques in dialog['questions']]
            ans_tokens = [tokenizer.tokenize(ans) for ans in dialog['answers']]

            if isinstance(dialog['img1'], int):
                img1_id = dialog['img1']
            else:
                img1_id = dialog['img1'].split('/')[-1]
                img1_id = int(img1_id.split('.')[0][4:])

            if isinstance(dialog['img2'], int):
                img2_id = dialog['img2']
            else:
                img2_id = dialog['img2'].split('/')[-1]
                img2_id = int(img2_id.split('.')[0][4:])

            if (id_list is None) or (i in id_list):
                self.data.append({
                    'questions': dialog['questions'],
                    'answers': dialog['answers'],
                    'ques_tokens': ques_tokens, 
                    'ans_tokens': ans_tokens, 
                    'candidate_ids': candidate_ids, 
                    'boxes': boxes, 
                    'labels': labels, 
                    'img1': img1_id, 
                    'img2': img2_id
                })

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        item = self.data[i]
        
        # append history
        input_ids = [self.tokenizer.cls_token_id]
        dialog_turn = len(item['ques_tokens'])
        for turn_t in range(dialog_turn):
            cur_input_ids = self.tokenizer.convert_tokens_to_ids(item['ques_tokens'][turn_t] + item['ans_tokens'][turn_t]) + [self.tokenizer.sep_token_id]
            input_ids.extend(cur_input_ids)
        

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.float)
        boxes = torch.tensor(item['boxes'], dtype=torch.float)
        relateive_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        vis_pe = torch.cat([boxes, relateive_area.unsqueeze(1)], dim=1)
        vis_pe[:,[0,2,4]] /= 800
        vis_pe[:, [1,3,4]] /= 480 # vis_pe: [x1 / w, y1 / h, x2 / w, y2 / h, (x2-x1)*(y2-y1) / (w*h)], w is 1366, h is 768


        candidate_ids = torch.tensor(item['candidate_ids'])
        labels = item['labels']
        candidate_mask = torch.ones_like(candidate_ids) # 1 retain

        ret = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'candidate_ids': candidate_ids,
            'boxes': vis_pe,
            'labels': labels,
            'candidate_mask': candidate_mask,
            'img1': item['img1'],
            'img2': item['img2'],
            'questions': item['questions'],
            'answers': item['answers'],
        }


        if self.with_img_feat:
            img1_info = self.img_reader[item['img1']]
            img2_info = self.img_reader[item['img2']]
            ret.update({
                'vis_feats1': img1_info['features'], 
                'rel_boxes1': img1_info['rel_boxes'],
                'vis_feats2': img2_info['features'],
                'rel_boxes2': img2_info['rel_boxes']
            })

        return ret



def collate_batch_guesser(batch):
    max_sequence_len = max([x['input_ids'].size(0) for x in batch])
    max_object_num = max(x['candidate_ids'].size(0) for x in batch)
    for i, x in enumerate(batch):
        batch[i]['input_ids'] = pad_sequence(x['input_ids'], max_sequence_len, y=0, dtype=torch.long)
        batch[i]['attention_mask'] = pad_sequence(x['attention_mask'], max_sequence_len, y=0, dtype=torch.float)
        
        batch[i]['candidate_ids'] = pad_sequence(x['candidate_ids'], max_object_num, y=0, dtype=torch.long)
        batch[i]['boxes'] = pad_sequence(x['boxes'], max_object_num, y=0, dtype=torch.float)
        batch[i]['candidate_mask'] = pad_sequence(x['candidate_mask'], max_object_num, y=0, dtype=torch.long)

    out = {}
    mergedBatch = {key: [d[key] for d in batch] for key in batch[0]}

    for key in mergedBatch:
        if key in ['questions', 'answers']:
            out[key] = mergedBatch[key]
            continue
        if isinstance(mergedBatch[key][0], int):
            out[key] = torch.tensor(mergedBatch[key])
        else:
            out[key] = torch.stack(mergedBatch[key])
    return out

