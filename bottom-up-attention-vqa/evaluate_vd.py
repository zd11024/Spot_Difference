"""
given VD questions, VQA model calculates answer
evaluate the performance of guessing with VQA model
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import base_model
from dataset import Dictionary, VD_question_dataset, image_pool_dataset, ImageFeaturesHdfReader

def to_var(x, use_cuda=True):
    if use_cuda:
        return x.cuda()
    return x

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--batch_size', type=int, default=2064) # ?
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--dataroot', type=str, default='/mnt/yardcephfs/mmyard/g_wxg_td_prc/mt/v_dozheng/projects/bottom-up-attention-vqa/data')
    parser.add_argument('--img_feature_file', type=str, default='/mnt/yardcephfs/mmyard/g_wxg_td_prc/mt/v_zpxu/visdial_zp/data/visdial/rcnn/features_faster_rcnn_x101_val.h5')
    parser.add_argument('--visdial_val_file', type=str, default='/mnt/yardcephfs/mmyard/g_wxg_td_prc/mt/v_dozheng/data/visdial/visdial_1.0_val.json')
    parser.add_argument('--model_path', type=str, default='/mnt/wfs/mmcommwfssz/project_wx-mm-spr-nlp/users/fandongmeng/1.experiments/v_dozheng/vqa/model.pth')
    parsed = vars(parser.parse_args())
    return parsed

print('Evaluating Vd questions')
params = parse_args()
print('params:', params)
# loading image feature
img_feature_reader=ImageFeaturesHdfReader(params['img_feature_file'])

# loading data
dictionary = Dictionary.load_from_file(params['dataroot'] + '/' + 'dictionary.pkl') 
dataset = VD_question_dataset(dictionary, img_feature_reader, visdial_val_file=params['visdial_val_file'], dataroot=params['dataroot'])
pool_dataset = image_pool_dataset(img_feature_reader)

# constrcuting model
print('constructing model..')
constructor = 'build_%s' % params['model']
model = getattr(base_model, constructor)(dataset, params['num_hid']).cuda()
state_dict = torch.load(params['model_path'])
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace('module.','') # remove `module.`
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)


# evaluating
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
dataloader_imgpool = DataLoader(pool_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=1)

n_pool = len(pool_dataset)
round_mean = [[] for i in range(10)]

print('starting evaluation...')
with torch.no_grad():
    for i, (image_id, image, ques) in enumerate(iter(dataloader)):
        use_cuda = True

        max_ques_len = ques.size(2)
        n_batch = ques.size(0)
        num_round = 10
        
        tgt_idx = to_var(torch.tensor([img_feature_reader.image_id_list.index(img_id) for img_id in image_id.tolist()]), use_cuda)  # (n_batch)
        round_prob = []

        question_list = []
        answer_list = []

        for round in range(num_round):
            cur_prob = []
            # cand_image_id_list = []
            # loading image pool
            for j, (cand_images, cand_image_ids) in enumerate(iter(dataloader_imgpool)):
                n_cand = cand_images.size(0)
                v = cand_images.unsqueeze(0).expand(n_batch, n_cand, 36, 2048).view(-1, 36, 2048)
                b = None
                q = ques[:, round, :].unsqueeze(1).expand(n_batch, n_cand, max_ques_len).view(-1, max_ques_len)
                
                v = to_var(v, use_cuda)
                q = to_var(q, use_cuda)

                pred = model(v, b, q, None)  # (n_batch * n_cand, n_ans)
                cur_prob += [pred.view(n_batch, n_cand, -1)]
                # cand_image_id_list += [cand_image_ids]

            cur_prob = torch.cat(cur_prob, dim=1)  # (n_batch, n_pool, n_ans)
            cur_prob = F.sigmoid(cur_prob)


            tgt_ans = torch.gather(cur_prob, index=tgt_idx.view(n_batch, 1, 1).expand(n_batch, 1, cur_prob.size(2)), dim=1).squeeze(1)  # (n_batch, n_ans)
            tgt_ans_idx = tgt_ans.max(dim=1)[1]  # (n_batch)
            cur_prob = torch.gather(cur_prob, index=tgt_ans_idx.view(n_batch, 1, 1).expand(n_batch, cur_prob.size(1), 1), dim=2).squeeze(2)  # (n_batch, n_pool)
            round_prob += [torch.log(cur_prob+1e-10)]
            
            # add question_list && answer_list
            question_list += [dictionary.decode(ques[0, round, :].tolist())]
            answer_list += [dataset.label2ans[tgt_ans_idx[0].item()]]
        
        # calculate accumulative probability
        prob_accum = to_var(torch.ones(n_batch, n_pool), use_cuda)
        for round in range(num_round):
            assert n_batch==1
            prob_accum = prob_accum + round_prob[round]  # (n_batch, n_pool)
            tgt_prob_accum = torch.gather(prob_accum, index=tgt_idx.view(n_batch, 1), dim=1).squeeze(1)  # (n_batch)
            mean = (prob_accum > tgt_prob_accum.unsqueeze(1)).float().sum(dim=1) + 1
            round_mean[round] += [mean[0].item()]
        
        if i%20==0:
            print(i, 'mean:', round_mean[round][-1])
            print('image_id', image_id[0].item(), 'img_idx', tgt_idx[0].item())
            for round in range(num_round):
                print('Q:', question_list[round], 'A:', answer_list[round])
            for round in range(10):
                mean = torch.tensor(round_mean[round]).mean()
                print('round %d mean %.3g' % (round, mean))

        
print('===========================================')
for round in range(10):
    mean = torch.tensor(round_mean[round]).mean()
    print('round %d mean %.3g' % (round, mean))
