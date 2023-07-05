"""
train VQA model
"""
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import h5py
import argparse
from torch.utils.data import DataLoader
import numpy as np

import utils
import base_model
from models.vqg import DecoderWithAttention
from dataset import Dictionary, VQAFeatureDataset, ImageFeaturesHdfReader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='/mnt/wfs/mmcommwfssz/project_wx-mm-spr-nlp/users/fandongmeng/1.experiments/v_dozheng/vqa')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--dataroot', type=str, default='/mnt/yardcephfs/mmyard/g_wxg_td_prc/mt/v_dozheng/projects/bottom-up-attention-vqa/data')
    parser.add_argument('--img_feature_file', type=str, default='/mnt/yardcephfs/mmyard/g_wxg_td_prc/mt/v_zpxu/visdial_zp/data/visdial/rcnn/features_faster_rcnn_x101_train.h5')
    parser.add_argument('--model_path', type=str, default='/mnt/wfs/mmcommwfssz/project_wx-mm-spr-nlp/users/fandongmeng/1.experiments/v_dozheng/vqg2/model_3.pth')
    
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--emb_dim', default=1000, type=int)
    parser.add_argument('--attention_dim', default=1024, type=int)
    parser.add_argument('--decoder_dim', default=1024, type=int)
    parsed = vars(parser.parse_args())
    return parsed


# if __name__ == '__main__':
#     params = parse_args()
#     dataroot = params['dataroot']
#     torch.manual_seed(params['seed'])
#     torch.cuda.manual_seed(params['seed'])
#     torch.backends.cudnn.benchmark = True
#     print('params:', params)

#     # loading image feature
#     img_feature_reader=ImageFeaturesHdfReader(params['img_feature_file'])

#     # loading dictionary
#     print('Start loading dataset.')    
#     dictionary = Dictionary.load_from_file(dataroot + '/' + 'dictionary.pkl') 
#     dataset = VQAFeatureDataset('train', dictionary, dataroot=dataroot, img_feature_reader=img_feature_reader)
#     batch_size = params['batch_size']
#     print('Finish loading model.')

#     # constrcuting model
#     print('constructing model..')
#     constructor = 'build_%s' % params['model']
#     model = getattr(base_model, constructor)(dataset, params['num_hid']).cuda()
#     state_dict = torch.load(params['model_path'])
#     # create new OrderedDict that does not contain `module.`
#     from collections import OrderedDict
#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         name = k.replace('module.','') # remove `module.`
#         new_state_dict[name] = v
#     model.load_state_dict(new_state_dict)

#     loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=1)

#     for i, (img_id, v, b, ques, ques_len, a) in enumerate(loader):
        
#         batch_size = img_id.size(0)
#         q1 = ques.unsqueeze(1).to('cuda').repeat(1, batch_size, 1).view(batch_size*batch_size, -1)
#         v1 = v.unsqueeze(0).to('cuda').repeat(batch_size, 1, 1, 1).view(batch_size*batch_size, 36, 2048)
#         pred = model(v1, None, q1, None).view(batch_size, batch_size, -1)  # (q, i, a)
#         pred = F.normalize(torch.sigmoid(pred), dim=-1)

#         p_a = pred.mean(1)  # (q, a)
        
#         p_i_a = (pred / batch_size).transpose(1, 2)  # (q, a, i)
#         p_i_a = p_i_a / p_a.unsqueeze(-1)  # (q, a, i)

#         H_a = - (p_i_a * torch.log(p_i_a)).sum(-1)  # (q, a)

#         ig_q = (p_a * H_a).sum(1)

#         if i%1==0:
#             print(img_id.tolist())
#             idx = ig_q.min(0)[1]
#             print('best ques:', dictionary.decode(ques[idx].tolist()), 'ig_q:', ig_q.tolist())
#             for i in range(20):
#                 print(dictionary.decode(ques[i].tolist()))
#         if i >=50:
#             break

if __name__=='__main__':
    params  = parse_args()
    dataroot = params['dataroot']
    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed(params['seed'])
    torch.backends.cudnn.benchmark = True
    print('params:', params)

    # loading image feature
    img_feature_reader=ImageFeaturesHdfReader(params['img_feature_file'])

    # loading dictionary
    print('Start loading dataset.')    
    dictionary = Dictionary.load_from_file(dataroot + '/' + 'dictionary.pkl') 
    train_dset = VQAFeatureDataset('train', dictionary, dataroot=dataroot, img_feature_reader=img_feature_reader, add_special=True)
    eval_dset = VQAFeatureDataset('val', dictionary, dataroot=dataroot, img_feature_reader=img_feature_reader, add_special=True)
    batch_size = params['batch_size']
    print('Finish loading dataset.')    

    # constructing model
    print('start constructing model.')
    model = DecoderWithAttention(attention_dim=params['attention_dim'], embed_dim=params['emb_dim'], decoder_dim=params['decoder_dim'], dropout=params['dropout'], dictionary=dictionary).to('cuda')
    state_dict = torch.load(params['model_path'])
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.','') # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    print('finish constructing model')

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1)
    model.eval()
    for i, (img_id, v, b, q, q_l, a) in enumerate(train_loader):     
        imgs = Variable(v).cuda()
        ques = Variable(q).cuda()
        ques_len = Variable(q_l).cuda()

        seq = model.beam_search(imgs)

        if i<=20:
            print(img_id, dictionary.decode(seq))
        else:
            break
    print('============================================')


