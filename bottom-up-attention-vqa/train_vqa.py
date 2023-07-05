"""
train VQA model
"""
import os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import h5py
import argparse
from torch.utils.data import DataLoader
import numpy as np

import utils
import base_model
from dataset import Dictionary, SpotDiffDataset, ImageFeaturesHdfReader, collate_fn


def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def train(model, train_loader, eval_loader, num_epochs, output, device):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    # logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0
    print('Staring training...')
    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()

        model.train()

        for i, batch in enumerate(train_loader):
            v = batch['features'].to(device)
            b = batch['spatials'].to(device)
            q = batch['question'].to(device)
            a = batch['target'].to(device)

            pred = model(v, b, q, a)
            loss = instance_bce_with_logits(pred, a)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.item() * v.size(0)
            train_score += batch_score

            if i%10==0:
                cur_epoch = epoch + i / (len(train_loader.dataset) / train_loader.batch_size)
                print_line = '[Epoch: %.3g][Loss: %.3g][score: %.3g]' % (cur_epoch, loss.item(), batch_score)
                print(print_line)

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        
        # eval
        model.eval()
        eval_score, bound = evaluate(model, eval_loader, device)

        print('epoch %d, time: %.2f' % (epoch, time.time()-t))
        print('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        print('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        if eval_score > best_eval_score:
            print('Saving model %.3g' % eval_score)
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score


def evaluate(model, dataloader, device):
    score = 0
    upper_bound = 0
    num_data = 0

    ques_type_correct = {}
    ques_type_total = {}

    for batch in dataloader:

        v = batch['features'].to(device)
        b = batch['spatials'].to(device)
        q = batch['question'].to(device)
        a = batch['target']
        ques_type = batch['ques_type']
        pred = model(v, b, q, None)
        batch_score = compute_score_with_logits(pred, a.to(device)).sum()
        score += batch_score
        upper_bound += (a.max(1)[0]).sum()
        num_data += pred.size(0)

        batch_size = v.size(0)
        item_score = compute_score_with_logits(pred, a.to(device)).sum(1)
        for i in range(batch_size):
            t = ques_type[i]
            ques_type_total[t] = ques_type_total.get(t, 0) + 1
            ques_type_correct[t] = ques_type_correct.get(t, 0) + item_score[i].item()

    print('===========================================')
    for k in ques_type_total:
        acc = 1.0 * ques_type_correct[k] / ques_type_total[k]
        print('[qtype: %s][total: %d][correct: %d][acc: %.4f]' % (k, ques_type_total[k], ques_type_correct[k], acc))
    
    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    return score, upper_bound



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='/mnt/wfs/mmcommwfssz/project_wx-mm-spr-nlp/users/fandongmeng/1.experiments/v_dozheng/vqa')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--dataroot', type=str, default='/mnt/yardcephfs/mmyard/g_wxg_td_prc/mt/v_dozheng/projects/bottom-up-attention-vqa/data')
    parser.add_argument('--img_feature_file', type=str, default='/mnt/yardcephfs/mmyard/g_wxg_td_prc/mt/v_zpxu/visdial_zp/data/visdial/rcnn/features_faster_rcnn_x101_train.h5')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val'])
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--use_gpu', type=int, default=1)
    parsed = vars(parser.parse_args())
    return parsed


if __name__ == '__main__':
    params = parse_args()
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
    train_dset = SpotDiffDataset('train', dictionary, dataroot=dataroot, img_feature_reader=img_feature_reader)
    eval_dset = SpotDiffDataset('val', dictionary, dataroot=dataroot, img_feature_reader=img_feature_reader)
    batch_size = params['batch_size']
    print('Finish loading model.')

    device = torch.device('cuda' if torch.cuda.is_available() and params['use_gpu'] else 'cpu')
    print(f'device: {device}')
    if params['mode']=='train':
        # assert params['model_path'] is not None, 'Model path cannot be None!!!'

        # constructing model
        print('Start construting model.')
        constructor = 'build_%s' % params['model']
        model = getattr(base_model, constructor)(train_dset, params['num_hid']).to(device)
        model.w_emb.init_embedding(dataroot + '/' + 'glove6b_init_300d.npy')
        print('Finish constructing model.')

        if torch.cuda.is_available() and params['use_gpu']:
            model = nn.DataParallel(model).cuda()

        # training
        train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=params['num_workers'], collate_fn=collate_fn)
        eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=params['num_workers'], collate_fn=collate_fn)
        train(model, train_loader, eval_loader, params['epochs'], params['output'], device)
    elif params['mode']=='val':

        def load_model(model, model_path):
            from collections import OrderedDict
            state_dict = torch.load(model_path, map_location='cpu')
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

        print('Start construting model.')
        constructor = 'build_%s' % params['model']
        model = getattr(base_model, constructor)(train_dset, params['num_hid']).to(device)
        load_model(model, params['model_path'])
        if torch.cuda.is_available() and params['use_gpu']:
            model = nn.DataParallel(model).to(device)
        eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=params['num_workers'], collate_fn=collate_fn)
        eval_score, bound = evaluate(model, eval_loader, device)
        print('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
