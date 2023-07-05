import os
import json
import pickle
import torch
import argparse
from torch.utils.data import DataLoader, SequentialSampler

from transformers import (
    AutoTokenizer
)

import sys
from dataloader.loader_utils import pad_sequence_list
import ray

def get_vqg_input(questions, answers, tokenizer, max_len):
    batch_size = len(questions)
    input_ids = []
    attention_mask = []
    token_type_ids = []
    for b in range(batch_size):
        tokens = ['[image]'] * 36 + ['[SEP]']
        types = ['[image]'] * 37
        turn = len(questions[b])
        for t in range(turn):
            cur_tokens = tokenizer.tokenize(questions[b][t]) + tokenizer.tokenize(answers[b][t]) + ['[SEP]']
            tokens += cur_tokens
            types += ['[context]'] * len(cur_tokens)
        
        tokens = tokens[:max_len-1] + ['[BOS]']
        types = types[:max_len-1] + ['[target]']
        pad_len = max_len - len(tokens)
        input_ids.append(torch.tensor(tokenizer.convert_tokens_to_ids(tokens)))
        attention_mask.append(torch.tensor([1] * len(tokens)))
        token_type_ids.append(torch.tensor(tokenizer.convert_tokens_to_ids(types)))
    
    # input_ids = torch.tensor(input_ids)
    # attention_mask = torch.tensor(attention_mask)
    # token_type_ids = torch.tensor(token_type_ids)
    input_ids = pad_sequence_list(input_ids, y=tokenizer.pad_token_id)
    attention_mask = pad_sequence_list(attention_mask)
    token_type_ids = pad_sequence_list(token_type_ids, y=tokenizer.pad_token_id)
    return input_ids, attention_mask, token_type_ids



def get_vqa_input(questions, answers):
    batch_size = len(answers)
    # if turn==0:
    #     sent = [questions[b][0] for b in range(batch_size)]
    # else:
    #     sent = [questions[b][turn-1] + answers[b][turn-1] + ' ' + questions[b][turn] for b in range(batch_size)]
    sent = []
    for b in range(batch_size):
        turn = len(questions[b])-1
        if turn==0:
            sent.append(questions[b][0])
        else:
            sent.append(questions[b][turn-1] + answers[b][turn-1] + ' ' + questions[b][turn])
    return sent

def get_guesser_input(questions, answers, tokenizer, max_len):
    batch_size = len(questions)
    input_ids = []
    attention_mask = []
    for b in range(batch_size):
        tokens = []
        turn = len(questions[b])
        for t in range(turn):
            tokens += tokenizer.tokenize(questions[b][t]) + tokenizer.tokenize(answers[b][t]) + [tokenizer.sep_token]
        pad_len = max_len - len(tokens)
        input_ids.append(tokenizer.convert_tokens_to_ids(tokens) + [0] * pad_len)
        attention_mask.append([1] * len(tokens) + [0] * pad_len)
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    return input_ids, attention_mask


@ray.remote(num_gpus=1)
def solve(args, id_list):

    sys.path.append(args.lxrt_path)
    
    if args.fix_question:
        from tools.cate_score import QuestionParser
        F = QuestionParser(ann_file=os.path.join(args.dataroot, 'annotation_definitions.json'),
                        spot_diff_file=os.path.join(args.dataroot, args.eval_file),
                        ref_text_file=os.path.join(args.dataroot, 'ref_text.json'),
                        asset_graph_file=os.path.join(args.dataroot, 'asset_graph.json'))

    with open(os.path.join(args.dataroot, 'cache', 'trainval_ans2label.pkl'), 'rb') as f:
        ans2label = pickle.load(f)
    with open(os.path.join(args.dataroot, 'cache', 'trainval_label2ans.pkl'), 'rb') as f:
        label2ans = pickle.load(f) 
    
    if args.with_action_cls:
        with open(os.path.join(args.dataroot, 'cache', 'label2act.pkl'), 'rb') as f:
            label2act = pickle.load(f)
    
    from lxmert.src.vqa_model import VQAModel
    vqa = VQAModel(args, len(ans2label), args.max_vqa_len).to(args.device)
    state_dict = torch.load(args.vqa_model)
    vqa.load_state_dict(state_dict)

    # loading vqg model
    from model.qgen import GPT2Generator
    vqg = GPT2Generator.from_pretrained(args.vqg_model).to(args.device)

    def load_qgen_tokenizer(model_name_or_path):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        special_tokens_dict = {'bos_token': '[BOS]', 'eos_token': '[EOS]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'sep_token': '[SEP]', 'unk_token': '[UNK]'}
        tokenizer.add_special_tokens(special_tokens_dict)
        tokens_dict = ['[image]', '[context]', '[target]']
        tokenizer.add_tokens(tokens_dict)
        return tokenizer

    vqg_tokenizer = load_qgen_tokenizer(args.gpt_model)


    from model.guesser import BertGuesser
    guesser = BertGuesser.from_pretrained(args.guesser_model).to(args.device)
    guesser_tokenizer = AutoTokenizer.from_pretrained(args.guesser_model)

    from dataloader.guesser_dataloader import SpotDiffDataset4Guesser, collate_batch_guesser
    eval_dataset = SpotDiffDataset4Guesser(guesser_tokenizer, args.eval_file, args, id_list=id_list)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=collate_batch_guesser,
        num_workers=args.num_workers,
    )
    
    # image feature dataloder
    from dataloader.loader_utils import ImageFeaturesHdfReader
    img_reader = ImageFeaturesHdfReader(args.img_feat_file)
    if args.q_use_golden_feat or args.a_use_golden_feat:
        golden_img_reader = ImageFeaturesHdfReader(args.golden_img_feat_file)
    q_img_reader = golden_img_reader if args.q_use_golden_feat else img_reader
    a_img_reader = golden_img_reader if args.a_use_golden_feat else img_reader

    dialogs = []
    acc_total = 0
    total = 0
    for i, batch in enumerate(eval_dataloader):
        img1, img2 = batch['img1'], batch['img2'] # tensor
        img1_info = q_img_reader.get_batch_img_info(img1.tolist())
        img2_info = a_img_reader.get_batch_img_info(img2.tolist())
        # q_feat, q_boxes, q_visual_attention_mask = img1_info['features'].to(args.device), img1_info['rel_boxes'].to(args.device), img1_info['visual_attention_mask'].to(args.device)
        q_feat = torch.cat([img1_info['features'], img1_info['rel_boxes']], dim=-1).to(args.device)
        a_feat, a_boxes, a_visual_attention_mask = img2_info['features'].to(args.device), img2_info['rel_boxes'].to(args.device), img2_info['visual_attention_mask'].to(args.device)

        gt_questions = batch['questions']
        gt_answers = batch['answers']
        batch_size = q_feat.size(0)
        questions = [[] for b in range(batch_size)]
        answers = [[] for b in range(batch_size)]
        actions = [[] for b in range(batch_size)]
        guess_every_round = [[] for b in range(batch_size)]

        for t in range(args.max_turn):

            if args.mode in ['vqg_vqa_guesser']:
                # use predicted questions
                q_input_ids, q_attention_mask, q_token_type_ids = get_vqg_input(questions, answers, vqg_tokenizer, args.max_hist_len)
                q_input_ids = q_input_ids.to(args.device)
                q_attention_mask = q_attention_mask.to(args.device)
                q_token_type_ids = q_token_type_ids.to(args.device)

                with torch.no_grad():
                    output_sequences = vqg.generate(
                            input_ids=q_input_ids,
                            token_type_ids=q_token_type_ids,
                            vis_feats=q_feat,
                            max_length=args.generate_length+q_input_ids.size(1),
                            top_k=args.top_k,
                            top_p=args.top_p,
                            repetition_penalty=args.repetition_penalty,
                            do_sample=True,
                            num_return_sequences=args.num_return_sequences,
                        )

                for b in range(batch_size):
                    x = output_sequences[b].tolist()
                    if args.reverse_target:
                        x.reverse()
                    ques = vqg_tokenizer.decode(output_sequences[b, q_input_ids.size(1):].tolist(), skip_special_tokens=True).encode('ascii', 'ignore').decode('ascii')
                    
                    if args.generate_action:
                        splits = ques.split('$')
                        if len(splits) < 2:
                            print(ques)
                        act = splits[0].strip()
                        ques = splits[1].strip()
                        actions[b].append(act)

                    if args.fix_question:
                        ques = F.fix_question(ques, img1[b].item())['ques_fix']

                    if args.with_action_cls:
                        act = label2act[vqg.pred_action[b].item()]
                        actions[b].append(act)

                    questions[b].append(ques)
            else:
                for b in range(batch_size):
                    turn = len(gt_questions[b])
                    if t < turn:
                        questions[b].append(gt_questions[b][t])

            if args.mode in ['vqg_vqa_guesser', 'golden_vqa_guesser']:
                # use predicted answers
                a_sent = get_vqa_input(questions, answers)
                with torch.no_grad():
                    logits = vqa(a_feat, a_boxes, a_sent, visual_attention_mask=a_visual_attention_mask)
                max_ind = logits.max(dim=1)[1]
                for b in range(batch_size):
                    ans = label2ans[max_ind[b].item()]
                    answers[b].append(ans)
            else:
                for b in range(batch_size):
                    turn = len(gt_questions[b])
                    if t < turn:
                        answers[b].append(gt_answers[b][t])

            if args.guess_every_round or t+1==args.max_turn:
                g_input_ids, g_attention_mask = get_guesser_input(questions, answers, guesser_tokenizer, args.max_hist_len)
                g_input_ids = g_input_ids.to(args.device)
                g_attention_mask = g_attention_mask.to(args.device)
                candidate_ids = batch['candidate_ids'].to(args.device)
                boxes = batch['boxes'].to(args.device)
                candidate_mask = batch['candidate_mask'].to(args.device)
                labels = batch['labels'].to(args.device)
                with torch.no_grad():
                    outputs = guesser(input_ids=g_input_ids, attention_mask=g_attention_mask, candidate_ids=candidate_ids, boxes=boxes, candidate_mask=candidate_mask, labels=labels)
                prediction = outputs[1]
                for b in range(batch_size):
                    guess_every_round[b].append(prediction[b].item())

        acc_total += (labels==prediction).float().sum().item()
        total += labels.size(0)
        if i%args.verbose_steps==0:
            print('===================================')
            print('Success: %.1f', (labels==prediction)[0].item())
            print(batch['img1'][0], batch['img2'][0])
            turn = len(questions[0])
            for t in range(turn):
                ques = questions[0][t]
                ans = answers[0][t]
                if args.include_action or args.with_action_cls:
                    act = actions[0][t]
                    print('Q_%d' %t, ques.encode('ascii', 'ignore').decode('ascii'), 'A_%d' %t, ans.encode('ascii', 'ignore').decode('ascii'), 'Act_%d' %t, act.encode('ascii', 'ignore').decode('ascii'))
                else:
                    print('Q_%d' %t, ques.encode('ascii', 'ignore').decode('ascii'), 'A_%d' %t, ans.encode('ascii', 'ignore').decode('ascii'))
            print('total: %d' % total)
            print('correct: %d' % acc_total)
            print('Accuracy: %.4f' % (acc_total / total))

        for b in range(batch_size):
            dialog = {
                'img1': img1[b].item(),
                'img2': img2[b].item(),
                'questions': questions[b],
                'answers': answers[b],
                'prediction': prediction[b].item(),
                'target': labels[b].item(),
            }
            if args.guess_every_round:
                dialog['guess_every_round'] = guess_every_round[b]
            dialogs.append(dialog)

    return dialogs



if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # vqa model
    parser.add_argument('--max_vqa_len', default=50, type=int)
    parser.add_argument("--fromScratch", dest='from_scratch', action='store_const', default=False, const=True,
                    help='If none of the --load, --loadLXMERT, --loadLXMERTQA is set, '
                            'the model would be trained from scratch. If --fromScratch is'
                            ' not specified, the model would load BERT-pre-trained weights by'
                            ' default. ')

    # vqg_model
    parser.add_argument('--max_hist_len', default=512, type=int)
    parser.add_argument('--max_target_len', default=30, type=int)

    # lxmert
    parser.add_argument('--llayers', default=9, type=int)
    parser.add_argument('--xlayers', default=5, type=int)
    parser.add_argument('--rlayers', default=5, type=int)

    # model path
    parser.add_argument('--vqa_model', default='', type=str)
    parser.add_argument('--vqg_model', default='', type=str)
    parser.add_argument('--bert_model', default='', type=str)
    parser.add_argument('--gpt_model', default='', type=str)
    parser.add_argument('--guesser_model', default='', type=str)

    # dataset
    parser.add_argument('--eval_file', default='', type=str)
    parser.add_argument('--img_feat_file', default='', type=str)
    parser.add_argument('--golden_img_feat_file', default='', type=str)
    parser.add_argument('--dataroot', default='', type=str)


    # eval options
    parser.add_argument('--eval_batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--max_turn', default=5, type=int)
    parser.add_argument('--reverse_target', default=0, type=int)

    # path
    parser.add_argument('--lxrt_path', default='lxmert/src', type=str)
    parser.add_argument('--butd_path', default='bottom-up-attention-vqa', type=str)

    # generation
    parser.add_argument(
        '--generate_length',
        type=int,
        default=30,
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="primarily useful for CTRL model; in that case, use 1.2",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="The number of samples to generate.",
    )
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)

    # evaluate options
    parser.add_argument('--verbose_steps', default=20, type=int)
    parser.add_argument('--use_cuda', default=1, type=int)
    parser.add_argument('--mode', default='vqg_vqa_guesser', type=str, choices=['vqg_vqa_guesser', 'golden_vqa_guesser', 'golden_golden_guesser'])
    parser.add_argument('--q_use_golden_feat', default=0, type=int)
    parser.add_argument('--a_use_golden_feat', default=0, type=int)
    parser.add_argument('--guess_every_round', default=0, type=int)

    # write options
    parser.add_argument('--do_write', default=0, type=int)
    parser.add_argument('--output_file', default='', type=str)

    # fix question
    parser.add_argument('--fix_question', default=0, type=int)
    parser.add_argument('--num_cpus', default=8, type=int)
    parser.add_argument('--repeat_time', default=1, type=int)


    # action
    parser.add_argument('--generate_action', default=0, type=int)
    parser.add_argument('--include_action', default=0, type=int)

    parser.add_argument('--with_action_cls', default=0, type=int)

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    print(args)

    with open(args.eval_file) as f:
        d = json.load(f)
        data_num = len(d)

    num_gpus = torch.cuda.device_count()
    total_list = list(range(data_num))
    id_lists = [total_list[i::num_gpus] for i in range(num_gpus)]
    
    if args.num_cpus != 0:
        ray.init(num_cpus=args.num_cpus)
    else:
        ray.init()
    
    dialogs = []
    for i in range(args.repeat_time):
        gen_list = []
        for i in range(num_gpus):
            gen_list.append(solve.remote(args, id_lists[i]))
        for i in range(num_gpus):
            dialogs.extend(ray.get(gen_list[i]))
    
    correct = 0
    for dialog in dialogs:
        if dialog['target']==dialog['prediction']:
            correct += 1
    print('[Total: %d][Correct: %d][Acc: %.4f]' % (len(dialogs), correct, correct / len(dialogs) ))


    
    if args.do_write:
        with open(args.output_file, 'w') as f:
            import json
            json.dump(dialogs, f)
