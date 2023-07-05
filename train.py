#!/usr/bin/env python3

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.

Adapted from:
https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_language_modeling.py
"""


import argparse
import glob
import json
import logging

from model.guesser import BertGuesser
import os
import random
import re
import shutil
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
    BertTokenizer,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

from utils import load_model_tokenizer, load_model_tokenizer_from_pretrained


# set random seed
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

# return sorted checkpoint list
def _sorted_checkpoints(
    args, checkpoint_prefix="checkpoint", use_mtime=False
) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(
        os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix))
    )

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append(
                    (int(regex_match.groups()[0]), path)
                )

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(
        0, len(checkpoints_sorted) - args.save_total_limit
    )
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(
            "Deleting older checkpoint [{}] due to args.save_total_limit".format(
                checkpoint
            )
        )
        shutil.rmtree(checkpoint)


"""
Return dataset and collate_fn
"""
def get_dataset(args, data_path, tokenizer):
    if args.mode in ['guesser']:
        from dataloader.guesser_dataloader import SpotDiffDataset4Guesser, collate_batch_guesser
        dataset = SpotDiffDataset4Guesser(tokenizer, data_path, args)
        collate_fn = collate_batch_guesser
    elif args.mode in ['questioner', 'answerer']:
        if args.model_type in ['gpt2']:
            from dataloader.qgen_dataloader import SpotDiffDataset, collate_batch
            dataset = SpotDiffDataset(tokenizer, data_path, args)
            collate_fn = collate_batch
    else:
        raise ValueError(
            'Mode should be in ["questioner", "answerer", "guesser"]!!!'
        )
    return dataset, collate_fn
    

def train(
    args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer
) -> Tuple[int, float]:
    """Train the model"""
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    # dataloader
    train_dataset, collate_fn = get_dataset(args, args.train_data_file, tokenizer)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=args.num_workers,
    )


    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    model = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model.resize_token_embeddings(len(tokenizer))

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "optimizer.pt"))
        )
        scheduler.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "scheduler.pt"))
        )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (
                len(train_dataloader) // args.gradient_accumulation_steps
            )
            steps_trained_in_current_epoch = global_step % (
                len(train_dataloader) // args.gradient_accumulation_steps
            )

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info(
                "  Will skip the first %d steps in the first epoch",
                steps_trained_in_current_epoch,
            )
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(args.num_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproducibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()            
            # forward pass
            if args.mode in ['questioner', 'answerer']:
                if args.model_type in ['gpt2']:
                    vis_feats = batch['vis_feats'].to(args.device)
                    input_ids = batch['input_ids'].to(args.device)
                    token_type_ids = batch['token_type_ids'].to(args.device)
                    labels = batch['labels'].to(args.device)
                
                for rnd in range(input_ids.size(1)):
                    if args.model_type in ['gpt2']:
                        outputs = model(vis_feats=vis_feats, input_ids=input_ids[:, rnd], token_type_ids=token_type_ids[:, rnd], labels=labels[:, rnd])      
                    loss = outputs[0]
                    if args.n_gpu > 1: 
                        loss = loss.mean()
                    if args.gradient_accumulation_steps:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    tr_loss += loss.item()
            elif args.mode=='guesser':
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                candidate_ids = batch['candidate_ids']
                boxes = batch['boxes']
                candidate_mask = batch['candidate_mask']
                labels = batch['labels']

                outputs  = model(input_ids=input_ids, attention_mask=attention_mask, candidate_ids=candidate_ids, boxes=boxes, candidate_mask=candidate_mask, labels=labels)
                loss = outputs[0]
                if args.n_gpu > 1: 
                    loss = loss.mean()
                if args.gradient_accumulation_steps:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                tr_loss += loss.item()

            # backward propagation
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                "eval_{}".format(key), value, global_step
                            )
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        "loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_step,
                    )
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir, "{}-{}".format(checkpoint_prefix, global_step)
                    )
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt") )
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt") )
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix='') -> Dict:
    eval_output_dir = args.output_dir

    eval_dataset, collate_fn = get_dataset(args, args.eval_data_file, tokenizer)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)
    
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=args.num_workers,
    )

    model = model.module if hasattr(model, "module") else model
    # multi-gpu evalution
    if args.n_gpu>1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    eval_steps_total = 0
    acc_total = 0.0
    model.eval()

    for batch in tqdm(eval_dataloader, desc='Evaluating'):
        if args.mode in ['questioner', 'answerer']:
            if args.model_type in ['gpt2']:
                vis_feats = batch['vis_feats'].to(args.device)
                input_ids = batch['input_ids'].to(args.device)
                token_type_ids = batch['token_type_ids'].to(args.device)
                labels = batch['labels'].to(args.device)
            
            cur_loss = 0

            for rnd in range(input_ids.size(1)):
                if args.model_type in ['gpt2']:
                    with torch.no_grad():
                        outputs = model(vis_feats=vis_feats, input_ids=input_ids[:, rnd], token_type_ids=token_type_ids[:, rnd], labels=labels[:, rnd])

                loss = outputs[0]
                if args.n_gpu>1:
                    loss = loss.mean()
                cur_loss += loss.item()
            eval_loss += cur_loss / input_ids.size(1)

        elif args.mode=='guesser':
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            candidate_ids = batch['candidate_ids'].to(args.device)
            boxes = batch['boxes'].to(args.device)
            candidate_mask = batch['candidate_mask'].to(args.device)
            labels = batch['labels'].to(args.device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, candidate_ids=candidate_ids, boxes=boxes, candidate_mask=candidate_mask, labels=labels)
                prediction = outputs[1]
                acc_total += (labels==prediction).float().mean().item()

        eval_steps_total += 1
    
    if args.mode in ['questioner', 'answerer']:
        eval_loss = eval_loss / eval_steps_total
        ppl = torch.exp(torch.tensor(eval_loss))
        result = {'perplexity': ppl, 'loss': eval_loss}
    else:
        acc = acc_total / eval_steps_total
        result = {'accuracy': acc}

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    # if args.mode in ['questioner', 'answerer']:
    #     model = model.module if hasattr(model, "module") else model
    #     for i, item in enumerate(eval_dataset):
    #         if i>3: break
    #         questions = item['questions']
    #         answers = item['answers']
    #         dialog_turn = item['dialog_turn']
            
    #         vis_feats = item['vis_feats'].unsqueeze(0).to(args.device)
    #         input_ids = tokenizer.convert_tokens_to_ids(['[image]'] * 36 + ['[SEP]'])
    #         token_type_ids = tokenizer.convert_tokens_to_ids(['[image]'] * 37)
    #         print('==================================================')
    #         for rnd in range(dialog_turn):
    #             if args.mode=='questioner':
    #                 cur_input_ids = input_ids + tokenizer.convert_tokens_to_ids(['[BOS]'])
    #                 cur_token_type_ids = token_type_ids + tokenizer.convert_tokens_to_ids(['[target]'])
    #             elif args.mode=='answerer':
    #                 cur_input_ids = input_ids + tokenizer.convert_tokens_to_ids(questions[rnd] + ['[BOS]'])
    #                 cur_token_type_ids = token_type_ids + tokenizer.convert_tokens_to_ids(len(questions[rnd]) * ['[context]'] + ['[target]'])

    #             cur_input_ids = torch.tensor(cur_input_ids).unsqueeze(0).to(args.device)
    #             cur_token_type_ids = torch.tensor(cur_token_type_ids).unsqueeze(0).to(args.device)
    #             with torch.no_grad():
    #                 output_sequences = model.generate(
    #                     input_ids=cur_input_ids,
    #                     token_type_ids=cur_token_type_ids,
    #                     vis_feats=vis_feats,
    #                     max_length=args.generate_length+cur_input_ids.size(1),
    #                     top_k=args.top_k,
    #                     top_p=args.top_p,
    #                     repetition_penalty=args.repetition_penalty,
    #                     do_sample=True,
    #                     num_return_sequences=args.num_return_sequences,

    #                 )
    #             generated_sequence = output_sequences[0]
    #             if args.mode=='questioner':
    #                 print('Q_%d' %rnd, tokenizer.decode(generated_sequence[cur_input_ids.size(1):], skip_special_tokens=True).encode('ascii', 'ignore').decode('ascii') )
    #             elif args.mode=='answerer':
    #                 print('Q_%d' %rnd, tokenizer.convert_tokens_to_string(questions[rnd]), 'A_%d' %rnd, tokenizer.decode(generated_sequence[cur_input_ids.size(1):], skip_special_tokens=True).encode('ascii', 'ignore').decode('ascii') )


    #             fact_tokens = questions[rnd] + answers[rnd] + ['[SEP]']
    #             input_ids.extend(tokenizer.convert_tokens_to_ids(fact_tokens) )
    #             token_type_ids.extend(tokenizer.convert_tokens_to_ids(len(fact_tokens) * ['[context]']))

    return result


def main():
    parser = argparse.ArgumentParser()
    # generation
    parser.add_argument(
        '--generate_length',
        type=int,
        default=25,
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

    # train
    parser.add_argument(
        '--mode',
        default='questioner',
        type=str,
        choices=['questioner', 'answerer', 'guesser']
    )
    parser.add_argument(
        '--num_workers',
        default=4,
        type=int,
        help='how many subprocesses to use for data loading. '
    )

    # Required parameters
    # file path
    parser.add_argument(
        "--train_data_file",
        default='data/spot_diff_train.json',
        type=str,
        required=True,
        help="The input training data file (a text file).",
    )
    parser.add_argument(
        "--eval_data_file",
        default='data/spot_diff_val.json',
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--img_feat_file",
        default='data/img_feat.h5',
        type=str,
        help='The image feature file.',
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="The model architecture to be trained or fine-tuned.",
        default='bert',
        choices=['bert', 'gpt2']
    )

    # Other parameters
    parser.add_argument(
        "--line_by_line",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--should_continue",
        action="store_true",
        help="Whether to continue from latest checkpoint in output_dir",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )
    parser.add_argument(
        "--state_dict_path",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--add_special_tokens",
        default=None,
        type=str,
        help="Optional file containing a JSON dictionary of special tokens that should be added to the tokenizer.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=1.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )

    parser.add_argument(
        "--logging_steps", type=int, default=500, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank, -1 means no distributed training",
    )
    parser.add_argument(
        "--server_ip", type=str, default="", help="For distant debugging."
    )
    parser.add_argument(
        "--server_port", type=str, default="", help="For distant debugging."
    )
    args = parser.parse_args()

    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError(
                "Used --should_continue but no checkpoint was found in --output_dir."
            )
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
        and not args.should_continue
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True
        )
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    if args.should_continue:
        model, tokenizer = load_model_tokenizer_from_pretrained(args.model_name_or_path, args.mode, args.model_type, args.device)
    else:
        model, tokenizer = load_model_tokenizer(args.model_name_or_path, args.mode, args.model_type, args.device, state_dict_path=args.state_dict_path)

    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        checkpoint_prefix = "checkpoint"
        # Save model checkpoint
        output_dir = os.path.join(
            args.output_dir, "{}-{}".format(checkpoint_prefix, global_step)
        )
        os.makedirs(output_dir, exist_ok=True)

        # Create output directory if needed
        if args.local_rank in [-1, 0]:
            os.makedirs(output_dir, exist_ok=True)

        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        # model = GPT2Generator.from_pretrained(output_dir) if args.mode in ['questioner', 'answerer'] else BertGuesser.from_pretrained(output_dir)
        # tokenizer = AutoTokenizer.from_pretrained(output_dir)
        model, _ = load_model_tokenizer_from_pretrained(output_dir, args.mode, args.model_type, args.device)
        model.to(args.device)

    evaluate(args, model, tokenizer)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = [
                os.path.dirname(c)
                for c in sorted(
                    glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)
                )
            ]
            logging.getLogger("transformers.modeling_utils").setLevel(
                logging.WARN
            )  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = (
                checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            )
            # model = GPT2Generator.from_pretrained(checkpoint) if args.mode in ['questioner', 'answerer'] else BertGuesser.from_pretrained(checkpoint)
            model, _ = load_model_tokenizer_from_pretrained(checkpoint, args.mode, args.model_type, args.device)
            model.to(args.device)

            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = {k + "_{}".format(global_step): v for k, v in result.items()}
            results.update(result)

    return results


if __name__ == "__main__":
    main()
