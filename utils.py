import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BertTokenizer,
) 

def load_model_tokenizer(model_name_or_path, mode, model_type, device, state_dict_path=None):
    if mode in ['guesser']:
        from model.guesser import BertGuesser

        config = AutoConfig.from_pretrained(model_name_or_path)
        config.category_num = 300
        config.cls_emb_size = 512
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = BertGuesser.from_pretrained(
                model_name_or_path,
                from_tf=bool('.ckpt' in model_name_or_path),
                config=config,
            )
        model.resize_token_embeddings(len(tokenizer))
    elif mode in ['questioner', 'answerer']:
        if model_type in ['gpt2']:
            from model.qgen import GPT2Generator

            config = AutoConfig.from_pretrained(model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            special_tokens_dict = {'bos_token': '[BOS]', 'eos_token': '[EOS]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'sep_token': '[SEP]', 'unk_token': '[UNK]'}
            tokenizer.add_special_tokens(special_tokens_dict)
            tokens_dict = ['[image]', '[context]', '[target]']
            tokenizer.add_tokens(tokens_dict)
            config.vis_embed = 2048 + 4
            config.input_drop = 0.3
            config.target_token_id = tokenizer.convert_tokens_to_ids('[target]')
            config.bos_token_id = tokenizer.bos_token_id
            config.eos_token_id = tokenizer.eos_token_id
            config.pad_token_id = tokenizer.pad_token_id
            model = GPT2Generator.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=config,
            )
            model.resize_token_embeddings(len(tokenizer))
    else:
        raise ValueError(
            'Mode should be in ["questioner", "answerer", "guesser"]!!!'
        )
    model = model.to(device)
    model = model.module if hasattr(model, 'module') else model
    return model, tokenizer

def load_model_tokenizer_from_pretrained(model_name_or_path, mode, model_type, device, state_dict_path=None):
    if mode in ['guesser']:
        from model.guesser import BertGuesser
        model = BertGuesser.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    elif mode in ['questioner', 'answerer']:
        if model_type=='vd_bert':
            from model.qgen_vd_bert import VDLMHeadModel
            model = VDLMHeadModel.from_pretrained(model_name_or_path)
            tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        elif model_type=='gpt2':
            from model.qgen import GPT2Generator
            model = GPT2Generator.from_pretrained(model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = model.to(device)
    model = model.module if hasattr(model, 'module') else model
    return model, tokenizer
