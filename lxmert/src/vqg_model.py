# coding=utf-8
# Copyleft 2019 project LXRT.
import os
from argparse import Namespace
from json import decoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig

from lxrt.modeling import LXRTFeatureExtraction, VISUAL_CONFIG

from transformers import (
    PreTrainedModel,
    AutoModelForCausalLM,
    PretrainedConfig,
    AutoConfig,
)
from transformers.modeling_outputs import (
    Seq2SeqLMOutput,
    BaseModelOutput,
)

def set_visual_config(args):
    VISUAL_CONFIG.l_layers = args.llayers
    VISUAL_CONFIG.x_layers = args.xlayers
    VISUAL_CONFIG.r_layers = args.rlayers

"""
load lxmert encoder
"""
def load_model(model, path):
    # Load state_dict from snapshot file
    print("Load LXMERT pre-trained model from %s" % path)
    state_dict = torch.load("%s_LXRT.pth" % path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_state_dict[key[len("module."):]] = value
        else:
            new_state_dict[key] = value
    state_dict = new_state_dict

    # Print out the differences of pre-trained and model weights.
    load_keys = set(state_dict.keys())
    model_keys = set(model.state_dict().keys())
    print()
    print("Weights in loaded but not in model:")
    for key in sorted(load_keys.difference(model_keys)):
        print(key)
    print()
    print("Weights in model but not in loaded:")
    for key in sorted(model_keys.difference(load_keys)):
        print(key)
    print()

    # Load weights to model
    model.load_state_dict(state_dict, strict=False)


class LXRTLMConfig(PretrainedConfig):
    model_type = 'encoder-decoder'
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert (
            'decoder' in kwargs
        ), 'Config has to be initialized wit hdecoder config'
        decoder_config = kwargs.pop('decoder')
        decoder_model_type = decoder_config.pop('model_type')

        self.decoder = AutoConfig.for_model(decoder_model_type, **decoder_config)
        self.is_encoder_decoder = True

    @classmethod
    def from_encoder_decoder_configs(self, decoder_config, **kwargs):

        # logger.info("Set `config.is_decoder=True` and `config.add_cross_attention=True` for decoder_config")
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True

        return self(decoder=decoder_config.to_dict(), **kwargs)

    def to_dict(self):
        import copy
        output = copy.deepcopy(self.__dict__)
        output['decoder'] = self.decoder.to_dict()
        output['model_type'] = self.__class__.model_type
        return output

# for question type classification
class QuestionTypeCls(nn.Module):
    def __init__(self, question_type_cls_num):
        super().__init__()
        print('use question type cls...')
        self.cls = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, question_type_cls_num)
        )
        self.question_type_ce = nn.CrossEntropyLoss(reduction='none')
    def forward(self, pooler, mask, target):
        """
        pooler: [b, hidden_size]
        mask: [b]
        target: [b]
        """
        logits = self.cls(pooler)
        loss = self.question_type_ce(logits, target)
        loss = (loss * mask).sum() / (mask.sum() + 1e-10)
        pred = logits.max(dim=-1)[1]
        return loss, pred


# for object classification
class ObjectCls(nn.Module):
    def __init__(self, object_cls_num):
        super().__init__()
        print('use object cls...')
        self.object_emb = nn.Embedding(object_cls_num, 512)
        self.object_proj = nn.Sequential(
            nn.Linear(512+4, 768),
            nn.Tanh()
        )
        self.pooler_proj = nn.Sequential(
            nn.Linear(768, 768),
            nn.Tanh()
        )
        self.object_bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, pooler, gt_objects_id, gt_boxes, target, mask):
        x = torch.cat([self.object_emb(gt_objects_id), gt_boxes], dim=2)  # (batch_size, max_instance_num, 516)
        x = self.object_proj(x)  # (batch_size, max_instance_num, 768)
        y = self.pooler_proj(pooler).unsqueeze(1)  # (batch_size, 1, 768)

        output = (x * y).sum(-1)  # (batch_size, max_instance_num)

        loss = self.object_bce(output, target.float()) # (b, c)
        loss = (loss * mask).sum() / (mask.sum() + 1e-10)
        pred = torch.sigmoid(output)
        return loss, pred

class DecoderObjectCls(nn.Module):
    def __init__(self, object_cls_num):
        super().__init__()
        print('use object cls...')
        self.object_emb = nn.Embedding(object_cls_num, 512)
        self.object_proj = nn.Sequential(
            nn.Linear(512+4, 768),
            nn.Tanh()
        )
        self.pooler_proj = nn.Sequential(
            nn.Linear(768, 768),
            nn.Tanh()
        )
        self.object_bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, decoder_hiddens, decoder_mask, gt_objects_id, gt_boxes, target, target_mask):
        x = torch.cat([self.object_emb(gt_objects_id), gt_boxes], dim=2)  # (batch_size, max_instance_num, 516)
        x = self.object_proj(x)  # (batch_size, max_instance_num, 768)
        y = self.pooler_proj(decoder_hiddens)  # (batch_size, decoder_len, 768)

        x = x.unsqueeze(1)
        y = y.unsqueeze(2)
        output = (x * y).sum(-1)  # (batch_size, decoder_seq_len, max_instance_num)
        
        decoder_seq_len = decoder_hiddens.size(1)
        target = target.unsqueeze(1).repeat(1, decoder_seq_len, 1)  # (batch_size, decoder_seq_len, max_instance_num)
        loss = self.object_bce(output, target.float())
        mask = decoder_mask.unsqueeze(2) * target_mask.unsqueeze(1)
        loss = (loss * mask).sum() / (mask.sum() + 1e-10)
        
        pred = output.max(dim=-1)[1]
        return loss, pred

# for decoder question type classification
class DecoderQuestionTypeCls(nn.Module):
    def __init__(self, question_type_cls_num):
        super().__init__()
        print('use question type cls...')
        self.question_type_cls_num = question_type_cls_num
        self.cls = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, question_type_cls_num)
        )
        self.question_type_ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, decoder_hiddens, decoder_mask, target, target_mask):
        """
        decoder_hiddens: [b, max_len, d]
        decoder_mask: [b, max_len]
        target: [b]
        target_mask: [b]
        """
        x = self.cls(decoder_hiddens)  # (b, max_len, question_type_cls_num)
        decoder_seq_len  = x.size(1)
        y = target.unsqueeze(1).repeat(1, decoder_seq_len)  # (b, max_len)
        loss = self.question_type_ce(x.view(-1, self.question_type_cls_num), y.view(-1))  # (b * max_len)
        loss = loss.view(-1, decoder_seq_len)
        mask = decoder_mask * target_mask.unsqueeze(1)  # (b, max_len)
        loss = (loss * mask).sum() / (mask.sum() + 1e-10)

        return (loss, )

# for spot diff classification
class SpotDiffCls(nn.Module):
    def __init__(self):
        super().__init__()
        print('use spot diff cls...')
        self.cls = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 2)
        )
        self.spot_diff_ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, pooler, mask, target):
        logits = self.cls(pooler)  # (b, 2)
        loss = self.spot_diff_ce(logits, target)
        loss = (loss * mask).sum() / (mask.sum() + 1e-10)
        pred = logits.max(dim=1)[1]
        return loss, pred

class VQGModel(PreTrainedModel):
    config_class = LXRTLMConfig
    def __init__(self, config, args, encoder=None, decoder=None):
        super().__init__(config)

        assert config.is_encoder_decoder, 'Model must be encoder-decoder model!!!'
        assert config.decoder.is_decoder, 'config.decoder is_decoder must be True!!!'
        assert config.decoder.add_cross_attention, 'config.decoder.add_cross_attention must be True!!!'

        set_visual_config(args)        
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = LXRTFeatureExtraction.from_pretrained(
                args.bert_model,
                mode='lxr'
            )

        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder =  AutoModelForCausalLM.from_config(config.decoder)

        self.decoder.config = self.config.decoder
        
        if hasattr(config, 'with_question_type_cls') and config.with_question_type_cls:
            self.question_type_cls = QuestionTypeCls(config.question_type_cls_num)
        
        if hasattr(config, 'with_object_cls') and config.with_object_cls:
            self.object_cls = ObjectCls(config.object_cls_num)

        if hasattr(config, 'with_spot_diff_cls') and config.with_spot_diff_cls:
            self.spot_diff_cls = SpotDiffCls()
        
        if hasattr(config, 'with_decoder_object_cls') and config.with_decoder_object_cls:
            self.decoder_object_cls = DecoderObjectCls(config.decoder_object_cls_num)
        
        if hasattr(config, 'with_decoder_question_type_cls') and config.with_decoder_question_type_cls:
            self.decoder_question_type_cls = DecoderQuestionTypeCls(config.decoder_question_type_cls_num)
        
        if hasattr(config, 'with_action_cls') and config.with_action_cls:
            self.action_cls = QuestionTypeCls(config.action_cls_num)
            self.action_embedding = nn.Embedding(self.config.action_cls_num, self.config.decoder.hidden_size)

    @classmethod
    def from_encoder_decoder_pretrained(self, args):
        set_visual_config(args)
        encoder = LXRTFeatureExtraction.from_pretrained(
            args.bert_model,
            mode='lxr'
        )
        load_model(encoder, args.lxmert_model)

        if args.decoder_model=='':
            decoder_config = AutoConfig.from_pretrained(args.bert_model)
        else:
            decoder_config = AutoConfig.from_pretrained(args.decoder_model)
        if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
            # logger.info(
            #     f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
            # )
            decoder_config.is_decoder = True
            decoder_config.add_cross_attention = True
        if 'gpt2' in args.decoder_model:
            decoder_config.n_layers = args.n_decoder_layers        
        else:
            decoder_config.num_hidden_layers = args.n_decoder_layers
        if args.decoder_model=='':
            # random initialization
            decoder = AutoModelForCausalLM.from_config(config=decoder_config)
        else:
            print(f'Loading weights from {args.decoder_model}!')
            decoder = AutoModelForCausalLM.from_pretrained(args.decoder_model, config=decoder_config)
                
        config = LXRTLMConfig.from_encoder_decoder_configs(decoder_config)

        # task1
        if args.with_question_type_cls:
            setattr(config, 'with_question_type_cls', args.with_question_type_cls)
            setattr(config, 'question_type_cls_num', args.question_type_cls_num)

        # task2
        if args.with_object_cls:
            setattr(config, 'with_object_cls', args.with_object_cls)
            setattr(config, 'object_cls_num', args.object_cls_num)

        # task3
        if args.with_spot_diff_cls:
            setattr(config, 'with_spot_diff_cls', args.with_spot_diff_cls)

        # task4
        if args.with_decoder_object_cls:
            setattr(config, 'with_decoder_object_cls', args.with_decoder_object_cls)
            setattr(config, 'decoder_object_cls_num', args.decoder_object_cls_num)

        # task5
        if args.with_decoder_question_type_cls:
            setattr(config, 'with_decoder_question_type_cls', args.with_decoder_question_type_cls)
            setattr(config, 'decoder_question_type_cls_num', args.decoder_question_type_cls_num)

        # task6
        if args.with_action_cls:
            setattr(config, 'with_action_cls', args.with_action_cls)
            setattr(config, 'action_cls_num', args.action_cls_num)


        return self(config, args, encoder=encoder, decoder=decoder)
    
    @classmethod
    def from_pretrained(self, pretrained_model_name_or_path, args):
        config = LXRTLMConfig.from_pretrained(pretrained_model_name_or_path)
        model = self(config, args)
        weights_path = os.path.join(pretrained_model_name_or_path, 'pytorch_model.bin')
        state_dict = torch.load(weights_path, map_location='cpu' if not torch.cuda.is_available() else None)
        #Print out the differences of pre-trained and model weights.
        load_keys = set(state_dict.keys())
        model_keys = set(model.state_dict().keys())
        print()
        print("Weights in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        # Load weights to model
        model.load_state_dict(state_dict, strict=False)
        return model
    
    # def get_input_embeddings(self):
    #     return self.encoder.bert.embeddings.word_embeddings

    # def get_output_embeddings(self):
    #     return self.decoder.get_output_embeddings()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    """
    Require:
        input_ids
        lang_attention_mask,
        token_type_ids,
        visual_feats,
        decoder_input_ids,
        labels
    """
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}
        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            token_type_ids = kwargs.get('token_type_ids', None)
            visual_feats = kwargs.get('visual_feats', None)
            visual_attention_mask = kwargs.get('visual_attention_mask', None)
            if visual_attention_mask is None:
                visual_attention_mask = torch.ones([visual_feats[0].size(0), visual_feats[0].size(1)], dtype=attention_mask.dtype, device=attention_mask.device)

            output = self.encoder(input_ids, token_type_ids, attention_mask, visual_feats=visual_feats, visual_attention_mask=visual_attention_mask)
            lang_feats, visn_feats = output[0]
            encoder_hidden_states = torch.cat([visn_feats, lang_feats], dim=1)
            attention_mask = torch.cat([visual_attention_mask, attention_mask], dim=1)
            if hasattr(self.config, 'with_action_cls') and self.config.with_action_cls:
                action_ids = kwargs.get('action_ids', None)  # (b)
                action_emb = self.action_embedding(action_ids).unsqueeze(1)  # (b, 1, d_hidden)
                action_attention_mask = torch.ones([attention_mask.size(0), 1], dtype=attention_mask.dtype, device=attention_mask.device)
                encoder_hidden_states = torch.cat([encoder_hidden_states, action_emb], dim=1)
                attention_mask = torch.cat([attention_mask, action_attention_mask], dim=1)

            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_hidden_states,
                hidden_states=(encoder_hidden_states, output[1]), # last hidden states and pooler
                attentions=None
            )
        
        encoder_hidden_states = encoder_outputs[0]
    
        # print('encoder_hidden_states', encoder_hidden_states.size())
         # Decode, don't pass labels!!!
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )
        # print('decoder_outputs_logits', decoder_outputs.logits.size())

        if labels is not None:
            flatten_logits = decoder_outputs.logits.contiguous().view(-1, self.decoder.config.vocab_size)
            flatten_labels = labels.contiguous().view(-1)
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(flatten_logits, flatten_labels)
            decoder_outputs.loss = lm_loss

        # print('loss', decoder_outputs.loss)
        if not return_dict:
            return decoder_outputs + encoder_outputs

        out = Seq2SeqLMOutput(
            loss=decoder_outputs.loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        return out

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past=past)
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
        }
        return input_dict

    def _prepare_encoder_decoder_kwargs_for_generation(self, input_ids: torch.LongTensor, model_kwargs):
        # retrieve encoder hidden states
        token_type_ids = model_kwargs.get('token_type_ids', None)
        attention_mask = model_kwargs.get('attention_mask', None)
        visual_feats = model_kwargs.get('visual_feats', None)
        visual_attention_mask = model_kwargs.get('visual_attention_mask', None)
        if visual_attention_mask is None:
            visual_attention_mask = torch.ones([visual_feats[0].size(0), visual_feats[0].size(1)], dtype=attention_mask.dtype, device=attention_mask.device)

        output = self.encoder(input_ids, token_type_ids, attention_mask, visual_feats=visual_feats, visual_attention_mask=visual_attention_mask)
        lang_feats, visn_feats = output[0]
        encoder_hidden_states = torch.cat([visn_feats, lang_feats], dim=1)
        attention_mask = torch.cat([visual_attention_mask, attention_mask], dim=1)

        if hasattr(self.config, 'with_action_cls') and self.config.with_action_cls:
            action_logits = self.action_cls.cls(output[1])
            action_ids = action_logits.max(dim=-1)[1]
            # action_prob = F.softmax(action_logits, dim=-1)
            # action_ids = torch.multinomial(action_prob, 1).squeeze(-1) # (b)
            self.pred_action = action_ids
            action_emb = self.action_embedding(action_ids).unsqueeze(1)  # (b, 1, d_hidden)
            action_attention_mask = torch.ones([attention_mask.size(0), 1], dtype=attention_mask.dtype, device=attention_mask.device)
            encoder_hidden_states = torch.cat([encoder_hidden_states, action_emb], dim=1)
            attention_mask = torch.cat([attention_mask, action_attention_mask], dim=1)

        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_hidden_states,
            hidden_states=(encoder_hidden_states, output[1]),
            attentions=None
        )
        model_kwargs['attention_mask'] = attention_mask
        model_kwargs['encoder_outputs'] = encoder_outputs

           
        return model_kwargs


    def _reorder_cache(self, past, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past, beam_idx)


