# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn

# from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU

# Max length including <bos> and <eos>
# MAX_VQA_LENGTH = 20


class VQAModel(nn.Module):
    def __init__(self, args, num_answers, max_vqa_len=50):
        super().__init__()
        
        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=max_vqa_len
        )
        hid_dim = self.lxrt_encoder.dim
        
        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

    def forward(self, feat, pos, sent, visual_attention_mask=None):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        x = self.lxrt_encoder(sent, (feat, pos), visual_attention_mask=visual_attention_mask)
        logit = self.logit_fc(x)

        return logit


