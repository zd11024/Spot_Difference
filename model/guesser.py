import torch
import torch.nn as nn

from transformers import (
    BertPreTrainedModel,
    BertModel,
    AutoConfig,
    AutoTokenizer,
    AutoModel,
)

class BertGuesser(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)

        self.cate_emb = nn.Embedding(config.category_num, config.cls_emb_size)
        self.fuse = nn.Sequential(
                        nn.Linear(config.cls_emb_size+5, config.hidden_size),
                        nn.Tanh()
                    )
        self.init_weights()
    

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, token_type_ids=None, candidate_ids=None, boxes=None, candidate_mask=None, labels=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,    
        )

        sequence_output = outputs[1]  # pooled output

        object_emb = self.fuse(torch.cat([self.cate_emb(candidate_ids), boxes], dim=2))
        logits = (sequence_output.unsqueeze(1) * object_emb).sum(2)  # (batch_size, object_num)
        logits.masked_fill_(~candidate_mask.bool(), float('-inf'))


        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        predictions = logits.max(dim=1)[1]

        return loss, predictions


