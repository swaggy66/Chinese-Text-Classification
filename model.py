import torch.nn as nn
from transformers import BertModel

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.max_len = config.pad_size
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.class_num)

    def forward(self, x):
        context = x[:,0]
        mask = x[:,1] 
        x = self.bert(input_ids = context, attention_mask = mask)
        # x = x.last_hidden_state[:,0,:]
        x = x.pooler_output
        x = self.fc(x)
        return x
