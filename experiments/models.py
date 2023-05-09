import torch
import torch.nn as nn
from transformers import BertModel, LongformerModel, AutoModel

class FD_BASE(nn.Module):
    '''
    single output
    '''
    def __init__(self, model, max_length, fine_tuning=False, ckp_path=''):
        super(FD_BASE, self).__init__()
        self.max_length = max_length
        self.fine_tuning = fine_tuning
        if ckp_path == '':
            if model == 'bert':
                self.model = BertModel.from_pretrained("bert-base-uncased")
            elif model == 'longformer':
                self.model = AutoModel.from_pretrained("allenai/longformer-base-4096")
        else:
            if model == 'bert':
                self.model = BertModel.from_pretrained(ckp_path)
            elif model == 'longformer':
                self.model = AutoModel.from_pretrained(ckp_path)

        self.linear1 = nn.Linear(self.max_length, 5)
        self.linear2 = nn.Linear(768, 2)

    def forward(self, input):
        if self.fine_tuning:
            outputs = self.model(**input)
            output = outputs.last_hidden_state
        else:
            with torch.no_grad():
                outputs = self.model(**input)
                output = outputs.last_hidden_state

        output = self.linear2(output)
        output = self.linear1(output.view(-1,2,self.max_length))

        return output.view(-1,5,2)

class FD_SIN(nn.Module):
    def __init__(self, model, fine_tuning=False, ckp_path=''):
        super(FD_SIN, self).__init__()
        self.fine_tuning = fine_tuning
        if ckp_path == '':
            if model == 'bert':
                self.model = BertModel.from_pretrained("bert-base-uncased")
            elif model == 'longformer':
                self.model = AutoModel.from_pretrained("allenai/longformer-base-4096")
        else:
            if model == 'bert':
                self.model = BertModel.from_pretrained(ckp_path)
            elif model == 'longformer':
                self.model = AutoModel.from_pretrained(ckp_path)

        self.linear = nn.Linear(768, 2)

    def forward(self, input):
        if self.fine_tuning:
            outputs = self.model(**input)
            pooled_output = outputs[1]
        else:
            with torch.no_grad():
                outputs = self.model(**input)
                pooled_output = outputs[1]

        logits = self.linear(pooled_output)

        return logits # B x 2


class SPT_NFD(nn.Module):
    def __init__(self, model='bert', finetuning=False):
        super(SPT_NFD, self).__init__()
        self.finetuning = finetuning
        if model == 'bert':
            self.model = BertModel.from_pretrained("bert-base-uncased")
        self.linear1 = nn.Linear(768, 2)
        self.linear2 = nn.Linear(768, 2)
        self.linear3 = nn.Linear(768, 2)
        self.linear4 = nn.Linear(768, 2)
        self.linear5 = nn.Linear(768, 2)

    def forward(self, input, label):
        if self.finetuning:
            outputs = self.model(**input)
            pooled_output = outputs[1]
        else:
            with torch.no_grad():
                outputs = self.model(**input)
                pooled_output = outputs[1]

        if label == 'ar':
            logits = self.linear1(pooled_output)
        elif label == 'hi':
            logits = self.linear2(pooled_output)
        elif label == 'mo':
            logits = self.linear3(pooled_output)
        elif label == 'co':
            logits = self.linear4(pooled_output)
        elif label == 'ec':
            logits = self.linear5(pooled_output)

        return logits


class sentenceBert(nn.Module):
    def __init__(self, model, finetuning, n_passages, fusion):
        super(sentenceBert, self).__init__()

        self.finetuning = finetuning
        self.n_passages = n_passages
        self.fusion = fusion

        if model == 'bert':
            self.model = BertModel.from_pretrained("bert-base-uncased")

        elif model == 'longformer':
            self.model = AutoModel.from_pretrained("allenai/longformer-base-4096")

        if self.fusion == 'concat':
            self.linear_o = nn.Linear(768*self.n_passages, 2)
        elif self.fusion == 'linear':
            self.linear_f = nn.Linear(self.n_passages, 1)
            self.linear_o = nn.Linear(768, 2)
        elif self.fusion == 'attention':
            self.attention_f = nn.TransformerEncoderLayer(768, 8, batch_first=True)
            self.linear_o = nn.Linear(768, 2)

    def forward(self, input):
        if self.finetuning:
            outputs = self.model(**input)
            pooled_output = outputs[1]
        else:
            with torch.no_grad():
                outputs = self.model(**input)
                pooled_output = outputs[1]

        if self.fusion == 'concat':
            return self.linear_o(pooled_output.view(-1, 768*self.n_passages))

        elif self.fusion == 'linear':
            output = self.linear_f(pooled_output.view(-1, 768, self.n_passages))
            return self.linear_o(output.view(-1, 768))

        elif self.fusion == 'attention':
            output = self.attention_f(pooled_output)
            return self.linear_o(output[:, 0, :])
