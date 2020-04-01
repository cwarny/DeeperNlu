import torch
import torch.nn as nn
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from functools import partial
from .attention import *

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_hidden_layers=0, drop_rate=0):
        super().__init__()

        layers = [nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(drop_rate)] + \
            sum([
                [nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(drop_rate)]
                for _ in range(n_hidden_layers)
            ], []) + \
            [nn.Linear(hidden_size, output_size)]
        
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)

class StackedRecurrent(nn.Sequential):
    def __init__(self, dropout=0, residual=math.inf):
        super().__init__()
        self.residual = residual
        self.dropout = dropout
    
    def forward(self, x, lengths, hidden=None):
        hidden = hidden or tuple([None] * len(self)) # `len(self)` returns the number of layers
        next_hidden = []
        for i,module in enumerate(self._modules.values()):
            # x_packed = pack_padded_sequence(x, lengths, enforce_sorted=False, batch_first=True)
            # output, h = module(x_packed, hidden[i])
            output, h = module(x, hidden[i])
            # output, _ = pad_packed_sequence(output, batch_first=True)
            next_hidden.append(h)
            # x, _ = pad_packed_sequence(x_packed, batch_first=True)
            if self.residual <= i and x.size(-1) == output.size(-1):
                x = output + x
            else:
                x = output
            x = F.dropout(x, self.dropout, self.training)
        return output, tuple(next_hidden)

class RecurrentNet(nn.Module):
    def __init__(self, 
            input_size, 
            hidden_size, 
            mode='LSTM', 
            batch_first=True, 
            dropout=0, 
            bidirectional=False, 
            num_layers=1, 
            residual=math.inf
        ):
        super().__init__()

        self.residual = residual
        
        recurrent_params = dict(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=batch_first, 
            dropout=dropout, 
            bidirectional=bidirectional
        )
        
        if mode == 'LSTM': recurrent_module = nn.LSTM
        elif mode == 'GRU': recurrent_module = nn.GRU
        elif mode == 'RNN': recurrent_module = nn.RNN
        else: raise Exception('Unknown mode: has to be one of [LSTM, GRU, RNN]')
        
        if self.residual < math.inf:
            dropout_rate = recurrent_params.pop('dropout')
            self.rnn = StackedRecurrent(dropout=dropout_rate, residual=self.residual)
            recurrent_params['num_layers'] = 1
            for i in range(num_layers):
                if i > 0: recurrent_params['input_size'] = recurrent_params['hidden_size'] * (2 if bidirectional else 1)
                module = recurrent_module(**recurrent_params)
                self.rnn.add_module(str(i), module)
        else: self.rnn = recurrent_module(**recurrent_params)
    
    def forward(self, x, lengths, apply_softmax=False):
        out, hidden = self.rnn(x, lengths) if self.residual < math.inf else self.rnn(x)
        return out, hidden

class RNNForIcAndNer(nn.Module):
    def __init__(self,
            label_vocab_size,
            intent_vocab_size,
            input_size,
            hidden_size,
            rnn_mode='LSTM',
            rnn_layers=7,
            rnn_residual=3,
            rnn_dropout=0.5,
            batch_first=True,
            bidirectional=True
        ):
        super().__init__()
        
        self.rnn_stack = RecurrentNet(
            input_size=input_size, 
            hidden_size=hidden_size,
            mode=rnn_mode,
            batch_first=batch_first,
            bidirectional=bidirectional,
            num_layers=rnn_layers,
            residual=rnn_residual,
            dropout=rnn_dropout
        )

        self.fc = nn.Linear(hidden_size*(2 if bidirectional else 1), label_vocab_size)
        self.classifier = nn.Linear(hidden_size*(2 if bidirectional else 1), intent_vocab_size)

    def forward(self, inp, lengths, initial_hidden_state=None, apply_softmax=False):
        bs = inp.size(0)
        rnn_output, _ = self.rnn_stack(inp, lengths, initial_hidden_state)
        last_rnn_output = rnn_output[torch.arange(0, bs), lengths-1] # gather last states
        ic_out = self.classifier(last_rnn_output)
        ner_out = self.fc(rnn_output)

        if apply_softmax:
            ner_out = F.softmax(ner_out, dim=-1)
            ic_out = F.softmax(ic_out, dim=-1)
        
        return ic_out, ner_out

class SequenceEncoder(nn.Module):
    def __init__(self, 
            vocab_size, 
            embedding_size, 
            out_channels, 
            padding_idx=1
        ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.cnn = nn.Conv1d(embedding_size, out_channels, 3, stride=1)
    
    def forward(self, x):
        seq_size = x.size(1)
        out = []
        for t in range(seq_size):
            x_t = x[:,t,:] # t-th word for each utterance in the batch
            x_t_embedded = self.emb(x_t)
            x_t_embedded = x_t_embedded.permute(0,2,1) # Conv1d needs the input channels in dim 1 and the sequence length (here the word length) in dim 2: (batch_size, in_channels, word_length)
            cnn_out = self.cnn(x_t_embedded) # (batch_size, out_channels, word_length-2) The -2 is because we didn't pad and we used kernel of size 3 with stride of 1
            pooled = F.max_pool1d(cnn_out, kernel_size=cnn_out.size(2)).squeeze(-1)
            out.append(pooled)
        out = torch.stack(out, dim=1)
        return out
    
class GeneralRelu(nn.Module):
    def __init__(self, leak=None, sub=None, maxv=None):
        super().__init__()
        self.leak, self.sub, self.maxv = leak, sub, maxv
    
    def forward(self, x):
        x = F.leaky_relu(x, self.leak) if self.leak is not None else F.relu(x)
        if self.sub is not None: x.sub_(self.sub)
        if self.maxv is not None: x.clamp_max_(self.maxv)
        return x

class SelfInitializableModule(nn.Module):
    def __init__(self, module, sub=0., **kwargs):
        super().__init__()
        self.module = module
        self.relu = GeneralRelu(sub=sub, **kwargs)
    
    def forward(self, x): return self.relu(self.module(x))
    
    @property
    def bias(self): return -self.relu.sub
    
    @bias.setter
    def bias(self, v): self.relu.sub = -v
    
    @property
    def weight(self): return self.module.weight

class BertForIcAndNer(nn.Module):
    def __init__(self, encoding_size, hidden_size, label_vocab_size, intent_vocab_size, n_hidden_layers=5, bert_model_name='bert-base-uncased', bert_model_cache=None, freeze=True, padding_idx=1):
        super().__init__()
        from transformers import BertModel
        self.encoder = BertModel.from_pretrained(bert_model_name, cache_dir=bert_model_cache)
        if freeze:
            for p in self.encoder.parameters(): p.requires_grad = False
        self.lin_ner = MultiLayerPerceptron(encoding_size, hidden_size, label_vocab_size, n_hidden_layers=n_hidden_layers)
        self.lin_ic = MultiLayerPerceptron(encoding_size, hidden_size, intent_vocab_size, n_hidden_layers=n_hidden_layers)
        self.padding_idx = padding_idx
    
    def forward(self, x, apply_softmax=False):
        text, *_ = x
        encoded = self.encoder(text)[0]
        ner_out = self.lin_ner(encoded[:,1:,:])
        ic_out = self.lin_ic(encoded[:,0,:])
        if apply_softmax: ner_out, ic_out = map(partial(F.softmax, dim=-1), [ner_out, ic_out])
        return ic_out, ner_out

class BertForMLM(nn.Module):
    def __init__(self, bert_model_dir):
        super().__init__()
        from transformers import BertForMaskedLM
        self.mlm = BertForMaskedLM.from_pretrained(bert_model_dir)
    
    def forward(self, x, apply_softmax=False):
        output = self.mlm(x[0])[0]
        if apply_softmax: output = F.softmax(output, dim=-1)
        return output, x[0]
