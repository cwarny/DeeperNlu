import torch.nn as nn
import math
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_head=None, p=0., bias=True, scale=True):
        '''
        n_heads is the number of heads
        d_head is the dimension of each head
        '''
        super().__init__()
        d_head = d_model//n_heads if d_head is None else d_head
        self.n_heads, self.d_head, self.scale = n_heads, d_head, scale
        self.q_wgt, self.k_wgt, self.v_wgt = [
            nn.Linear(d_model, n_heads*d_head, bias=bias) 
            for o in range(3)
        ]
        self.out = nn.Linear(n_heads*d_head, d_model, bias=bias)
        self.drop_att, self.drop_res = nn.Dropout(p), nn.Dropout(p)
        self.ln = nn.LayerNorm(d_model)
    
    def forward(self, q, kv, mask=None):
        return self.ln(q + self.drop_res(self.out(self._apply_attention(q, kv, mask=mask))))
    
    def create_attn_mat(self, x, layer, bs):
        return layer(x).view(bs, x.size(1), self.n_heads, self.d_head).permute(0,2,1,3)
    
    def _apply_attention(self, q, kv, mask=None):
        bs,seq_len = q.size(0), q.size(1)
        wq,wk,wv = map(lambda o: self.create_attn_mat(*o,bs), zip((q,kv,kv), (self.q_wgt, self.k_wgt, self.v_wgt)))
        attn_score = wq @ wk.transpose(2,3)
        if self.scale: attn_score /= math.sqrt(self.d_head)
        if mask is not None:
            attn_score = attn_score.float().masked_fill(mask, -float('inf')).type_as(attn_score)
        attn_prob = self.drop_att(F.softmax(attn_score, dim=-1))
        attn_vec = attn_prob @ wv
        return attn_vec.permute(0,2,1,3).contiguous().view(bs, seq_len, -1)

class SimpleSelfAttention(nn.Module):
    def __init__(self, attention_size):
        super().__init__()
        self.attention = nn.Parameter(torch.FloatTensor(attention_size, 1))
        torch.nn.init.xavier_normal_(self.attention)
    
    def forward(self, x):
        attention_score = torch.matmul(x, self.attention).squeeze(-1)
        attention_score = F.softmax(attention_score, dim=1).view(x.size(0), x.size(1), 1)
        scored_x = x * attention_score
        condensed_x = torch.sum(scored_x, dim=1)
        return condensed_x