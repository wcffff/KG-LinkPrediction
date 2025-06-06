from turtle import forward
import dgl
import dgl.function as fn
# from regex import P
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import ccorr
import numpy as np
from models import CompGraphConv, CompGCN

class MH_attention(nn.Module):
    def __init__(
        self, 
        emb_dim,
        num_heads,  
        d_k,
        d_v,
        dropout=0.2,
        ):
        super().__init__()
        
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        
        self.Wq = nn.Linear(self.emb_dim, self.num_heads * self.d_k, bias=False)
        self.Wk = nn.Linear(self.emb_dim, self.num_heads * self.d_k, bias=False)
        self.Wv = nn.Linear(self.emb_dim, self.num_heads * self.d_v, bias=False)
        
        nn.init.normal_(self.Wq.weight, mean=0, std=np.sqrt(2.0 / (self.emb_dim + self.d_k)))
        nn.init.normal_(self.Wk.weight, mean=0, std=np.sqrt(2.0 / (self.emb_dim + self.d_k)))
        nn.init.normal_(self.Wv.weight, mean=0, std=np.sqrt(2.0 / (self.emb_dim + self.d_v)))
        
        self.dropout = nn.Dropout(dropout)   
        self.dropout1 = nn.Dropout(0.2)   
        self.layernorm = nn.LayerNorm(self.emb_dim)
        self.softmax = nn.Softmax(dim=2)
        
        self.fc = nn.Linear(self.num_heads*self.d_v, self.emb_dim, bias=False)
        nn.init.xavier_normal_(self.fc.weight)
        
    def forward(self, x):
        # sub_emb:[batch_size, embed_dim]
        # rel_emb:[batch_size, embed_dim]
        # x:[batch_size, 2, embed_dim]
        batch_size = x.size(0)  
        sequence_length = x.size(1)
        residual = x
        
        # Linear transformation
        query = self.Wq(x)  
        key = self.Wk(x)  
        value = self.Wv(x) 
        
        # Reshape for multi-head attention
        query = query.view(batch_size, sequence_length, self.num_heads, self.d_k)
        key = key.view(batch_size, sequence_length, self.num_heads, self.d_k)
        value = value.view(batch_size, sequence_length, self.num_heads, self.d_v)
        
        # Transpose dimensions for matrix multiplication
        query = query.transpose(1, 2).contiguous().view(-1, sequence_length, self.d_k)
        key = key.transpose(1, 2).contiguous().view(-1, sequence_length, self.d_k)
        value = value.transpose(1, 2).contiguous().view(-1, sequence_length, self.d_v)
        
        # Scaled dot-product attention -> scores:[batch_size, num_heads, sequence_length, sequence_length]
        scores = th.bmm(query, key.transpose(1, 2)) / th.sqrt(th.tensor(self.d_k, dtype=th.float32))
        
        attention_weights = self.softmax(scores)  # [batch_size, num_heads, sequence_length, sequence_length]
        attention_output = self.dropout1(attention_weights)
        
        attention_output = th.bmm(attention_weights, value)  # [batch_size, num_heads, sequence_length, dk]
        
        attention_output = attention_output.view(batch_size, self.num_heads, sequence_length, self.d_v)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)  # [batch_size, sequence_length, embed_dim]
        attention_output = self.dropout(self.fc(attention_output))  # [batch_size, sequence_length, embed_dim]
        attention_output = self.layernorm(attention_output + residual)
        # print(attention_output.shape)
        
        return attention_output
    
class FeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.3):
        super().__init__()
        self.w1 = nn.Conv1d(d_in, d_hid, 1)
        self.w2 = nn.Conv1d(d_hid, d_in, 1)
        
        nn.init.kaiming_normal_(self.w1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.w2.weight, mode="fan_in", nonlinearity="relu")
        
        self.layernorm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layernorm(output + residual)  
        
        return output

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, d_k, d_v, hid_dim):
        super().__init__()
        self.MH_attn = MH_attention(embed_dim, num_heads, d_k, d_v)
        self.feedforward = FeedForward(embed_dim, hid_dim)
    
    def forward(self, x):
        x = self.MH_attn(x)
        x = self.feedforward(x)
        return x

class Attention_Layer(nn.Module):
    
    def __init__(
        self, 
        num_bases,
        num_rel,
        num_ent,
        in_dim,
        layer_size,
        comp_fn="sub",
        batchnorm=True,
        dropout=0.1,
        layer_dropout=[0.3],
        num_filt=200,
        hid_drop=0.3,
        feat_drop=0.3,
        ker_sz=5,
        k_w=5,
        k_h=5,
        num_heads=64,
        d_k=50,
        d_v=32
    ):
        super(Attention_Layer, self).__init__()
        
        self.embed_dim = layer_size[-1]
        self.hid_drop = hid_drop
        self.feat_drop = feat_drop
        self.padding = 0
        
        # self.gnn_encoder = CompGCN(num_bases,num_rel,num_ent,in_dim,layer_size,comp_fn,batchnorm,dropout,layer_dropout)
        self.n_feats = nn.Parameter(th.Tensor(num_ent, in_dim))
        self.r_feats = nn.Parameter(th.Tensor(num_rel, in_dim))
        
        nn.init.xavier_normal_(self.n_feats)
        nn.init.xavier_normal_(self.r_feats)
        
        self.attn_encoder = EncoderLayer(in_dim, num_heads, d_k, d_v, hid_dim=400)
        self.attn_encoder1 = EncoderLayer(in_dim, num_heads, d_k, d_v, hid_dim=400)
        
        self.bn_ent = nn.BatchNorm1d(in_dim)
        self.bn_rel = nn.BatchNorm1d(in_dim)
        self.bias = nn.Parameter(th.zeros(num_ent))
        
    def forward(self, graph, sub, rel):
        sub_emb = self.n_feats[sub, :]
        rel_emb = self.r_feats[rel, :]
        
        sub_emb = self.bn_ent(sub_emb)
        rel_emb = self.bn_rel(rel_emb)
        
        x = th.stack([sub_emb, rel_emb], dim=1)
        batch_size = sub_emb.size(0)
        
        x = self.attn_encoder(x)
        x = self.attn_encoder1(x)
        
        src_edges = x[:, 1, :]
        scores = th.mm(src_edges, self.n_feats.transpose(0, 1))
        scores += self.bias.expand_as(scores)
        scores = th.sigmoid(scores)
        
        return scores               

# if __name__ == '__main__':
#     # x->[batch_sz, 2, embed_dim]
#     x = th.rand(64, 3, 200)
#     # print(x)
#     model = EncoderLayer(200, 20, 512)
#     out = model(x)
#     print(out)
    
#     x1 = th.rand(64, 200)
#     x2 = th.rand(64, 200)
#     x3 = th.stack([x1, x2], dim=1)
#     print(x3.shape)




    