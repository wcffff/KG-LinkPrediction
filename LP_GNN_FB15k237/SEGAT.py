import dgl
import dgl.function as fn
from torch import dropout
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import ccorr
import numpy as np
from SAttLE import Encoder
from CNN_decoder import CNN_Decoder, InteractE, ConvE, Rel_InteractE
import random


class NodeLayer(nn.Module):
    def __init__(self, in_dim, out_dim, comp_fn="sub", batch_norm=True):
        super().__init__()
        
        self.w = nn.Linear(in_dim, out_dim)
        nn.init.xavier_normal_(self.w.weight)
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # 添加两个新的权重矩阵
        self.w_o = nn.Linear(in_dim, out_dim)
        self.w_i = nn.Linear(in_dim, out_dim)
        
        # 初始化权重
        nn.init.xavier_normal_(self.w_o.weight)
        nn.init.xavier_normal_(self.w_i.weight)
        
        self.activation = nn.Tanh()
        self.bn = th.nn.BatchNorm1d(out_dim)
    
    def in_out_calc(self, g, comp_emb):
        # 根据出边和入边做聚合
        with g.local_scope():
            g.edata['comp_h'] = comp_emb
            in_edges_idx = th.nonzero(
                g.edata["in_edges_mask"], as_tuple=False
            ).squeeze()
            out_edges_idx = th.nonzero(
                g.edata["out_edges_mask"], as_tuple=False
            ).squeeze()

            comp_h_O = self.w_o(comp_emb[out_edges_idx])
            comp_h_I = self.w_i(comp_emb[in_edges_idx])

            new_comp_h = th.zeros(comp_emb.shape[0], self.out_dim).to(
                comp_emb.device
            )
            new_comp_h[out_edges_idx] = comp_h_O
            new_comp_h[in_edges_idx] = comp_h_I

        return new_comp_h
    
    def forward(self, g, ent_emb):
        assert g.number_of_nodes() == ent_emb.shape[0]
        
        with g.local_scope():
            g.ndata['emb'] = ent_emb
            
            # attention
            g.apply_edges(fn.u_dot_v('emb', 'emb', 'attn'))
            g.edata['attn'] = dgl.ops.edge_softmax(g, g.edata['attn'])
            
            # aggregation
            g.apply_edges(lambda edges: {'comp_emb': edges.src['emb']})
            g.edata['comp_emb'] = self.in_out_calc(g, g.edata['comp_emb'])
            g.edata['comp_emb'] = g.edata['comp_emb'] * g.edata['attn']
            g.update_all(fn.copy_e('comp_emb', 'm'), fn.sum('m', 'neigh'))
            neigh_ent_emb = g.ndata['neigh']
            # g.update_all(fn.u_mul_e('emb', 'attn', 'm'), fn.sum('m', 'neigh'))
            # neigh_ent_emb = g.ndata['neigh']
            
            # apply linear transformation
            # neigh_ent_emb = self.w(neigh_ent_emb)
            neigh_ent_emb = self.bn(neigh_ent_emb)
            neigh_ent_emb = self.activation(neigh_ent_emb)
            
        return neigh_ent_emb
   
class EdgeLayer(nn.Module):
    def __init__(self, in_dim, out_dim, comp_fn="sub", batch_norm=True):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.w = nn.Linear(in_dim, out_dim)
        nn.init.xavier_normal_(self.w.weight)
        
        self.w_attn = nn.Linear(in_dim, out_dim)
        self.w_o = nn.Linear(in_dim, out_dim)
        self.w_i = nn.Linear(in_dim, out_dim)
        
        nn.init.xavier_normal_(self.w_attn.weight)
        nn.init.xavier_normal_(self.w_o.weight)
        nn.init.xavier_normal_(self.w_i.weight)
        
        self.activation = nn.Tanh()
        self.bn = th.nn.BatchNorm1d(out_dim)
        
    def in_out_calc(self, g, comp_emb):
        # 根据出边和入边做聚合
        with g.local_scope():
            g.edata['comp_h'] = comp_emb
            in_edges_idx = th.nonzero(
                g.edata["in_edges_mask"], as_tuple=False
            ).squeeze()
            out_edges_idx = th.nonzero(
                g.edata["out_edges_mask"], as_tuple=False
            ).squeeze()

            comp_h_O = self.w_o(comp_emb[out_edges_idx])
            comp_h_I = self.w_i(comp_emb[in_edges_idx])

            new_comp_h = th.zeros(comp_emb.shape[0], self.out_dim).to(
                comp_emb.device
            )
            new_comp_h[out_edges_idx] = comp_h_O
            new_comp_h[in_edges_idx] = comp_h_I

        return new_comp_h
    
    def forward(self, g, ent_emb, rel_emb):
        assert g.number_of_nodes() == ent_emb.shape[0]
        
        with g.local_scope():
            g.ndata['emb'] = ent_emb
            g.edata['emb'] = rel_emb[g.edata['etype']]
            
            # attention
            g.apply_edges(fn.e_dot_v('emb', 'emb', 'attn'))
            g.edata['attn'] = dgl.ops.edge_softmax(g, g.edata['attn'])
            
            # aggregation
            g.edata['emb'] = self.in_out_calc(g, g.edata['emb'])
            g.edata['emb'] = g.edata['emb'] * g.edata['attn']
            g.update_all(fn.copy_e('emb', 'm'), fn.sum('m', 'neigh'))
            
            neigh_ent_emb = g.ndata['neigh']
            
            # apply linear transformation
            neigh_ent_emb = self.bn(neigh_ent_emb)
            neigh_ent_emb = self.activation(neigh_ent_emb)
            
        return neigh_ent_emb    

class CompLayer(nn.Module):
    def __init__(self, in_dim, out_dim, comp_fn="mul", batch_norm=True):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.w = nn.Linear(in_dim, out_dim)
        nn.init.xavier_normal_(self.w.weight)
        
        self.w_attn = nn.Linear(in_dim, out_dim)
        self.w_o = nn.Linear(in_dim, out_dim)
        self.w_i = nn.Linear(in_dim, out_dim)
        
        nn.init.xavier_normal_(self.w_attn.weight)
        nn.init.xavier_normal_(self.w_o.weight)
        nn.init.xavier_normal_(self.w_i.weight)
        
        self.comp_fn = comp_fn
        
        self.activation = nn.Tanh()
        self.attn_act = nn.LeakyReLU(0.2)
        self.bn = th.nn.BatchNorm1d(out_dim)
    
    def in_out_calc(self, g, comp_emb):
        # 根据出边和入边做聚合
        with g.local_scope():
            g.edata['comp_h'] = comp_emb
            in_edges_idx = th.nonzero(
                g.edata["in_edges_mask"], as_tuple=False
            ).squeeze()
            out_edges_idx = th.nonzero(
                g.edata["out_edges_mask"], as_tuple=False
            ).squeeze()

            comp_h_O = self.w_o(comp_emb[out_edges_idx])
            comp_h_I = self.w_i(comp_emb[in_edges_idx])

            new_comp_h = th.zeros(comp_emb.shape[0], self.out_dim).to(
                comp_emb.device
            )
            new_comp_h[out_edges_idx] = comp_h_O
            new_comp_h[in_edges_idx] = comp_h_I

        return new_comp_h
 
    
    def forward(self, g, ent_emb, rel_emb):
        assert g.number_of_nodes() == ent_emb.shape[0]
        
        with g.local_scope():
            g.ndata['emb'] = ent_emb
            g.edata['emb'] = rel_emb[g.edata['etype']]
            
            if self.comp_fn == "sub":
                g.apply_edges(fn.u_sub_e('emb', 'emb', 'comp_emb'))
            elif self.comp_fn == "mul":
                g.apply_edges(fn.u_mul_e('emb', 'emb', 'comp_emb'))
            else:
                raise NotImplementedError
            
            g.apply_edges(fn.e_dot_v('comp_emb', 'emb', 'attn'))
            g.edata['attn'] = dgl.ops.edge_softmax(g, g.edata['attn'])
            
            g.edata['comp_emb'] = self.in_out_calc(g, g.edata['comp_emb'])
            g.edata['comp_emb'] = g.edata['comp_emb'] * g.edata['attn']
            g.update_all(fn.copy_e('comp_emb', 'm'), fn.sum('m', 'neigh'))
            
            neigh_ent_emb = g.ndata['neigh']
            
            # apply linear transformation
            neigh_ent_emb = self.bn(neigh_ent_emb)
            neigh_ent_emb = self.activation(neigh_ent_emb)          
            
        return neigh_ent_emb

class SEGNN_Layer(nn.Module):
    def __init__(self, in_dim, out_dim, comp_fn="mul", batch_norm=True, layer_dropout=0.3, num_samples=20):
        super().__init__()
        self.comp_fn = comp_fn
        
        self.edge_layer = EdgeLayer(in_dim, out_dim)
        self.node_layer = NodeLayer(in_dim, out_dim)
        self.comp_layer = CompLayer(in_dim, out_dim, comp_fn)
        
        self.w_loop = nn.Linear(in_dim, out_dim)
        self.w_rel = nn.Linear(in_dim, out_dim)
        nn.init.xavier_normal_(self.w_loop.weight)
        nn.init.xavier_normal_(self.w_rel.weight)

        self.ent_drop = nn.Dropout(layer_dropout)
        self.rel_drop = nn.Dropout(layer_dropout-0.2)
        
        self.act = nn.Tanh()
        self.act2 = nn.Tanh()
        
        self.num_samples = num_samples
        self.w_rel_edge = nn.Linear(in_dim, out_dim)
        nn.init.xavier_normal_(self.w_rel_edge.weight)
    
    def update_relations(self, ent_emb, rel_emb, rel2ent_pairs):
        device = rel_emb.device
        num_rels = rel_emb.shape[0]
        all_heads = []
        all_tails = []
        rel_mask = []  # 记录每个关系的实体对数量
        
        # 预处理所有关系的实体对
        for rel_id in range(num_rels):
            ent_pairs = rel2ent_pairs[rel_id]
            if not ent_pairs:
                continue
                
            # 采样或使用全部实体对
            if len(ent_pairs) > self.num_samples:
                ent_pairs = random.sample(ent_pairs, self.num_samples)
                
            heads, tails = zip(*ent_pairs)
            all_heads.extend(heads)
            all_tails.extend(tails)
            rel_mask.extend([rel_id] * len(ent_pairs))
        
        # 批量处理
        all_heads = th.tensor(all_heads, device=device)
        all_tails = th.tensor(all_tails, device=device)
        rel_mask = th.tensor(rel_mask, device=device)
        
        # 一次性获取所有实体嵌入
        head_embs = ent_emb[all_heads]
        tail_embs = ent_emb[all_tails]
        
        # 批量线性变换
        pair_embs = tail_embs-head_embs
        rel_updates = self.w_rel_edge(pair_embs)
        
        # 使用scatter_mean聚合更新
        updated_rel_emb = th.zeros_like(rel_emb)
        rel_count = th.zeros(num_rels, device=device)
        
        updated_rel_emb.scatter_add_(0, rel_mask.unsqueeze(1).expand(-1, rel_updates.size(1)), rel_updates)
        rel_count.scatter_add_(0, rel_mask, th.ones_like(rel_mask, dtype=th.float))
        rel_count = th.clamp(rel_count.unsqueeze(1), min=1)  # 避免除零
        
        updated_rel_emb = updated_rel_emb / rel_count
        
        return updated_rel_emb
    
    def forward(self, g, ent_emb, rel_emb, rel2ent_pairs):
        ent_emb, rel_emb = self.ent_drop(ent_emb), self.rel_drop(rel_emb)
        edge_ent_emb = self.edge_layer(g, ent_emb, rel_emb)
        node_ent_emb = self.node_layer(g, ent_emb)
        comp_ent_emb = self.comp_layer(g, ent_emb, rel_emb)
        # loop_ent_emb = self.act(self.w_loop(ent_emb))
        loop_ent_emb = ent_emb
        
        final_ent_emb = comp_ent_emb + loop_ent_emb + edge_ent_emb + node_ent_emb
        
        # # updated_rel_emb = self.update_relations(ent_emb, rel_emb, rel2ent_pairs)
        # final_rel_emb = self.w_rel(rel_emb) # + self.act2(updated_rel_emb))
        updated_rel_emb = self.update_relations(ent_emb, rel_emb, rel2ent_pairs)
        # final_rel_emb = self.w_rel(rel_emb) + self.act2(updated_rel_emb)
        final_rel_emb = rel_emb + self.act2(updated_rel_emb)
               
        return final_ent_emb, final_rel_emb
        

class SEGNN(nn.Module):
    def __init__(self, num_ent, num_rel, in_dim, out_dim, layer_size, comp_fn="mul", batch_norm=True, layer_dropout=0.3):
        super().__init__()
        self.comp_fn = comp_fn
        
        self.node_embedding = nn.Parameter(th.Tensor(num_ent, in_dim))
        self.rel_embedding = nn.Parameter(th.Tensor(num_rel, in_dim))
        nn.init.xavier_normal_(self.node_embedding)
        nn.init.xavier_normal_(self.rel_embedding)
        
        self.layers = nn.ModuleList(
            [SEGNN_Layer(in_dim, out_dim, comp_fn, batch_norm, layer_dropout) 
             for _ in range(layer_size)])
        self.dropouts = nn.ModuleList()
        for i in range(layer_size):
            self.dropouts.append(nn.Dropout(layer_dropout))
        
    def forward(self, g, rel2ent_pairs):
        n_feats = self.node_embedding
        r_feats = self.rel_embedding
        for layer, dropout in zip(self.layers, self.dropouts):
            n_feats, r_feats = layer(g, n_feats, r_feats, rel2ent_pairs)
            n_feats = dropout(n_feats)
        
        return n_feats, r_feats       
    
class Final_Model(nn.Module):
    def __init__(self, num_nodes,
            num_rels, n_layers,
            d_embd, d_k, d_v,
            d_model, d_inner, n_head,
            label_smoothing=0.1,
            **kwargs):
        super().__init__()
        self.gnn_encoder = SEGNN(
            num_nodes, num_rels, d_embd, d_embd,
            n_layers, comp_fn="mul", batch_norm=True, layer_dropout=0.3
        )
        self.cnn_decoder = Rel_InteractE(
            d_embd, 10, 20, 64, num_rels, num_nodes,
            filter_sz=(3, 3), input_drop=0.3, hid_drop=0.3, feat_drop=0.3, num_filt=64
        )
        # self.cnn_decoder = CNN_Decoder(
        #     d_embd, 10, 20, 32, num_rels, num_nodes,
        #     filter1_sz=(1, 3), filter2_sz=(3, 3), filter3_sz=(1, 5),
        #     input_drop=0.3, hid_drop=0.3, feat_drop=0.3
        # )
        # self.cnn_decoder = InteractE(
        #     d_embd, 10, 20, 64, num_rels, num_nodes,
        #     filter_sz=(3, 3), input_drop=0.3, hid_drop=0.3, feat_drop=0.3, num_filt=64
        # )
        # self.attn_decoder = Encoder(
        #     d_embd, 1, n_head, d_k, d_v,
        #     d_model, d_inner, **kwargs
        # )
        self.ent_bn = nn.BatchNorm1d(d_embd)
        self.rel_bn = nn.BatchNorm1d(d_embd)
    
    def forward(self, g, sub, rel, rel2ent_pairs):
        ### 待完成 注释掉SAttle中的embedding部分，将输入改为encoder的输出
        
        # 全图编码
        n_feats, r_feats = self.gnn_encoder(g, rel2ent_pairs)
        scores = self.cnn_decoder(n_feats, r_feats, sub, rel)
        # sub_emb = n_feats[sub, :]
        # rel_emb = r_feats[rel, :]
        
        # sub_emb = self.ent_bn(sub_emb)[:, None]
        # rel_emb = self.rel_bn(rel_emb)[:, None]
        # edges = th.hstack((sub_emb, rel_emb))
        
        # emb_edges = self.attn_decoder(edges)
        # src_edges = emb_edges[:, 1, :]
        # scores = th.mm(src_edges, n_feats.transpose(0, 1))
              
        return scores
    
    def cal_loss(self, scores, labels, label_smooth=True):
        pred_loss = F.binary_cross_entropy_with_logits(scores, labels)
        return pred_loss

