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
from CNN_decoder import ConvE, InteractE, Rel_InteractE

class ATTLayer(nn.Module):
    def __init__(self, channel, reduction=18):
        super(ATTLayer, self).__init__()
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel * 3, bias=False),
            # nn.Sigmoid()
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x) # (1500,32,1,1)
        y = y.view(b, c) # (1500,32)
        y = self.fc(y) # (1500,32)
        y = y.view(b, c*3, 1, 1) # (1500,32,1,1)
        y1, y2, y3 = th.split(y, [self.channel, self.channel, self.channel], dim=1)
        return y1,y2,y3

class SELayer(nn.Module):
    def __init__(self, channel, reduction=18):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x) # (1500,32,1,1)
        y = y.view(b, c) # (1500,32)
        y = self.fc(y) # (1500,32)
        y = y.view(b, c, 1, 1) # (1500,32,1,1)
        y = y.expand_as(x) # (1500,32,20,20)
        return x * y

class Conv_with_Kernel(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, num_rel, reshape_H, reshape_W):
        super(Conv_with_Kernel, self).__init__()
        self.filter_sz = kernel_size
        self.h = kernel_size[0]
        self.w = kernel_size[1]
        
        self.reshape_H = reshape_H 
        self.reshape_W = reshape_W
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        self.filter_dim = in_channel * out_channel * self.h * self.w
        self.filter = th.nn.Embedding(num_rel, self.filter_dim, padding_idx=0)
        nn.init.xavier_normal_(self.filter.weight.data)
        self.bn = nn.BatchNorm2d(out_channel)      
    
    def forward(self, x, rel, batch_sz):
        f = self.filter(rel)
        f = f.reshape(batch_sz * self.in_channel * self.out_channel, 1, self.h, self.w)
        
        x = F.conv2d(x, f, groups=batch_sz, padding=(int(self.h-1)//2, int(self.w-1)//2))
        x = x.reshape(batch_sz, self.out_channel, self.reshape_H, self.reshape_W)
        x = self.bn(x)
        
        return x
    
class CNN_Decoder(nn.Module):
    def __init__(self, d_embd, k_w, k_h, output_channel, num_rel, num_ent,
                 filter1_sz = (1, 5), filter2_sz = (3, 3), filter3_sz = (5, 1), 
                 input_drop=0.3, hid_drop=0.3, feat_drop=0.3):
        super().__init__()
        
        assert int(k_w * k_h) == d_embd
        
        self.k_w = k_w
        self.k_h = k_h
        
        self.input_drop = nn.Dropout(input_drop)   
        self.hid_drop = nn.Dropout(hid_drop)
        self.feat_drop = nn.Dropout2d(feat_drop)
        
        self.in_channel = 1
        self.out_channel = output_channel
        
        # batchnorm and dropout
        self.bn0 = th.nn.BatchNorm2d(1)
        self.bn2 = th.nn.BatchNorm1d(d_embd)
        
        # 3 kernel filters
        self.conv1 = Conv_with_Kernel(self.in_channel, self.out_channel, filter1_sz, num_rel, 20, 20)
        self.conv2 = Conv_with_Kernel(self.in_channel, self.out_channel, filter2_sz, num_rel, 20, 20)
        self.conv3 = Conv_with_Kernel(self.in_channel, self.out_channel, filter3_sz, num_rel, 20, 20)
        
        # SE layer and ATT layer
        self.se1 = SELayer(output_channel, reduction = int(0.5*output_channel))
        self.se2 = SELayer(output_channel, reduction = int(0.5*output_channel))
        self.se3 = SELayer(output_channel, reduction = int(0.5*output_channel))
        self.att = ATTLayer(output_channel,reduction = int(0.5*output_channel))
        
        # FC layer
        self.reshapeH = 20
        self.reshapeW = 20
        fc_sz = self.reshapeH * self.reshapeW * (3 * output_channel)
        self.fc = th.nn.Linear(fc_sz, d_embd)
        self.register_parameter('b', th.nn.Parameter(th.zeros(num_ent)))
        
        nn.init.xavier_normal_(self.fc.weight)
        
        self.chequer_perm = self.get_permutation(k_w, k_h)
    
    def chunk_and_reshape(self, x):
        t1, t2 = x.chunk(2)
        reshape = th.cat([t2, t1], dim=0)
        return reshape
    
    def get_permutation(self, kw, kh):
        ori_sub = th.arange(kw * kh)
        ori_rel = th.arange(kw * kh, 2 * kw * kh)
        
        # rsp_sub = self.chunk_and_reshape(ori_sub)
        # rsp_rel = self.chunk_and_reshape(ori_rel)
        
        perm = self.get_chessboard(ori_sub.view(kh, kw), ori_rel.view(kh, kw), kw, kh)
        # perm2 = self.get_chessboard(ori_sub.view(kh, kw), rsp_rel.view(kh, kw), kw, kh)
        # perm3 = self.get_chessboard(rsp_sub.view(kh, kw), ori_rel.view(kh, kw), kw, kh)
        # perm4 = self.get_chessboard(rsp_sub.view(kh, kw), rsp_rel.view(kh, kw), kw, kh)
        
        # perm = th.stack([perm1, perm2, perm3, perm4], dim=0)
        
        return perm
      
    def get_chessboard(self, sub_arrange, rel_arrange, kw, kh):
        # 初始化结果张量
        result = th.zeros(2 * kw, kh)

        # 棋盘格交错排列
        for i in range(kh):
            # 交替填充 A 和 B 的行
            if i % 2 == 0:
                # 偶数行：A 的元素在前，B 的元素在后
                result[::2, i] = sub_arrange[i, :]  # 填充 A 的行
                result[1::2, i] = rel_arrange[i, :]  # 填充 B 的行
            else:
                # 奇数行：B 的元素在前，A 的元素在后
                result[::2, i] = sub_arrange[i, :]  # 填充 B 的行
                result[1::2, i] = rel_arrange[i, :]  # 填充 A 的行
                
        # print(result.shape)    
        result = result.transpose(0, 1).contiguous()   
        result = result.view(-1)

        return result.to(th.int)
    
    def forward(self, n_feats, r_feats, sub, rel):
        # get chessboard permutation of x
        sub_emb = n_feats[sub, :]
        rel_emb = r_feats[rel, :]
        batch_sz = sub_emb.shape[0]
        comb_emb = th.cat([sub_emb, rel_emb], dim=1)
        tmp = comb_emb[:, self.chequer_perm]
        stack_inp = tmp.reshape(-1, 1, 2*self.k_w, self.k_h)
        x = self.bn0(stack_inp)
        x = self.input_drop(x)
        x = x.permute(1, 0, 2, 3)
        
        # convolution
        # shape->(batch_size, filter_dim)
        x1 = self.conv1(x, rel, batch_sz)
        x2 = self.conv2(x, rel, batch_sz)
        x3 = self.conv3(x, rel, batch_sz)
        
        # # squeeze and excitation
        # x1 = self.se1(x1)
        # x2 = self.se2(x2)
        # x3 = self.se3(x3)
        
        # attention
        x = x1 + x2 + x3
        y1, y2, y3 = self.att(x)
        y1 = y1.expand_as(x1)
        y2 = y2.expand_as(x2)
        y3 = y3.expand_as(x3)
        x1 = x1 * y1
        x2 = x2 * y2
        x3 = x3 * y3
        
        x = th.cat([x1, x2, x3], dim=1)
        x = th.relu(x)
        x = self.feat_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        # print(x.shape)
        x = self.hid_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = th.mm(x, n_feats.transpose(0, 1))
        x += self.b.expand_as(x)        
        pred = x
        return pred     

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
            g.edata['emb'] = rel_emb[g.edata['etype']] # * g.edata['norm']
            
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
            g.edata['emb'] = rel_emb[g.edata['etype']] # * g.edata['norm']
            
            # print("fuck:", self.comp_fn)
            
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
    def __init__(self, in_dim, out_dim, comp_fn="mul", batch_norm=True, layer_dropout=0.3):
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
        # self.rel_drop = nn.Dropout(0.0)
        
        self.act = nn.Tanh()
    
    def forward(self, g, ent_emb, rel_emb):
        ent_emb, rel_emb = self.ent_drop(ent_emb), self.rel_drop(rel_emb)
        edge_ent_emb = self.edge_layer(g, ent_emb, rel_emb)
        node_ent_emb = self.node_layer(g, ent_emb)
        comp_ent_emb = self.comp_layer(g, ent_emb, rel_emb)
        loop_ent_emb = self.act(self.w_loop(ent_emb))
        loop_ent_emb = ent_emb
        
        final_ent_emb = comp_ent_emb + loop_ent_emb + edge_ent_emb + node_ent_emb
        final_rel_emb = self.w_rel(rel_emb)
               
        return final_ent_emb, final_rel_emb
        

class SEGNN(nn.Module):
    def __init__(self, num_ent, num_rel, in_dim, out_dim, layer_size, comp_fn="mul", batch_norm=True, layer_dropout=0.3):
        super().__init__()
        self.comp_fn = comp_fn
        
        self.node_embedding = nn.Parameter(th.Tensor(num_ent, in_dim))
        self.rel_embedding = nn.Parameter(th.Tensor(num_rel, in_dim))
        nn.init.xavier_normal_(self.node_embedding)
        nn.init.xavier_normal_(self.rel_embedding)
        
        print(layer_dropout)
        self.layers = nn.ModuleList(
            [SEGNN_Layer(in_dim, out_dim, comp_fn, batch_norm, layer_dropout) 
             for _ in range(layer_size)])
        self.dropouts = nn.ModuleList()
        for i in range(layer_size):
            self.dropouts.append(nn.Dropout(layer_dropout))
        
    def forward(self, g):
        n_feats = self.node_embedding
        r_feats = self.rel_embedding
        # print(self.comp_fn)
        for layer, dropout in zip(self.layers, self.dropouts):
            n_feats, r_feats = layer(g, n_feats, r_feats)
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
        # self.cnn_decoder = CNN_Decoder(
        #     d_embd, 10, 20, 32, num_rels, num_nodes,
        #     filter1_sz=(1, 5), filter2_sz=(3, 3), filter3_sz=(5, 1),
        #     input_drop=0.3, hid_drop=0.3, feat_drop=0.3
        # )
        self.cnn_decoder = ConvE(
            d_embd, 10, 20, 64, num_rels, num_nodes,
            filter_sz=(3, 3), input_drop=0.3, hid_drop=0.3, feat_drop=0.3, num_filt=64
        )
        # self.attn_decoder = Encoder(
        #     d_embd, 1, n_head, d_k, d_v,
        #     d_model, d_inner, **kwargs
        # )
        self.ent_bn = nn.BatchNorm1d(d_embd)
        self.rel_bn = nn.BatchNorm1d(d_embd)
    
    def forward(self, g, sub, rel):
        ### 待完成 注释掉SAttle中的embedding部分，将输入改为encoder的输出
        
        # 全图编码
        n_feats, r_feats = self.gnn_encoder(g)
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

