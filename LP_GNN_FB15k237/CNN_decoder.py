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
        
        # squeeze and excitation
        x1 = self.se1(x1)
        x2 = self.se2(x2)
        x3 = self.se3(x3)
        
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
    
# Use convE as the score function
class ConvE(nn.Module):
    def __init__(self, d_embd, k_w, k_h, output_channel, num_rel, num_ent,
                 filter_sz = (3, 3), input_drop=0.3, hid_drop=0.3, feat_drop=0.3,
                 num_filt=64):
        super().__init__()

        assert int(k_w * k_h) == d_embd
        
        self.k_w = k_w
        self.k_h = k_h
        
        self.num_filt = num_filt
        self.ker_sz = filter_sz
        
        self.in_channel = 1
        self.out_channel = output_channel
        self.embed_dim = d_embd
        
        # batchnorm and dropout
        self.bn0 = th.nn.BatchNorm2d(1)
        self.bn1 = th.nn.BatchNorm2d(self.num_filt)
        self.bn2 = th.nn.BatchNorm1d(d_embd)

        # dropouts and conv module to the combined (sub+rel) emb
        self.hidden_drop = th.nn.Dropout(hid_drop)
        self.feature_drop = th.nn.Dropout(feat_drop)
              
        self.m_conv1 = th.nn.Conv2d(
            1,
            out_channels=self.num_filt,
            kernel_size=self.ker_sz,
            stride=1,
            padding=0,
            bias=False,
        )

        flat_sz_h = int(2 * self.k_w) - self.ker_sz[0] + 1
        flat_sz_w = self.k_h - self.ker_sz[0] + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.num_filt
        
        print("flat_sz: ", self.flat_sz)
        print(flat_sz_h, flat_sz_w)
        self.fc = th.nn.Linear(self.flat_sz, self.embed_dim)

        # bias to the score
        self.bias = nn.Parameter(th.zeros(num_ent))

    # combine entity embeddings and relation embeddings
    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.embed_dim)
        stack_inp = th.cat([e1_embed, rel_embed], 1)
        stack_inp = th.transpose(stack_inp, 2, 1).reshape(
            (-1, 1, 2 * self.k_w, self.k_h)
        )
        return stack_inp

    def forward(self, n_feats, r_feats, sub, rel):
        # get sub_emb and rel_emb via compGCN
        # print("sub: ", sub.shape)
        sub_emb = n_feats[sub, :]
        rel_emb = r_feats[rel, :]
        
        # print("sub_emb: ", sub_emb.shape)
        
        # combine the sub_emb and rel_emb
        stk_inp = self.concat(sub_emb, rel_emb)
        # use convE to score the combined emb
        # print(stk_inp.shape)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        # print(x.shape)        
        x = x.view(-1, self.flat_sz)
        # print(x.shape)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        # compute score
        x = th.mm(x, n_feats.transpose(1, 0))
        # add in bias
        x += self.bias.expand_as(x)
        score = x
        return score
    
class InteractE(nn.Module):
    def __init__(self, d_embd, k_w, k_h, output_channel, num_rel, num_ent,
                 filter_sz = (3, 3), input_drop=0.3, hid_drop=0.3, feat_drop=0.3,
                 num_filt=64):
        super().__init__()

        assert int(k_w * k_h) == d_embd
        
        self.k_w = k_w
        self.k_h = k_h
        self.embed_dim = d_embd
        
        self.num_filt = num_filt
        self.ker_sz = filter_sz
        
        self.in_channel = 1
        self.out_channel = output_channel
        
        # batchnorm and dropout
        self.bn0 = th.nn.BatchNorm2d(1)
        self.bn1 = th.nn.BatchNorm2d(self.num_filt)
        self.bn2 = th.nn.BatchNorm1d(d_embd)

        # dropouts and conv module to the combined (sub+rel) emb
        self.hidden_drop = th.nn.Dropout(hid_drop)
        self.feature_drop = th.nn.Dropout(feat_drop)
           
        self.m_conv1 = th.nn.Conv2d(
            1,
            out_channels=self.num_filt,
            kernel_size=self.ker_sz,
            stride=1,
            padding=0,
            bias=False,
        )

        flat_sz_h = int(2 * self.k_w) - self.ker_sz[0] + 1
        flat_sz_w = self.k_h - self.ker_sz[0] + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.num_filt
        
        print("flat_sz: ", self.flat_sz)
        print(flat_sz_h, flat_sz_w)
        self.fc = th.nn.Linear(self.flat_sz, self.embed_dim)

        # bias to the score
        self.bias = nn.Parameter(th.zeros(num_ent))
        self.chequer_perm = self.interleave_tensors_chessboard(self.k_w, self.k_h)

    # combine entity embeddings and relation embeddings
    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.embed_dim)
        stack_inp = th.cat([e1_embed, rel_embed], 1)
        stack_inp = th.transpose(stack_inp, 2, 1).reshape(
            (-1, 1, 2 * self.k_w, self.k_h)
        )
        return stack_inp

    def interleave_tensors_chessboard(self, kw, kh):
        # 创建两个输入张量
        A = th.arange(kw * kh).reshape(kw * kh).float()
        B = th.arange(kw * kh, 2  * kw * kh).reshape(kw * kh).float()

        # 将 A 和 B 重塑为 (batch, kh, kw)
        A_reshaped = A.view(kh, kw)
        B_reshaped = B.view(kh, kw)

        # 初始化结果张量
        result = th.zeros(2 * kw, kh)

        # 棋盘格交错排列
        for i in range(kh):
            # 交替填充 A 和 B 的行
            if i % 2 == 0:
                # 偶数行：A 的元素在前，B 的元素在后
                result[::2, i] = A_reshaped[i, :]  # 填充 A 的行
                result[1::2, i] = B_reshaped[i, :]  # 填充 B 的行
            else:
                # 奇数行：B 的元素在前，A 的元素在后
                result[::2, i] = B_reshaped[i, :]  # 填充 B 的行
                result[1::2, i] = A_reshaped[i, :]  # 填充 A 的行
                
        # print(result.shape)    
        result = result.transpose(0, 1).contiguous()   
        result = result.view(-1)

        return result.to(th.int)
    
    def forward(self, n_feats, r_feats, sub, rel):
        # get sub_emb and rel_emb via compGCN
        # print("sub: ", sub.shape)
        sub_emb = n_feats[sub, :]
        rel_emb = r_feats[rel, :]
        # print("sub_emb: ", sub_emb.shape)
        comb_emb = th.cat([sub_emb, rel_emb], dim=1)
        tmp = comb_emb[:, self.chequer_perm]
        stack_inp = tmp.reshape(-1, 1, 2 * self.k_w, self.k_h)       
        # print(stack_inp.shape)
        
        stack_inp = self.bn0(stack_inp)
        x = self.m_conv1(stack_inp)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        # print(x.shape)
        x = th.mm(x, n_feats.transpose(1, 0))
        # add in bias
        x += self.bias.expand_as(x)
        return x

class Rel_InteractE(nn.Module):
    def __init__(self, d_embd, k_w, k_h, output_channel, num_rel, num_ent,
                 filter_sz = (3, 3), input_drop=0.3, hid_drop=0.3, feat_drop=0.3,
                 num_filt=64):
        super().__init__()

        assert int(k_w * k_h) == d_embd
        
        self.k_w = k_w
        self.k_h = k_h
        self.embed_dim = d_embd
        
        self.num_filt = num_filt
        self.ker_sz = filter_sz
        
        self.in_channel = 1
        self.out_channel = output_channel
        
        # batchnorm and dropout
        self.bn0 = th.nn.BatchNorm2d(1)
        self.bn1 = th.nn.BatchNorm2d(self.num_filt)
        self.bn2 = th.nn.BatchNorm1d(d_embd)

        # dropouts and conv module to the combined (sub+rel) emb
        self.hidden_drop = th.nn.Dropout(hid_drop)
        self.feature_drop = th.nn.Dropout(feat_drop)
           
        # self.m_conv1 = th.nn.Conv2d(
        #     1,
        #     out_channels=self.num_filt,
        #     kernel_size=self.ker_sz,
        #     stride=1,
        #     padding=0,
        #     bias=False,
        # )
        self.conv1 = Conv_with_Kernel(self.in_channel, self.out_channel, filter_sz, num_rel, 20, 20)

        # flat_sz_h = int(2 * self.k_w) - self.ker_sz[0] + 1
        # flat_sz_w = self.k_h - self.ker_sz[0] + 1
        # self.flat_sz = flat_sz_h * flat_sz_w * self.num_filt
        
        self.flat_sz = 2 * self.k_w * self.k_h * self.out_channel
        print("flat_sz: ", self.flat_sz)
        # print(flat_sz_h, flat_sz_w)
        self.fc = th.nn.Linear(self.flat_sz, self.embed_dim)

        # bias to the score
        self.bias = nn.Parameter(th.zeros(num_ent))
        self.chequer_perm = self.interleave_tensors_chessboard(self.k_w, self.k_h)

    # combine entity embeddings and relation embeddings
    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.embed_dim)
        stack_inp = th.cat([e1_embed, rel_embed], 1)
        stack_inp = th.transpose(stack_inp, 2, 1).reshape(
            (-1, 1, 2 * self.k_w, self.k_h)
        )
        return stack_inp

    def interleave_tensors_chessboard(self, kw, kh):
        # 创建两个输入张量
        A = th.arange(kw * kh).reshape(kw * kh).float()
        B = th.arange(kw * kh, 2  * kw * kh).reshape(kw * kh).float()

        # 将 A 和 B 重塑为 (batch, kh, kw)
        A_reshaped = A.view(kh, kw)
        B_reshaped = B.view(kh, kw)

        # 初始化结果张量
        result = th.zeros(2 * kw, kh)

        # 棋盘格交错排列
        for i in range(kh):
            # 交替填充 A 和 B 的行
            if i % 2 == 0:
                # 偶数行：A 的元素在前，B 的元素在后
                result[::2, i] = A_reshaped[i, :]  # 填充 A 的行
                result[1::2, i] = B_reshaped[i, :]  # 填充 B 的行
            else:
                # 奇数行：B 的元素在前，A 的元素在后
                result[::2, i] = B_reshaped[i, :]  # 填充 B 的行
                result[1::2, i] = A_reshaped[i, :]  # 填充 A 的行
                
        # print(result.shape)    
        result = result.transpose(0, 1).contiguous()   
        result = result.view(-1)

        return result.to(th.int)
    
    def forward(self, n_feats, r_feats, sub, rel):
        # get sub_emb and rel_emb via compGCN
        # print("sub: ", sub.shape)
        sub_emb = n_feats[sub, :]
        rel_emb = r_feats[rel, :]
        # print("sub_emb: ", sub_emb.shape)
        comb_emb = th.cat([sub_emb, rel_emb], dim=1)
        tmp = comb_emb[:, self.chequer_perm]
        stack_inp = tmp.reshape(-1, 1, 2 * self.k_w, self.k_h)       
        # print(stack_inp.shape)
        
        stack_inp = self.bn0(stack_inp)
        batch_sz = stack_inp.shape[0]
        x = stack_inp.permute(1, 0, 2, 3)
        
        x = self.conv1(x, rel, batch_sz)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        # print(x.shape)
        x = th.mm(x, n_feats.transpose(1, 0))
        # add in bias
        x += self.bias.expand_as(x)
        return x
    
