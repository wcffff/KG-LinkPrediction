import torch
import torch.nn as nn
import utils
import torch.nn.functional as F


class ConvE(nn.Module):
    def __init__(self, h_dim, out_channels, ker_sz):
        super().__init__()
        cfg = utils.get_global_config()
        self.cfg = cfg
        dataset = cfg.dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.bn2 = torch.nn.BatchNorm1d(h_dim)

        self.conv_drop = torch.nn.Dropout(cfg.conv_drop)
        self.fc_drop = torch.nn.Dropout(cfg.fc_drop)
        self.k_h = cfg.k_h
        self.k_w = cfg.k_w
        assert self.k_h * self.k_w == h_dim
        self.conv = torch.nn.Conv2d(1, out_channels=out_channels, stride=1, padding=0,
                                    kernel_size=ker_sz, bias=False)
        flat_sz_h = int(2 * self.k_h) - ker_sz + 1
        flat_sz_w = self.k_w - ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * out_channels
        self.fc = torch.nn.Linear(self.flat_sz, h_dim, bias=False)
        self.ent_drop = nn.Dropout(cfg.ent_drop_pred)

    def forward(self, head, rel, all_ent):
        # head (bs, h_dim), rel (bs, h_dim)
        # concatenate and reshape to 2D
        c_head = head.view(-1, 1, head.shape[-1])
        c_rel = rel.view(-1, 1, rel.shape[-1])
        c_emb = torch.cat([c_head, c_rel], 1)
        c_emb = torch.transpose(c_emb, 2, 1).reshape((-1, 1, 2 * self.k_h, self.k_w))

        x = self.bn0(c_emb)
        x = self.conv(x)  # (bs, out_channels, out_h, out_w)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv_drop(x)
        x = x.view(-1, self.flat_sz)  # (bs, out_channels * out_h * out_w)
        x = self.fc(x)  # (bs, h_dim)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc_drop(x)  # (bs, h_dim)
        # inference
        # all_ent: (n_ent, h_dim)
        all_ent = self.ent_drop(all_ent)
        x = torch.mm(x, all_ent.transpose(1, 0))  # (bs, n_ent)
        x = torch.sigmoid(x)
        return x


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
        self.filter = torch.nn.Embedding(num_rel, self.filter_dim, padding_idx=0)
        nn.init.xavier_normal_(self.filter.weight.data)
        self.bn = nn.BatchNorm2d(out_channel)      
    
    def forward(self, x, rel, batch_sz):
        f = self.filter(rel)
        f = f.reshape(batch_sz * self.in_channel * self.out_channel, 1, self.h, self.w)
        
        x = F.conv2d(x, f, groups=batch_sz, padding=(int(self.h-1)//2, int(self.w-1)//2))
        x = x.reshape(batch_sz, self.out_channel, self.reshape_H, self.reshape_W)
        x = self.bn(x)
        
        return x

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
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.num_filt)
        self.bn2 = torch.nn.BatchNorm1d(d_embd)

        # dropouts and conv module to the combined (sub+rel) emb
        self.hidden_drop = torch.nn.Dropout(hid_drop)
        self.feature_drop = torch.nn.Dropout(feat_drop)
           
        self.m_conv1 = torch.nn.Conv2d(
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
        self.fc = torch.nn.Linear(self.flat_sz, self.embed_dim)

        # bias to the score
        self.bias = nn.Parameter(torch.zeros(num_ent))
        self.chequer_perm = self.interleave_tensors_chessboard(self.k_w, self.k_h)

    # combine entity embeddings and relation embeddings
    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.embed_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape(
            (-1, 1, 2 * self.k_w, self.k_h)
        )
        return stack_inp

    def interleave_tensors_chessboard(self, kw, kh):
        # 创建两个输入张量
        A = torch.arange(kw * kh).reshape(kw * kh).float()
        B = torch.arange(kw * kh, 2  * kw * kh).reshape(kw * kh).float()

        # 将 A 和 B 重塑为 (batch, kh, kw)
        A_reshaped = A.view(kh, kw)
        B_reshaped = B.view(kh, kw)

        # 初始化结果张量
        result = torch.zeros(2 * kw, kh)

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

        return result.to(torch.int)
    
    def forward(self, n_feats, r_feats, sub, rel):
        # get sub_emb and rel_emb via compGCN
        # print("sub: ", sub.shape)
        sub_emb = n_feats[sub, :]
        rel_emb = r_feats[rel, :]
        # print("sub_emb: ", sub_emb.shape)
        comb_emb = torch.cat([sub_emb, rel_emb], dim=1)
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
        x = torch.mm(x, n_feats.transpose(1, 0))
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
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.num_filt)
        self.bn2 = torch.nn.BatchNorm1d(d_embd)

        # dropouts and conv module to the combined (sub+rel) emb
        self.hidden_drop = torch.nn.Dropout(hid_drop)
        self.feature_drop = torch.nn.Dropout(feat_drop)
           
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
        self.fc = torch.nn.Linear(self.flat_sz, self.embed_dim)

        # bias to the score
        self.bias = nn.Parameter(torch.zeros(num_ent))
        self.chequer_perm = self.interleave_tensors_chessboard(self.k_w, self.k_h)

    # combine entity embeddings and relation embeddings
    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.embed_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape(
            (-1, 1, 2 * self.k_w, self.k_h)
        )
        return stack_inp

    def interleave_tensors_chessboard(self, kw, kh):
        # 创建两个输入张量
        A = torch.arange(kw * kh).reshape(kw * kh).float()
        B = torch.arange(kw * kh, 2  * kw * kh).reshape(kw * kh).float()

        # 将 A 和 B 重塑为 (batch, kh, kw)
        A_reshaped = A.view(kh, kw)
        B_reshaped = B.view(kh, kw)

        # 初始化结果张量
        result = torch.zeros(2 * kw, kh)

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

        return result.to(torch.int)
    
    def forward(self, sub, rel, n_feats):
        # get sub_emb and rel_emb via compGCN
        # print("sub: ", sub.shape)
        # print("sub_emb: ", sub_emb.shape)
        comb_emb = torch.cat([sub, rel], dim=1)
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
        x = torch.mm(x, n_feats.transpose(1, 0))
        # add in bias
        x += self.bias.expand_as(x)
        return x
