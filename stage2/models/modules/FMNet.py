import functools
# from warnings import filters
import torch.nn as nn
import torch
import models.modules.arch_util as arch_util
import torch.nn.functional as F
from models.modules.arch_util import initialize_weights
# from utils.gpu_memory_log import gpu_memory_log
# import math
from models.modules.mem_block import Memory
# from functools import reduce
# from operator import mul
from einops import rearrange


class MoEGate(nn.Module):
    def __init__(self, embed_dim, num_experts=4, num_experts_per_tok=2):
        super().__init__()
        self.gating_dim = embed_dim  # tc
        self.n_routed_experts = num_experts  # 4
        self.top_k = num_experts_per_tok  # 2
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))  # 可学习的权重 [e,tc]
        nn.init.xavier_uniform_(self.weight)

        self.gap = nn.AdaptiveMaxPool2d(1)
        self.gap2 = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):  # [b,h,w,tc]    
        b,h,w,tc=x.shape
        x = x.permute(0, 3, 1, 2)  # [b,tc,h,w]
        x = self.gap(x) + self.gap2(x)  # [b,tc,1,1]
        x = x.view(-1, tc)  # [b,tc]

        ### compute gating score
        logits = F.linear(x, self.weight, None)  # [b,tc]*[e,tc]=[b,e]
        scores = logits.softmax(dim=-1)
        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)  # [b,topk]
        return topk_idx, topk_weight
    
    
class MoeMLP(nn.Module):
    def __init__(self, embed_dim, intermediate_size):
        super().__init__()
        self.embed_dim = embed_dim  # 64
        self.intermediate_size = intermediate_size  # 64x4
        self.gate_proj = nn.Linear(self.embed_dim, self.intermediate_size, bias=False)  # 门控
        self.up_proj = nn.Linear(self.embed_dim, self.intermediate_size, bias=False)  # 上投影
        self.down_proj = nn.Linear(self.intermediate_size, self.embed_dim, bias=False)  # 下投影
        self.act_fn = nn.ReLU()
    def forward(self, x):  # [b,h,w,tc] -> [b,h,w,tc]    
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class SparseMoeBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self, embed_dim, mlp_ratio=4, num_experts=4, num_experts_per_tok=2, num_frame=8):  # num_experts=8,num_experts_per_tok=2,pretraining_tp=2
        super().__init__()
        self.num_experts_per_tok = num_experts_per_tok  # 2
        self.experts = nn.ModuleList([MoeMLP(embed_dim*num_frame, embed_dim*num_frame//mlp_ratio) for i in range(num_experts)])  # 8个专家
        self.gate = MoEGate(embed_dim*num_frame, num_experts, num_experts_per_tok)  # 门控机制
        self.n_shared_experts = 2
        
        if self.n_shared_experts is not None:
            intermediate_size =  embed_dim * self.n_shared_experts // mlp_ratio
            self.shared_experts = MoeMLP(embed_dim*num_frame, intermediate_size*num_frame)  # 共享专家

    def forward(self, x):  # [b,h,w,tc]
        b,h,w,tc = x.shape
        identity = x

        # 门控选择
        topk_idx, topk_weight = self.gate(x)  # [b,h,w,tc] -> [b,topk],[b,topk]
        
        y = torch.zeros_like(x).to(x.device)  # [b,h,w,tc]
        # 专家计算
        for i in range(b):  # batch
            experts = topk_idx[i]  # [topk]
            weights = topk_weight[i]  # [topk]
            expert_out = torch.zeros((1, h, w, tc)).to(x.device)  # [1,h,w,tc]

            for j in range(len(experts)):
                weight = weights[j].view(-1, 1, 1, 1)  # [1,1,1,1]
                expert = self.experts[experts[j]]  # 定义专家
                expert_output = expert(x[i:i+1])  # [1,h,w,tc]
                expert_output_weighted = expert_output* weight  # [1,h,w,tc]
                expert_out += expert_output_weighted

            y[i] = expert_out

        # 共享专家
        if self.n_shared_experts is not None:
            y = y + self.shared_experts(identity)  # [b,h,w,tc] -> [b,h,w,tc] 

        return y  # [b,h,w,tc]


class Attention(nn.Module):
    def __init__(self, dim, t, num_heads):
        super(Attention, self).__init__()
        self.t = t
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads*t, 1, 1))
        
        self.qkv = nn.Conv2d(dim*t,dim*t*3,3,1,1,groups=num_heads*t,bias=True)
        self.project = nn.Conv2d(dim*t,dim*t,3,1,1,groups=num_heads*t,bias=True)
        
    def forward(self, x):   # [b,h,w,t*c] 
        b,h,w,tc = x.shape  
        x = x.permute(0, 3, 1, 2)  # [b,tc,h,w]
        
        qkv = self.qkv(x)  # [b,3tc,h,w]
        q,k,v = qkv.chunk(3,dim=1)  # [b,tc,h,w]
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads*self.t)  # b,n,c,hw
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads*self.t)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads*self.t)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads*self.t, h=h, w=w)
        out = self.project(out)  # [b,tc,h,w]
        out = out.permute(0,2,3,1)  # [b,h,w,tc]
        return out


class MoETrans(nn.Module):
    def __init__(self, 
                 dim, 
                 t,
                 num_heads, 
                 norm_layer=nn.LayerNorm,):
        super(MoETrans, self).__init__()

        self.norm1 = norm_layer(dim*t)
        self.attn = Attention(dim, t, num_heads)
        self.norm2 = norm_layer(dim*t)
        # self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.moe = SparseMoeBlock(embed_dim=dim, mlp_ratio=16, num_experts=4, num_experts_per_tok=2)

    def forward(self, x):  # # [b,t*c,h,w]        
        x = x.permute(0, 2, 3, 1)  # [b,t*c,h,w]  -> [b,h,w,t*c] 
        
        # import pdb;pdb.set_trace()
        x = x + self.attn(self.norm1(x))  # [b,h,w,tc]
        x = x + self.moe(self.norm2(x))  # [b,h,w,tc]
        return x



class MOE(nn.Module):
    def __init__(self, nf=32, dropout_ratio=0.0):
        super(MOE, self).__init__()
        self.nf = nf  
        self.dropout_ratio = dropout_ratio  
        
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        
        self.fc_a = nn.Conv3d(nf,nf,kernel_size=(3,1,1),stride=(1,1,1),padding=(1,0,0),groups=1,bias=True)
        self.fc_b = nn.Conv3d(nf,nf,kernel_size=(3,1,1),stride=(1,1,1),padding=(1,0,0),groups=1,bias=True)
        self.fc_c = nn.Conv3d(nf,nf,kernel_size=(3,1,1),stride=(1,1,1),padding=(1,0,0),groups=1,bias=True)
        self.fuse = nn.Conv3d(nf,nf,kernel_size=(3,1,1),stride=(1,1,1),padding=(1,0,0),groups=1,bias=True)
        
        # self.fc_a = nn.Conv2d(nf, nf, kernel_size=1, bias=True, stride=1, padding=0, groups=1)
        # self.fc_b = nn.Conv2d(nf, nf, kernel_size=1, bias=True, stride=1, padding=0, groups=1)
        # self.fc_c = nn.Conv2d(nf, nf, kernel_size=1, bias=True, stride=1, padding=0, groups=1)
        # self.fuse = nn.Conv2d(nf, nf, kernel_size=1, bias=True, stride=1, padding=0, groups=1)
        
        self.softmax=nn.Softmax(dim=0)
        
    def forward(self, x):  # [b,t,c,h,w]*3
        if len(x) == 4:  
            a, b, c = x[1], x[2], x[3]  
        else:
            a, b, c = x[0].permute(0,2,1,3,4), x[1].permute(0,2,1,3,4), x[2].permute(0,2,1,3,4)  # [b,c,t,h,w]
        
        wa = self.fc_a(a)  
        wb = self.fc_b(b)  
        wc = self.fc_c(c)
        wm = self.softmax(torch.stack([wa, wb, wc], dim=0))
        
        a = a * wm[0,:].squeeze(0)  
        b = b * wm[1,:].squeeze(0)  
        c = c * wm[2,:].squeeze(0)  
        
        out = a + b + c  
        
        if self.dropout_ratio > 0:  
            out = self.dropout(out)  
        
        out = self.fuse(out) + out  #  [b,c,t,h,w]
        out = out.permute(0,2,1,3,4)  # [b,t,c,h,w]
        
        return out



class ResidualBlock_noBN(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class FMNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, act_type='relu', opt=None):  # 3,3,64,16,relu,
        super(FMNet, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 2, 1, bias=True)

        # fm_block = functools.partial(FMBlock, nf=nf, opt=opt)
        # if opt['FM_blockNumber'] == 0:  # 1
        #     self.recon_trunk_fm = nn.Identity()
        # else:
        #     self.recon_trunk_fm = arch_util.make_layer(fm_block, opt['FM_blockNumber'])

        res_block = functools.partial(ResidualBlock_noBN, nf=nf)
        if nb - opt['FM_blockNumber'] == 0:
            self.recon_trunk_res = nn.Identity()
        else:
            self.recon_trunk_res = arch_util.make_layer(res_block, nb - opt['FM_blockNumber'])

        self.upconv = nn.Conv2d(nf, nf*4, 3, 1, 1, bias=True)
        self.upsampler = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        # 记忆模块相关
        self.max_len = 2  # 2
        self.train_scene_list = set()  # 训练数据场景
        self.test_scene_list = set()  # 测试数据场景
        self.memory = Memory(dim=nf, max_len=self.max_len)

        self.mlp = nn.Sequential(
            nn.Conv2d(nf, nf // 4, kernel_size=1) ,
            nn.ReLU(inplace=True),
            nn.Conv2d(nf// 4, nf, kernel_size=1) 
        )
        
        # self.propagation = MoETrans(dim=nf,t=8,num_heads=8)
        self.propagation = MOE(nf=nf)
        
        ### initialization ###
        initialize_weights([self.conv_first, self.upconv, self.HRconv, self.conv_last, self.memory, self.mlp, self.propagation], 0.1)
        # initialize_weights([self.conv_first, self.upconv, self.HRconv, self.conv_last, self.memory, self.mlp], 0.1)

    def forward(self, x, scene):  # BTCHW
        
        ### 记忆模块初始化 -------------------------------------- ###
        if self.training:
            self.memory.clear_test_memory(self.test_scene_list)
            self.test_scene_list = set()
        
        b,t,c,h,w = x.shape
        x = x.reshape(b*t,c,h,w)  # [t][t]...[t]
        
        fea = self.act(self.conv_first(x))  # [b*t,c,h,w]
        # out = self.recon_trunk_res(fea)
        # out = self.recon_trunk_fm(out)
        
        tmp = []
        for idx, layer in enumerate(self.recon_trunk_res):  # [0,14]
            fea = layer(fea)
            if idx in [4,9,14]:
                bt2,c2,h2,w2 = fea.shape
                fea_tmp = fea.reshape(b,t,c2,h2,w2)
                tmp.append(fea_tmp)
        
        out = self.propagation(tmp)  # [b,t,c,h,w]*3
        
        
        # MoE模块
        # bt2,c2,h2,w2 = out.shape
        # out = out.reshape(b,t*c2,h2,w2)  # [b*t,c,h,w]
        # out = self.propagation(out)  # [b,t*c,h,w]
        # out = out.reshape(b,t,c2,h2,w2)  # [b,t,c,h,w]

        # 记忆模块
        # bt1,c1,h1,w1 = out.shape
        # out = out.reshape(b,t,c2,h2,w2)
        out = self.memory_forward(out, scene)  # [b,t,c,h,w]
        # out = out.reshape(bt1,c1,h1,w1)
        out = out.reshape(b*t,c2,h2,w2)
        
        out = self.act(self.upsampler(self.upconv(out)))
        out = self.conv_last(self.act(self.HRconv(out)))
        out = out.reshape(b,t,c,h,w)
        return out


    # ### 记忆模块处理 ###
    # def memory_forward(self, feats, scenes):  # batch个scene
    #     ### 场景分类 -------------------------------------------------
    #     n,t,c,h,w = feats.size()  # BTCHW

    #     # import pdb;pdb.set_trace()

    #     # new_scenes = []  # 新场景flag
    #     # for scene in scenes:
    #     #     if scene not in self.train_scene_list and scene not in self.test_scene_list:
    #     #         new_scenes.append(True)
    #     #     else:
    #     #         new_scenes.append(False)

    #     ### 记忆检索 -------------------------------------------------
    #     mem_res = []  # 检索结果 BTCHW
        
    #     # 逐帧处理，每次处理batch
    #     for i in range(t):
    #         batch_frame = feats[:,i,:,:,:]  # BCHW
    #         batch_frame = self.memory.match_mem(batch_frame, scenes) + batch_frame  # (b,c,h,w)
    #         batch_frame = self.mlp(batch_frame) + batch_frame  # (b,c,h,w)
    #         mem_res.append(batch_frame)
    #         # ---添加到记忆库
    #         mem_frame = self.memory.mem_prepare(torch.cat([feats[:,i,:,:,:],batch_frame],dim=1))  # (b,c,h,w)
    #         self.memory.add_memory(mem_frame.detach(), scenes)

    #         # ---更新场景
    #         if i==0:
    #             # new_scenes = [False] * len(new_scenes)
    #             if self.training:
    #                 self.train_scene_list = self.train_scene_list | set(scenes)
    #             else:
    #                 self.test_scene_list = self.test_scene_list | set(scenes)
    #         del batch_frame,mem_frame
        
    #     mem_res = torch.stack(mem_res,dim=1)  # (b,t,c,h,w)        
    #     return mem_res


    ### 记忆模块处理 ###
    def memory_forward(self, feats, scenes):  # batch个scene
        # 场景分类 -------------------------------------------------
        new_scenes = []
        scene_initialized = []
        for scene in scenes:
            if scene not in self.train_scene_list and scene not in self.test_scene_list:
                new_scenes.append(True)
                scene_initialized.append(False)
            else:
                new_scenes.append(False)
                scene_initialized.append(True) 

        # 记忆检索 -------------------------------------------------
        n,d,c,h,w = feats.size()  # BTCHW
        
        # ---逐场景处理（batch）
        mem_batchs = []
        for i in range(n): 
            scene = scenes[i]  # 场景
            feat = feats[i]  # (d,c,h,w)
            mem_frames = []
            
            # ---逐帧处理 (d)
            for j in range(d):
                # ---特征增强
                frame = feat[j:j+1,...]  # (1,c,h,w)
                if scene_initialized[i]:  # 已有场景
                    frame_feat = self.memory.match_mem(frame, scene) + frame  # (1,c,h,w)
                elif new_scenes[i]: # 新场景
                    frame_feat = frame
                    scene_initialized[i] = True
                frame_feat = self.mlp(frame_feat) + frame_feat  # (1,c,h,w)
                mem_frames.append(frame_feat)  # 每帧结果
                # ---添加到记忆库
                # mem_frame = self.memory.mem_prepare(torch.cat([frame,frame_feat],dim=1))
                mem_frame = self.memory.mem_prepare(frame_feat)
                self.memory.add_memory(mem_frame.detach(),scene)
                # ---删除多余记忆
                self.memory.trim_memory(scene)
                del frame_feat

            mem_frames = torch.cat(mem_frames,dim=0)  # d,c,h,w
            mem_batchs.append(mem_frames)  # 每个场景结果
            del mem_frames

            # ---更新场景
            new_scenes[i] = False
            if self.training:
                self.train_scene_list = self.train_scene_list | set(scene)
            else:
                self.test_scene_list = self.test_scene_list | set(scene)
                
        mem_batchs = torch.stack(mem_batchs,dim=0)  # n,d,c,h,w
        return mem_batchs



