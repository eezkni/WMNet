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
from functools import reduce
from operator import mul



# def window_partition(x, window_size):
#     """ Partition the input into windows. Attention will be conducted within the windows.

#     Args:
#         x: (B, D, H, W, C)
#         window_size (tuple[int]): window size

#     Returns:
#         windows: (B*num_windows, window_size*window_size, C)
#     """
#     BT, H, W, C = x.shape
#     x = x.view(BT, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
#     windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, reduce(mul, window_size), C)  # 
#     # windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)

#     return windows

# def compute_mask(H, W, window_size, shift_size, device):
#     """ Compute attnetion mask for input of size (H, W). @lru_cache caches each stage results. """

#     img_mask = torch.zeros((1, H, W, 1), device=device)  # [1,Hp,Wp,1]
#     cnt = 0
#     for h in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
#         for w in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
#             img_mask[:, h, w, :] = cnt
#             cnt += 1
#     mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
#     mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
#     attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
#     attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
#     return attn_mask




class MoETrans(nn.Module):
    """ Swin Transformer Layer (STL).
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for mutual and self attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm.
    """
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(8, 8),
                 shift_size=(0, 0),
                 qkv_bias=True,
                 qk_scale=None,
                 norm_layer=nn.LayerNorm,
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias,
                                    qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        self.moe = SparseMoeBlock(embed_dim=dim, mlp_ratio=16, num_experts=4, num_experts_per_tok=2)

    def forward(self, x):  # [bt,c,h,w]
        bt, c, h, w = x.shape
        x = x.reshape(bt, h, w, c)  # [bt,h,w,c]
        
        x = x + self.forward_part1(x)        
        x = x + self.forward_part2(x)

        x = x.reshape(bt, c, h, w)  # [bt,c,h,w]
        return x

        
    def forward_part1(self, x):  # [bt,h,w,c]
        BT, H, W, C = x.shape

        # pad features
        pad_l = pad_t = 0
        pad_r = (self.window_size[0] - W % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0), mode='reflect')
        _, Hp, Wp, _ = x.shape


        x = self.norm1(x)  # [bt,h,w,c]


        # cyclic shift
        if any(i > 0 for i in self.shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
            attn_mask = compute_mask(Hp, Wp, self.window_size, self.shift_size, x.device)
            # attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C

        # attention / shifted attention
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C

        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C

        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :]

        return x







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
        self.max_len = 2
        self.train_scene_list = set()  # 训练数据场景
        self.test_scene_list = set()  # 测试数据场景
        self.memory = Memory(dim=nf, max_len=self.max_len)

        self.mlp = nn.Sequential(
            nn.Conv2d(nf, nf // 4, kernel_size=1) ,
            nn.ReLU(inplace=True),
            nn.Conv2d(nf// 4, nf, kernel_size=1) 
        )
        
        self.propagation = MoETrans(nf,num_heads=8,window_size=(8, 8),shift_size=(0, 0))
        
        ### initialization ###
        initialize_weights([self.conv_first, self.upconv, self.HRconv, self.conv_last, self.memory, self.mlp], 0.1)

    def forward(self, x, scene):  # BTCHW
        
        ### 记忆模块初始化 -------------------------------------- ###
        if self.training:
            self.memory.clear_test_memory(self.test_scene_list)
            self.test_scene_list = set()
        
        b,t,c,h,w = x.shape
        x = x.reshape(b*t,c,h,w)
        
        fea = self.act(self.conv_first(x))
        out = self.recon_trunk_res(fea)  # [bt,c,h,w]
        # out = self.recon_trunk_fm(out)
        
        # MoE模块
        out = self.propagation(out)  # [bt,c,h,w]
        
        # 记忆模块
        bt1,c1,h1,w1 = out.shape
        out = out.reshape(b,t,c1,h1,w1)
        out = self.memory_forward(out, scene)
        out = out.reshape(bt1,c1,h1,w1)
        
        out = self.act(self.upsampler(self.upconv(out)))
        out = self.conv_last(self.act(self.HRconv(out)))
        out = out.reshape(b,t,c,h,w)
        return out


    ### 记忆模块处理 ###
    def memory_forward(self, feats, scenes):  # batch个scene
        ### 场景分类 -------------------------------------------------
        n,t,c,h,w = feats.size()  # BTCHW

        # import pdb;pdb.set_trace()

        # new_scenes = []  # 新场景flag
        # for scene in scenes:
        #     if scene not in self.train_scene_list and scene not in self.test_scene_list:
        #         new_scenes.append(True)
        #     else:
        #         new_scenes.append(False)

        ### 记忆检索 -------------------------------------------------
        mem_res = []  # 检索结果 BTCHW
        
        # 逐帧处理，每次处理batch
        for i in range(t):
            batch_frame = feats[:,i,:,:,:]  # BCHW
            batch_frame = self.memory.match_mem(batch_frame, scenes) + batch_frame  # (b,c,h,w)
            batch_frame = self.mlp(batch_frame) + batch_frame  # (b,c,h,w)
            mem_res.append(batch_frame)
            # ---添加到记忆库
            mem_frame = self.memory.mem_prepare(torch.cat([feats[:,i,:,:,:],batch_frame],dim=1))  # (b,c,h,w)
            self.memory.add_memory(mem_frame.detach(), scenes)

            # ---更新场景
            if i==0:
                # new_scenes = [False] * len(new_scenes)
                if self.training:
                    self.train_scene_list = self.train_scene_list | set(scenes)
                else:
                    self.test_scene_list = self.test_scene_list | set(scenes)
            del batch_frame,mem_frame
        
        mem_res = torch.stack(mem_res,dim=1)  # (b,t,c,h,w)        
        return mem_res


    # ### 记忆模块处理 ###
    # def memory_forward(self, feats, scenes):  # batch个scene
    #     # 场景分类 -------------------------------------------------
    #     new_scenes = []
    #     scene_initialized = []
    #     for scene in scenes:
    #         if scene not in self.train_scene_list and scene not in self.test_scene_list:
    #             new_scenes.append(True)
    #             scene_initialized.append(False)
    #         else:
    #             new_scenes.append(False)
    #             scene_initialized.append(True) 

    #     # 记忆检索 -------------------------------------------------
    #     n,d,c,h,w = feats.size()  # BTCHW
        
    #     # ---逐场景处理（batch）
    #     mem_batchs = []
    #     for i in range(n): 
    #         scene = scenes[i]  # 场景
    #         feat = feats[i]  # (d,c,h,w)
    #         mem_frames = []
            
    #         # ---逐帧处理 (d)
    #         for j in range(d):
    #             # ---特征增强
    #             frame = feat[j:j+1,...]  # (1,c,h,w)
    #             if scene_initialized[i]:  # 已有场景
    #                 frame_feat = self.memory.match_mem(frame, scene) + frame  # (1,c,h,w)
    #             elif new_scenes[i]: # 新场景
    #                 frame_feat = frame
    #                 scene_initialized[i] = True
    #             frame_feat = self.mlp(frame_feat) + frame_feat  # (1,c,h,w)
    #             mem_frames.append(frame_feat)  # 每帧结果
    #             # ---添加到记忆库
    #             mem_frame = self.memory.mem_prepare(torch.cat([frame,frame_feat],dim=1))
    #             self.memory.add_memory(mem_frame.detach(),scene)
    #             # ---删除多余记忆
    #             self.memory.trim_memory(scene)
    #             del frame_feat

    #         mem_frames = torch.cat(mem_frames,dim=0)  # d,c,h,w
    #         mem_batchs.append(mem_frames)  # 每个场景结果
    #         del mem_frames

    #         # ---更新场景
    #         new_scenes[i] = False
    #         if self.training:
    #             self.train_scene_list = self.train_scene_list | set(scene)
    #         else:
    #             self.test_scene_list = self.test_scene_list | set(scene)
                
    #     mem_batchs = torch.stack(mem_batchs,dim=0)  # n,d,c,h,w
    #     return mem_batchs



