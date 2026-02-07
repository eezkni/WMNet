import torch
import torch.nn as nn
from einops import rearrange
# from einops.layers.torch import Rearrange
import torch.nn.functional as F

# class Memory(nn.Module):
#     def __init__(self, dim, max_len):
#         super().__init__()
#         self.mem = {}
#         self.max_len = max_len
#         dim2 = dim//2
        
#         self.mk = nn.Linear(dim2, dim2, bias=False)
#         self.mv = nn.Linear(dim2, dim2, bias=False)
#         self.mq = nn.Linear(dim2, dim2, bias=False)
#         self.proj = nn.Linear(dim2, dim)

#         ### 用于记忆检索 ----------------------------------------------------------
#         self.feat_prepare = nn.Sequential(
#             nn.Conv2d(dim, dim2, kernel_size=(3, 3), padding=(1, 1), groups=8),  # (1,64,64,64) 
#             nn.ReLU(inplace=True),
#         )

#         ### 用于记忆更新 ----------------------------------------------------------
#         self.mem_prepare = nn.Sequential(
#             nn.Conv2d(dim*2, dim2, kernel_size=(3, 3), padding=(1, 1), groups=8),  # (d,96,256,256)
#             nn.ReLU(inplace=True),
#         )

#     # 记忆检索
#     def match_mem(self, feat, scenes):  # [bchw], [b]
#         # import pdb;pdb.set_trace()
        
#         # 卷积降维
#         feat = self.feat_prepare(feat)  # (b,c,h,w)
#         b, c, h, w = feat.shape
        
#         # values = [self.mem.get(scene, torch.zeros((c,h*w,self.max_len)).cuda()) for scene in scenes]
#         values = [self.mem.get(scene, feat[idx].reshape(c,-1,1).repeat(1, 1, self.max_len)) for idx, scene in enumerate(scenes)]
#         values = torch.stack(values, dim=0).cuda()  # bcsl
        
#         q = self.mq(rearrange(feat, 'b c h w -> b (h w) c'))  # (b,hw,c)
#         k = self.mk(rearrange(values, 'b c s l -> b l s c'))  # b,l,hw,c    (b,c=64,s=h*w,len=30)
#         v = self.mv(rearrange(feat, 'b c h w -> b (h w) c')) # b,hw,c

#         attn = torch.einsum('b s c, b l s c -> b l s', q, k)  # b,l,hw
#         attn = F.softmax(attn, dim=-1)  # b,l,hw (对h和w归一化)
#         attn = torch.einsum('b l s, d s c -> b s c', attn, v)  # b,hw,c

#         out = self.proj(attn).permute(0,2,1)
#         out = out.view(b, c*2, h, w) # 1,96,32,32

#         # mask = torch.tensor(new_scenes, dtype=torch.float32).cuda()  # 转换为浮点数张量 (b,)
#         # mask = 1 - mask  # True -> 0, False -> 1
#         # mask = mask.view(-1, 1, 1, 1)

#         # out = out * mask
#         return out

#     # 添加记忆
#     def add_memory(self, mem_frame, scenes):
#         # mem_frame = mem_frame.flatten(start_dim=2)  # b,c,h,w -> b,c,hw 

#         for idx, scene in enumerate(scenes):
#             if scene not in self.mem:
#                 cur_frame = mem_frame[idx].flatten(start_dim=1)  # c,hw
#                 self.mem[scene] = cur_frame.unsqueeze(-1).repeat(1, 1, self.max_len)  # c,hw,l
#             else:
#                 cur_frame = mem_frame[idx].flatten(start_dim=1) 
#                 self.mem[scene] = torch.cat([self.mem[scene], cur_frame.unsqueeze(-1)], dim=-1)  # c,hw,l
#                 self.mem[scene] = self.mem[scene][..., -self.max_len:]

#     # 清除记忆
#     def clear_test_memory(self, test_scene_list):
#         for test_scene in test_scene_list:
#             self.mem.pop(test_scene, None)


class Memory(nn.Module):
    def __init__(self, dim, max_len):
        super().__init__()
        self.mem = {}
        self.max_len = max_len
        dim2 = dim//2
        
        self.mk = nn.Linear(dim2, dim2, bias=False)
        self.mv = nn.Linear(dim2, dim2, bias=False)
        self.mq = nn.Linear(dim2, dim2, bias=False)
        self.proj = nn.Linear(dim2, dim)

        ### 用于记忆检索 ----------------------------------------------------------
        self.feat_prepare = nn.Sequential(
            nn.Conv2d(dim, dim2, kernel_size=(3, 3), padding=(1, 1), groups=8),  # (1,64,64,64) 
            nn.ReLU(inplace=True),
        )

        ### 用于记忆更新 ----------------------------------------------------------
        # self.mem_prepare = nn.Sequential(
        #     nn.Conv2d(dim*2, dim2, kernel_size=(3, 3), padding=(1, 1), groups=8),  # (d,96,256,256)
        #     nn.ReLU(inplace=True),
        # )
        self.mem_prepare = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), groups=8),  # (d,96,256,256)
            # nn.Conv2d(dim*2, dim2, kernel_size=(3, 3), padding=(1, 1), groups=8),  # (d,96,256,256)
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim2, kernel_size=(3, 3), padding=(1, 1), groups=8),
        )

    # 记忆检索
    def match_mem(self, feat, scene):
        
        # 卷积降维
        feat = self.feat_prepare(feat)  # (1,c//2,h,w)
        d, c, h, w = feat.shape
        
        q = self.mq(rearrange(feat, 'd c h w -> d (h w) c')) # 1,hw,c//2
        k = self.mk(rearrange(self.mem[scene], 'd c s l -> d l s c')) # 1,l,hw,c//2    (1,c=64,s=h*w,len=30)
        v = self.mv(rearrange(feat, 'd c h w -> d (h w) c')) # 1,hw,c//2

        attn = torch.einsum('d s c, d l s c -> d l s', q, k)
        attn = F.softmax(attn, dim=-1)  # d,l,s (对h和w归一化)
        attn = torch.einsum('d l s, d s c -> d s c', attn, v)  # 1,hw,c//2

        # import pdb;pdb.set_trace()

        out = self.proj(attn).permute(0,2,1)
        out = out.view(d, c*2, h, w) # (1,c,h,w)
        return out

    # 添加记忆
    def add_memory(self, mem_feat, scene):
        # import pdb;pdb.set_trace()
        mem_feat = mem_feat.flatten(start_dim=2) # 1 c h w -> 1 c hw 
        # self.tmp_mem[scene] = torch.stack(self.tmp_mem[scene], dim=-1)
        if scene not in self.mem:
            self.mem[scene] = mem_feat.unsqueeze(-1)
        else:
            self.mem[scene] = torch.cat([self.mem[scene], mem_feat.unsqueeze(-1)], dim=-1)

    # 删除记忆
    def trim_memory(self, scene):
        # 前面的扔掉，留下剩下的self.max_mem
        if self.mem[scene].size(-1) > self.max_len:
            self.mem[scene] = self.mem[scene][..., -self.max_len:]

    # 清除记忆
    def clear_test_memory(self, test_scene_list):
        for test_scene in test_scene_list:
            self.mem.pop(test_scene, None)

