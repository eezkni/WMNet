import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from utils.util import DWT,IWT
import pytorch_ssim

logger = logging.getLogger('base')

class GenerationModel(BaseModel):
    def __init__(self, opt):
        super(GenerationModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = self.netG # self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            # loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        ### 掩码相关
        self.dwt, self.idwt = DWT(), IWT()
        self.maskratio = train_opt['maskratio_min']  # 0.0
        self.maskratio_max = train_opt['maskratio_max']  # 0.5
        self.maskratio_inc = train_opt['maskratio_increase']  # 0.01
        self.maskratio_interval = train_opt['niter'] / (self.maskratio_max/self.maskratio_inc + 1)


    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ  BTCHW
        
        if need_GT:
            self.real_H = data['GT'].to(self.device)  # GT  BTCHW


    def optimize_parameters(self, step):  # BTCHW
        self.optimizer_G.zero_grad()
        
        ### 掩码操作 (随机掩码) --------------------------------------###
        #-- 更新基础掩码率
        if step % self.maskratio_interval  == 0:  
            self.maskratio += self.maskratio_inc
            
        # #-- 小波变换
        # lq_dwt = self.dwt(self.var_L)
        # lq_L = lq_dwt[:,:,:3,:,:]  # [0,2]
        # lq_H = lq_dwt[:,:,3:,:,:]  # [-1,1]
        # #-- 高频全零掩码
        # lq_H_mask = torch.zeros_like(lq_H)  
        # #-- 低频随机掩码
        # if self.maskratio>0:  
        #     lq_L_mask = self.random_mask(lq_L, 1, self.maskratio)
        #     lq_mask = self.idwt(torch.cat((lq_L_mask,lq_H_mask),dim=2)).detach()
        # else:
        #     lq_L_mask = lq_L
        #     lq_mask = self.idwt(torch.cat((lq_L_mask,lq_H_mask),dim=2)).detach()
        # # lq_L_mask = lq_L
        # # lq_mask = self.idwt(torch.cat((lq_L_mask,lq_H_mask),dim=2)).detach()
        # #-- 逆小波变换
        
        #-- 小波变换1
        lq_dwt = self.dwt(self.var_L)
        lq_L = lq_dwt[:,:,:3,:,:]  # [0,2]
        lq_H = lq_dwt[:,:,3:,:,:]  # [-1,1]
        
        #-- 小波变换2
        lq_dwt2 = self.dwt(lq_L)
        lq_LL = lq_dwt2[:,:,:3,:,:]  # [0,2]
        lq_HH = lq_dwt2[:,:,3:,:,:]  # [-1,1]
        
        #-- 小波变换3
        lq_dwt3 = self.dwt(lq_LL)
        lq_LLL = lq_dwt3[:,:,:3,:,:]  # [0,2]
        lq_HHH = lq_dwt3[:,:,3:,:,:]  # [-1,1]
        
        # 随机掩码
        if self.maskratio>0:  
            lq_LLL_mask = self.random_mask(lq_LLL, 1, self.maskratio)
        else:
            lq_LLL_mask = lq_LLL
        
        lq_HHH_mask = torch.zeros_like(lq_HHH)  
        lq_LL_mask = self.idwt(torch.cat((lq_LLL_mask,lq_HHH_mask),dim=2))
        
        lq_HH_mask = torch.zeros_like(lq_HH)  
        lq_L_mask = self.idwt(torch.cat((lq_LL_mask,lq_HH_mask),dim=2))
        
        lq_H_mask = torch.zeros_like(lq_H)  
        lq_mask = self.idwt(torch.cat((lq_L_mask,lq_H_mask),dim=2))
        
        self.var_L = lq_mask.detach()
        
        
        ### 自重建 --------------------------------------###
        self.fake_H = self.netG(self.var_L)  # pred  BTCHW
        
        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        
        # ### SSIM损失
        # test_results = []
        # for i in range(self.real_H.shape[1]):
        #     ssim = 1 - pytorch_ssim.ssim(self.fake_H[:,i, ...], self.real_H[:,i, ...])
        #     test_results.append(ssim)
        
        # l_ssim = sum(test_results) / len(test_results) 
        # loss = l_ssim + l_pix
        # loss.backward()
        
        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()
        # self.log_dict['l_ssim'] = l_ssim.item()


    def random_mask(self, x, patch, maskratio):  # 掩码大小,掩码率
        B, T, C, H, W = x.shape
        patches_h = H // patch
        patches_w = W // patch
        total_patches = patches_h * patches_w
        num_masked_patches = int(total_patches * maskratio)

        mask_indices = torch.randint(0, total_patches, (B, T, num_masked_patches), device=x.device) 
        mask = torch.ones(B, T, C, H, W, device=x.device)

        for b in range(B):
            for t in range(T):
                # 对每个时间步和每个图像应用不同的掩码
                patch_indices = mask_indices[b, t]  # 获取该时间步和该图像的掩码位置
                
                patch_y = torch.div(patch_indices, patches_w, rounding_mode='floor') * patch
                patch_x = (patch_indices % patches_w) * patch
                
                for idx in range(num_masked_patches):
                    mask[b, t, :, patch_y[idx]:patch_y[idx] + patch, patch_x[idx]:patch_x[idx] + patch] = 0
        x_masked = x * mask
        return x_masked


    def test(self):  # 1TCHW
        self.netG.eval()
        
        # ### 掩码操作
        # #-- 小波变换
        # lq_dwt = self.dwt(self.var_L)
        # lq_L = lq_dwt[:,:,:3,:,:]  # [0,2]
        # lq_H = lq_dwt[:,:,3:,:,:]  # [-1,1]
        # #-- 高频全零掩码
        # lq_H_mask = torch.zeros_like(lq_H)
        # #-- 低频随机掩码
        # lq_L_mask = self.random_mask(lq_L, 1, self.maskratio_max)
        # lq_mask = self.idwt(torch.cat((lq_L_mask,lq_H_mask),dim=2)).detach()
        # # lq_L_mask = lq_L
        # # lq_mask = self.idwt(torch.cat((lq_L_mask,lq_H_mask),dim=2)).detach()
        
        # self.var_L = lq_mask
        
        ### 图像尺寸较大，逐张输入
        with torch.no_grad():
            fake_frames = []
            for t in range(self.var_L.size(1)):
                single_frame = self.var_L[:, t:t+1, :, :, :] 
                fake_single_frame = self.netG(single_frame) 
                fake_frames.append(fake_single_frame)
            self.fake_H = torch.cat(fake_frames, dim=1)
            # self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L[0].detach()  # BTCHW >> TCHW
        out_dict['SR'] = self.fake_H[0].detach()
        # out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        # out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        
        if need_GT:
            out_dict['GT'] = self.real_H[0].detach()
            # out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)