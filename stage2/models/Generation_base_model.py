import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
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

        total = sum(p.numel() for p in self.netG.parameters() if p.requires_grad)
        print('******** Params: ', total)

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
            # self.ssim_loss = pytorch_ssim.SSIM(window_size = 11)

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

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ  BTCHW
        self.scene = data['scene']  # list  [1,2,...,B]
        if need_GT:
            self.real_H = data['GT'].to(self.device)  # GT  BTCHW

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L, self.scene)  # pred  BTCHW
        
        test_results = []
        for i in range(self.real_H.shape[1]):
            ssim = 1 - pytorch_ssim.ssim(self.fake_H[:,i, ...], self.real_H[:,i, ...])
            test_results.append(ssim)
        
        l_ssim = sum(test_results) / len(test_results) 
        
        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        loss = l_ssim + l_pix
        
        loss.backward()
        # l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()
        self.log_dict['l_ssim'] = l_ssim.item()

    def test(self):
        self.netG.eval()
        
        # import pdb;pdb.set_trace()
        
        ### 图像尺寸较大，逐张输入
        # with torch.no_grad():
        #     fake_frames = []
        #     for t in range(self.var_L.size(1)):
        #         single_frame = self.var_L[:, t:t+1, :, :, :] 
        #         fake_single_frame = self.netG(single_frame, self.scene)  # [1]
        #         fake_frames.append(fake_single_frame)
        #         del fake_single_frame
        #     self.fake_H = torch.cat(fake_frames, dim=1)
        #     del fake_frames
        #     # self.fake_H = self.netG(self.var_L)

        ### 先划分长度，再划分patch
        with torch.no_grad():
            b, t, c, h, w = self.var_L.size()
            frame_num = 8
            frame_overlap = 2
            stride = frame_num - frame_overlap

            d_idx_list = list(range(0, t-frame_num, stride)) + [max(0, t-frame_num)]
            E = torch.zeros(b, t, c, h, w).to(self.device)
            W = torch.zeros(b, t, 1, 1, 1).to(self.device)

            for d_idx in d_idx_list:
                lq_clip = self.var_L[:, d_idx:d_idx+frame_num, ...]
                out_clip = self._test_clip(lq_clip)
                out_clip_mask = torch.ones((b, min(frame_num, t), 1, 1, 1)).to(self.device)

                E[:, d_idx:d_idx+frame_num, ...].add_(out_clip)
                W[:, d_idx:d_idx+frame_num, ...].add_(out_clip_mask)
            self.fake_H = E.div_(W)
            del E,W,d_idx_list
        self.netG.train()


    def _test_clip(self, lq):
        b, t, c, h, w = lq.size()
        patch_size = 512
        overlap_size = 64
        stride = patch_size - overlap_size
    
        h_idx_list = list(range(0, h-patch_size, stride)) + [max(0, h-patch_size)]
        w_idx_list = list(range(0, w-patch_size, stride)) + [max(0, w-patch_size)]
        E = torch.zeros(b, t, c, h, w).to(self.device)
        W = torch.zeros_like(E).to(self.device)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = lq[..., h_idx:h_idx+patch_size, w_idx:w_idx+patch_size]
                out_patch = self.netG(in_patch,self.scene).detach()
                out_patch_mask = torch.ones_like(out_patch).to(self.device)

                E[..., h_idx:(h_idx+patch_size), w_idx:(w_idx+patch_size)].add_(out_patch)
                W[..., h_idx:(h_idx+patch_size), w_idx:(w_idx+patch_size)].add_(out_patch_mask)
        output = E.div_(W)
        return output


    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L[0]  # BTCHW >> TCHW
        out_dict['SR'] = self.fake_H[0]
        # out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        # out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        
        if need_GT:
            out_dict['GT'] = self.real_H[0]
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