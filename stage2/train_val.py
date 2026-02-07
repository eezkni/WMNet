import os
import math
import argparse
import random
import logging
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from utils import extra_util
from data import create_dataloader, create_dataset
from models import create_model

import numpy as np
import torch.nn.functional as F
import pytorch_ssim
from collections import OrderedDict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import cv2

os.environ["CUDA_VISIBLE_DEVICES"]="3"

def valid(val_loader, model, current_step, tb_logger_fig):
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    # vpbar = tqdm(total=len(val_loader), desc="Testing Progress", ncols=100)
    for idx, val_data in enumerate(val_loader):
    # for val_data in val_loader:  # 1TCHW
        model.feed_data(val_data)
        model.test()
        print("testing...")

        visuals = model.get_current_visuals()
        sr_img = visuals['SR']  # TCHW
        gt_img = visuals['GT']
        
        
        # import pdb;pdb.set_trace()
        
        input_folder = val_data['LQ_path'][0]
        lq_img_paths = sorted([os.path.join(input_folder, filename) for filename in os.listdir(input_folder)
                        if filename.lower().endswith(('png', 'jpg', 'jpeg'))])
        
        
        for i in range(sr_img.shape[0]):  # CHW
            mse =  F.mse_loss(sr_img[i:i+1, ...].clamp_(0, 1), gt_img[i:i+1, ...].clamp_(0, 1))
            psnr = (20 * torch.log10(1.0 / torch.sqrt(mse)))
            ssim = pytorch_ssim.ssim(sr_img[i:i+1, ...].clamp_(0, 1), gt_img[i:i+1, ...].clamp_(0, 1))
            print(psnr.item(),",",ssim.item())

            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            
            
            
            # 保存图像
            sr = sr_img[i:i+1, ...].squeeze().cpu().numpy()  # CHW,0-1,RGB
            sr = np.transpose(sr,(1,2,0))
            sr = np.uint16(np.clip(sr*65535,0,65535))
            sr = sr[..., ::-1]
                        
            input_path = lq_img_paths[i]
            input_name = os.path.basename(input_path)
            folder_name = os.path.basename(input_folder)
            out_folder = os.path.join("/remote-home/nizhangkai/zy/24_KAIR/Final-stage2-best/Results/video_new", folder_name, input_name)
            dir_path = os.path.dirname(out_folder)
            os.makedirs(dir_path, exist_ok=True)
            # sr_pil.save(out_folder, format='PNG')
            # import pdb;pdb.set_trace()
            cv2.imwrite(out_folder,sr)


        # if idx % 5  == 0:
        #     test_img = torch.cat([visuals['LQ'][0], visuals['SR'][0], visuals['GT'][0]], dim=-2)
        #     tb_logger_fig.add_image(f'valid/{idx}',test_img, current_step, dataformats='CHW')
        #     del test_img
            
        del val_data,visuals,sr_img,gt_img
        # vpbar.update(1)
        
    avg_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    avg_ssim = sum(test_results['ssim']) / len(test_results['ssim']) 
    return avg_psnr, avg_ssim, 0

def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default="options/train/fmnet_final.yml", help='Path to option YMAL file.')
    parser.add_argument('--debug', type=bool, default=False, help='whether to perform debug mode for VSCode')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)


    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    if parser.parse_args().debug == True:  # False
        opt['gpu_ids'] = [0]
        opt['dist'] = False
        opt['datasets']['train']['n_workers'] = 0
        opt['datasets']['train']['batch_size'] = 2


    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None


    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO, screen=False, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO, screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        logger_val = logging.getLogger('val')
        
        # tensorboard logger
        tb_logger_cur = SummaryWriter(log_dir=os.path.join(opt['path']['log'], 'tb_logger_cur', opt['name']))
        tb_logger_fig = SummaryWriter(log_dir=os.path.join(opt['path']['log'], 'tb_logger_fig', opt['name']))
            #tb_logger = SummaryWriter(log_dir=  + '/tb_logger/' + opt['name'])
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)


    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():   
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None


    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    best_psnr = 0.0
    best_psnr_iter = 0
    best_ssim = 0.0
    best_ssim_iter = 0

    if_break = False
    # if current_step < total_iters:
    #     #### training
    #     logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    #     first_time = True
        
    #     pbar = tqdm(total=total_iters, desc="Training Progress", ncols=100, initial=current_step)
        
    #     for epoch in range(start_epoch, total_epochs + 1):
    #         if opt['dist']:
    #             train_sampler.set_epoch(epoch)
    #         for _, train_data in enumerate(train_loader):
    #             if first_time:
    #                 start_time = time.time()
    #                 first_time = False
    #             current_step += 1
    #             if current_step > total_iters:
    #                 if_break = True
    #                 break
                                
    #             #### training
    #             model.feed_data(train_data)
    #             model.optimize_parameters(current_step)
                
    #             #### update learning rate
    #             model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

    #             #### log
    #             if current_step % opt['logger']['print_freq'] == 0:
    #                 end_time = time.time()
    #                 logs = model.get_current_log()
    #                 message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, time:{:.3f}> '.format(
    #                     epoch, current_step, model.get_current_learning_rate(), end_time-start_time)
    #                 for k, v in logs.items():
    #                     message += '{:s}: {:.4e} '.format(k, v)
    #                     # tensorboard logger
    #                     if opt['use_tb_logger'] and 'debug' not in opt['name']:
    #                         if rank <= 0:
    #                             tb_logger_cur.add_scalar(k, v, current_step)
    #                 if rank <= 0:
    #                     logger.info(message)
    #                 start_time = time.time()

    #             #### save models and training states
    #             if current_step % opt['logger']['save_checkpoint_freq'] == 0:
    #                 if rank <= 0:
    #                     logger.info('Saving models and training states.')
    #                     model.save(current_step)
    #                     model.save_training_state(epoch, current_step)

    #             # validation
    #             if current_step % opt['train']['val_freq'] == 0 and rank <= 0:
    #                 avg_psnr, avg_ssim, avg_deltaITP = valid(val_loader, model, current_step, tb_logger_fig)
                    
    #                 # log
    #                 logger.info('# Validation # PSNR: {:.4f} # SSIM: {:.4f} # deltaITP: {:.4f}'.format(avg_psnr, avg_ssim, avg_deltaITP))
    #                 # tensorboard logger
    #                 tb_logger_cur.add_scalar('psnr', avg_psnr, current_step)
    #                 tb_logger_cur.add_scalar('ssim', avg_ssim, current_step)
                    
    #                 if avg_psnr > best_psnr:
    #                     best_psnr = avg_psnr
    #                     best_psnr_iter = current_step
    #                 if avg_ssim > best_ssim:
    #                     best_ssim = avg_ssim
    #                     best_ssim_iter = current_step
                    
    #                 # logger_val = logging.getLogger('val')  # validation logger
    #                 logger_val.info('# PSNR: {:.4f} <iter:{:8,d}> # SSIM: {:.4f} <iter:{:8,d}>.'.format(best_psnr, best_psnr_iter, best_ssim, best_ssim_iter))
    #             pbar.update(1)

    #         if if_break == True:
    #             break

    #     if rank <= 0:
    #         model_parameters = filter(lambda p: p.requires_grad, model.netG.parameters())
    #         params = int(sum([np.prod(p.size()) for p in model_parameters]))
    #         logger.info('Params: {:3.4f} [M]'.format((params / 1024**2)))
    #         logger.info('Saving the final model.')
    #         model.save('latest')
    #         logger.info('End of training.')
    # else:
    #     avg_psnr, avg_ssim, avg_deltaITP = valid(val_loader, model, current_step, tb_logger_fig)
    #     # logger_val = logging.getLogger('val')  # validation logger
    #     logger_val.info('<epoch:{:3d}, iter:{:8,d}> PSNR: {:.4f} # SSIM: {:.4f} # deltaITP: {:.4f}'.format(start_epoch, current_step, avg_psnr, avg_ssim, avg_deltaITP))
    avg_psnr, avg_ssim, avg_deltaITP = valid(val_loader, model, current_step, tb_logger_fig)
    # logger_val = logging.getLogger('val')  # validation logger
    logger_val.info('<epoch:{:3d}, iter:{:8,d}> PSNR: {:.4f} # SSIM: {:.4f} # deltaITP: {:.4f}'.format(start_epoch, current_step, avg_psnr, avg_ssim, avg_deltaITP))

if __name__ == '__main__':
    main()