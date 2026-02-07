import random
import numpy as np
import torch
import torch.utils.data as data
import data.util as util
import os

### 指定文件夹，每次加载文件夹中的多张图像

class LQGT_dataset(data.Dataset):
    '''
    Read LQ (Low Quality, here is LR) and GT image pairs.
    If only GT image is provided, generate LQ image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(LQGT_dataset, self).__init__()
        self.opt = opt
        self.frames = 8
        # self.data_type = self.opt['data_type']
        # self.paths_LQ, self.paths_GT = None, None
        # self.sizes_LQ, self.sizes_GT = None, None
        # self.LQ_env, self.GT_env = None, None  # environment for lmdb

        self.gt_subfolders = sorted([os.path.join(opt['dataroot_GT'], subfolder) for subfolder in os.listdir(opt['dataroot_GT'])
                                     if os.path.isdir(os.path.join(opt['dataroot_GT'], subfolder))])
        self.lq_subfolders = sorted([os.path.join(opt['dataroot_LQ'], subfolder) for subfolder in os.listdir(opt['dataroot_LQ'])
                                     if os.path.isdir(os.path.join(opt['dataroot_LQ'], subfolder))])
        assert len(self.gt_subfolders) == len(self.lq_subfolders)

        # self.sizes_GT, self.paths_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])
        # self.sizes_LQ, self.paths_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])
        # assert self.paths_GT, 'Error: GT path is empty.'

        # if self.paths_LQ and self.paths_GT:
        #     assert len(self.paths_LQ) == len(
        #         self.paths_GT
        #     ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
        #         len(self.paths_LQ), len(self.paths_GT))

    def __getitem__(self, index):
        # GT_path, LQ_path = None, None
        
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        gt_subfolder = self.gt_subfolders[index]
        lq_subfolder = self.lq_subfolders[index]

        gt_img_paths = sorted([os.path.join(gt_subfolder, filename) for filename in os.listdir(gt_subfolder)
                               if filename.lower().endswith(('png', 'jpg', 'jpeg'))])
        lq_img_paths = sorted([os.path.join(lq_subfolder, filename) for filename in os.listdir(lq_subfolder)
                               if filename.lower().endswith(('png', 'jpg', 'jpeg'))])

        if self.opt['phase'] == 'train':
            assert len(gt_img_paths) >= self.frames, f"GT subfolder {gt_subfolder} does not have enough images!"
            assert len(lq_img_paths) >= self.frames, f"LQ subfolder {lq_subfolder} does not have enough images!"

            start_index = random.randint(0, len(gt_img_paths) - self.frames)

            selected_gt_img_paths = gt_img_paths[start_index:start_index+self.frames]
            selected_lq_img_paths = lq_img_paths[start_index:start_index+self.frames]

            gt_images = [util.read_img(None, img_path) for img_path in selected_gt_img_paths]  # 0-1 HWC BGR
            lq_images = [util.read_img(None, img_path) for img_path in selected_lq_img_paths]

            gt_images, lq_images = util.paired_random_crop(gt_images, lq_images, GT_size, scale)
            
            # import pdb;pdb.set_trace()
            
            # lq_images.append(gt_images)
            images = lq_images+gt_images
            images = util.augment(images, self.opt['use_flip'],self.opt['use_rot'])
            images = util.img2tensor(images)  # 0-1 CHW RGB

            img_gts = torch.stack(images[len(images) // 2:], dim=0)
            img_lqs = torch.stack(images[:len(images) // 2], dim=0)

        else:
            gt_images = [util.read_img(None, img_path) for img_path in gt_img_paths]  # 0-1 HWC BGR
            lq_images = [util.read_img(None, img_path) for img_path in lq_img_paths]
            
            gt_images = util.img2tensor(gt_images) 
            lq_images = util.img2tensor(lq_images) 
            img_gts = torch.stack(gt_images, dim=0)
            img_lqs = torch.stack(lq_images, dim=0)

        return {'LQ': img_lqs, 'GT': img_gts, 'LQ_path': lq_subfolder, 'GT_path': gt_subfolder}


        # ### get GT image
        # GT_path = self.paths_GT[index]
        # # 
        # img_GT = util.read_img(self.GT_env, GT_path)  # 从文件夹中读取图像

        # if self.opt['color']:
        #     img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]

        # ### get LQ image
        # if self.paths_LQ:
        #     LQ_path = self.paths_LQ[index]
        #     img_LQ = util.read_img(self.LQ_env, LQ_path)

        # if self.opt['phase'] == 'train':
            
        #     H, W, C = img_LQ.shape
        #     H_gt, W_gt, C = img_GT.shape
        #     if H != H_gt:
        #         print('*******wrong image*******:{}'.format(LQ_path))
        #     LQ_size = GT_size // scale

        #     # randomly crop
        #     if GT_size is not None:
        #         rnd_h = random.randint(0, max(0, H - LQ_size))
        #         rnd_w = random.randint(0, max(0, W - LQ_size))
        #         img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
        #         rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
        #         img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

        #     # augmentation - flip, rotate
        #     img_LQ, img_GT = util.augment([img_LQ, img_GT], self.opt['use_flip'],
        #                                   self.opt['use_rot'])

        # # BGR to RGB, HWC to CHW, numpy to tensor
        # if img_GT.shape[2] == 3:
        #     img_GT = img_GT[:, :, [2, 1, 0]]
        #     img_LQ = img_LQ[:, :, [2, 1, 0]]

        # H, W, _ = img_LQ.shape

        # img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        # img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()

        # if LQ_path is None:
        #     LQ_path = GT_path
        # return {'LQ': img_LQ, 'GT': img_GT, 'LQ_path': LQ_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.gt_subfolders)
