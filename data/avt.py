import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *


class AvtDataset(Dataset):
    def __init__(self, args, split='train', load_ref=False):
        self.args = args
        self.root_dir = args.datadir
        self.split = split
        downsample = args.imgScale_train if split=='train' else args.imgScale_test
        assert int(640*downsample)%32 == 0 or int(480*downsample)%32 == 0, \
            f'image width is {int(640*downsample)}, it should be divisible by 32, you may need to modify the imgScale'
        self.img_wh = (int(640*downsample),int(480*downsample))
        self.define_transforms()


        self.blender2opencv = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        if not load_ref:
            self.read_meta()

        self.white_back = False

    def read_meta(self):
        with open(os.path.join(self.root_dir, f"transforms.json"), 'r') as f:
            self.meta = json.load(f)

        aa = np.arange(len(self.meta['frames']))

        test_list = aa[::8]
        train_list = aa
        self.img_idx = train_list if self.split=='train' else test_list
        self.meta['frames'] = [self.meta['frames'][idx] for idx in self.img_idx]
        print(f'===> {self.split}ing index: {self.img_idx}')

        w, h = self.img_wh
        # self.focal = 0.5 * 640 / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length
        self.focal = self.meta['fx']
        self.focal *= self.img_wh[0] / 640  # modify focal length to match size self.img_wh

        # bounds, common for all scenes
        self.near = 0.1
        self.far = 3.0
        self.bounds = np.array([self.near, self.far])

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal,self.focal])  # (h, w, 3)


        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        for frame in self.meta['frames']:
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            self.poses += [pose]
            c2w = torch.FloatTensor(pose)

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            self.image_paths += [image_path]
            img = Image.open(image_path)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
            self.all_masks += [img[:,-1:]>0]
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            self.all_rgbs += [img]

            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)

            self.all_rays += [torch.cat([rays_o, rays_d,
                                         self.near * torch.ones_like(rays_o[:, :1]),
                                         self.far * torch.ones_like(rays_o[:, :1])],
                                        1)]  # (h*w, 8)
            self.all_masks += []


        self.poses = np.stack(self.poses)
        if 'train' == self.split:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
            self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)

    def read_source_views(self, file=f"transforms.json", pair_idx=None, device=torch.device("cpu")):
        with open(os.path.join(self.root_dir, file), 'r') as f:
            meta = json.load(f)

        w, h = self.img_wh
        # focal = 0.5 * 800 / np.tan(0.5 * meta['camera_angle_x'])  # original focal length
        focal = meta['fx']
        focal *= self.img_wh[0] / 640  # modify focal length to match size self.img_wh

        src_transform = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # if do not specify source views, load index from pairing file
        if pair_idx is None:
            aa = np.arange(len(meta['frames']))
            train_list = aa[[2,41,61]]
            pair_idx = train_list
            print(f'====> ref idx: {pair_idx}')

        imgs, proj_mats = [], []
        intrinsics, c2ws, w2cs = [],[],[]
        for i,idx in enumerate(pair_idx):
            frame = meta['frames'][idx]
            c2w = np.array(frame['transform_matrix']) @ self.blender2opencv
            w2c = np.linalg.inv(c2w)
            c2ws.append(c2w)
            w2cs.append(w2c)

            # build proj mat from source views to ref view
            proj_mat_l = np.eye(4)
            intrinsic = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])
            intrinsics.append(intrinsic.copy())
            intrinsic[:2] = intrinsic[:2] / 4  # 4 times downscale in the feature space
            proj_mat_l[:3, :4] = intrinsic @ w2c[:3, :4]
            if i == 0:  # reference view
                ref_proj_inv = np.linalg.inv(proj_mat_l)
                proj_mats += [np.eye(4)]
            else:
                proj_mats += [proj_mat_l @ ref_proj_inv]

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            img = Image.open(image_path)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            # img = img[:3] * img[-1:] + (1 - img[-1:])  # blend A to RGB
            img = img[:3]
            imgs.append(src_transform(img))

        pose_source = {}
        pose_source['c2ws'] = torch.from_numpy(np.stack(c2ws)).float().to(device)
        pose_source['w2cs'] = torch.from_numpy(np.stack(w2cs)).float().to(device)
        pose_source['intrinsics'] = torch.from_numpy(np.stack(intrinsics)).float().to(device)

        near_far_source = [0.1,3.]
        imgs = torch.stack(imgs).float().unsqueeze(0).to(device)
        proj_mats = torch.from_numpy(np.stack(proj_mats)[:,:3]).float().unsqueeze(0).to(device)
        return imgs, proj_mats, near_far_source, pose_source

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        if self.split == 'train':  # use data in the buffers
            # view, ray_idx = torch.randint(0,len(self.all_rays),(1,)), torch.randperm(self.all_rays.shape[1])[:self.args.batch_size]
            # sample = {'rays': self.all_rays[view,ray_idx],
            #           'rgbs': self.all_rgbs[view,ray_idx]}
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else:  # create data for each image separately
            # frame = self.meta['frames'][idx]
            # c2w = torch.FloatTensor(frame['transform_matrix']) @ self.blender2opencv

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            mask = self.all_masks[idx] # for quantity evaluation

            sample = {'rays': rays,
                      'rgbs': img,
                      'mask': mask}
        return sample