from Sample import sample_rays_np
import numpy as np
import torch

def get_rgb_rays(images, poses, hwf,device):
    rays_o_list = list()
    rays_d_list = list()
    rays_rgb_list = list()
    H, W, focal = hwf
    for i in range(images.shape[0]):
        img = images[i]
        pose = poses[i]
        rays_o, rays_d = sample_rays_np(H, W, focal, pose)

        rays_o_list.append(rays_o.reshape(-1, 3))
        rays_d_list.append(rays_d.reshape(-1, 3))
        rays_rgb_list.append(img.reshape(-1, 3))

    rays_o_npy = np.concatenate(rays_o_list, axis=0)
    rays_d_npy = np.concatenate(rays_d_list, axis=0)
    rays_rgb_npy = np.concatenate(rays_rgb_list, axis=0)
    rays = torch.tensor(np.concatenate([rays_o_npy, rays_d_npy, rays_rgb_npy], axis=1), dtype=torch.float32,
                        device=device)
    return rays

def get_rays(poses, hwf,device):
    rays_o_list = list()
    rays_d_list = list()
    H, W, focal = hwf
    for i in range(poses.shape[0]):
        pose = poses[i]
        rays_o, rays_d = sample_rays_np(H, W, focal, pose)

        rays_o_list.append(rays_o.reshape(-1, 3))
        rays_d_list.append(rays_d.reshape(-1, 3))

    rays_o_npy = np.concatenate(rays_o_list, axis=0)
    rays_d_npy = np.concatenate(rays_d_list, axis=0)
    rays = torch.tensor(np.concatenate([rays_o_npy, rays_d_npy], axis=1), dtype=torch.float32,
                        device=device)
    return rays