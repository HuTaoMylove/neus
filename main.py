import os.path
import sys
import torch.nn.functional as F
import imageio
import numpy as np
import setproctitle
import torch
from torch.utils.tensorboard import SummaryWriter
from Network import SDFNetwork, RenderingNetwork, SingleVarianceNetwork
from Render import render_rays
from configs import get_config
from load_data.load_blender import load_blender_data, get_image_res
from ray import get_rays, get_rgb_rays
from utils import to_4x4
from itertools import chain

args = get_config()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
datadir = args.basedir + args.things
name = datadir.split('/')[-1]
N_samples = (args.co_samples, args.re_samples)
H, W = get_image_res(datadir, args.factor)
bound = (2., 6.)
exp_name = f'{name}_({N_samples[0]},{N_samples[1]})_bias_{args.bias}_bs_{args.Batch_size}_pt_{args.perturb}_wbg_{args.white_bkgd}_res_{H}x{W}'
logdir = './log/' + exp_name

if args.only_reconstruct:
    dicts = torch.load(
        logdir + f'/epoch_latest.pth',
        map_location=device)
    net = SDFNetwork(multires=6, bias=args.bias).to(device)
    net.load_state_dict(dicts['sdf'])
    net.save_mesh(
        logdir + '/' + args.things + '.ply',
        resolution=H, device=device)
    sys.exit()

lr = args.lr
images, poses, render_poses, hwf, i_split = load_blender_data(datadir, args.factor, args.test_skip, args.val_skip)
i_train, _, i_test = i_split
masks = images[..., -1]
if args.white_bkgd:
    images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
else:
    images = images[..., :3]
test_img, test_pose = images[i_test], poses[i_test]
test_masks = masks[i_test]
if len(test_img.shape) == 3:
    test_img = np.expand_dims(test_img, 0)
    test_pose = np.expand_dims(test_pose, 0)
render_poses = to_4x4(poses[i_test])
images, poses = images[i_train], poses[i_train]
masks = masks[i_train]
print("Process rays data!")

rays = get_rgb_rays(images, poses, hwf, device)
rays_mask = torch.from_numpy(masks).to(device).reshape(-1, 1).float()
test_rays = get_rgb_rays(test_img, test_pose, hwf, device).reshape(-1, 9).split(args.Batch_size, dim=0)
test_masks = torch.from_numpy(test_masks).to(device).reshape(-1, 1).float().split(args.Batch_size, dim=0)
if render_poses is not None:
    val_rays = get_rays(render_poses, hwf, device).reshape(-1, 6).split(args.Batch_size, dim=0)

#############################
# training parameters
#############################
N = rays.shape[0]
iterations = N // args.Batch_size
print(f"There are {iterations} batches of rays and each batch contains {args.Batch_size} rays")

sdf_network = SDFNetwork(multires=6, bias=args.bias).to(device)
deviation_network = SingleVarianceNetwork(init_val=0.3).to(device)
color_network = RenderingNetwork(multires_view=4).to(device)

if args.render:
    dicts = torch.load(
        logdir + f'/epoch_{args.model_idx}.pth',
        map_location=device)
    sdf_network.load_state_dict(dicts['sdf'])
    deviation_network.load_state_dict(dicts['s'])
    color_network.load_state_dict(dicts['color'])
    rgb_list = list()
    depth_list = list()
    val_frames = list()
    for i in range(len(val_rays)):
        r, m = val_rays[i], test_masks[i]
        rays_o, rays_d = torch.chunk(r, 2, dim=-1)
        rays_od = (rays_o, rays_d)
        rgb, depth, _ = render_rays(sdf_network, color_network, deviation_network, rays_od, bound=bound,
                                    N_samples=N_samples,
                                    device=device,
                                    use_view=args.use_view, perturb=args.perturb,
                                    cos_anneal_ratio=1.0
                                    )
        if args.white_bkgd:
            rgb = rgb * m + (1. - m)
        rgb_list.append(rgb.detach())
        d = (depth.detach() - bound[0]) / (bound[1] - bound[0])
        d = torch.cat([d.unsqueeze(-1),
                       torch.zeros([d.shape[0], 1], device=device),
                       torch.ones([d.shape[0], 1], device=device) - d.unsqueeze(-1)], dim=-1)
        depth_list.append(d)
    rgb = torch.cat(rgb_list, dim=0).reshape(-1, H, W, 3)

    depth = torch.cat(depth_list, dim=0).reshape(-1, H, W, 3)
    result = torch.cat([rgb, depth], dim=2)

    for f in (255 * np.clip(result.cpu().numpy(), 0, 1)).astype(np.uint8):
        val_frames.append(f)
    imageio.mimwrite(logdir + f'/{args.model_idx}_results.mp4', val_frames, fps=30, quality=7)
    sys.exit()

if args.reload:
    assert os.path.exists(logdir + f'/epoch_latest.pth'), 'history files should exist!'
    dicts = torch.load(logdir + f'/epoch_latest.pth')
    last_e = dicts['epoch']
    sdf_network.load_state_dict(dicts['sdf'])
    deviation_network.load_state_dict(dicts['s'])
    color_network.load_state_dict(dicts['color'])
else:
    last_e = 0


optimizer = torch.optim.Adam(
    chain(sdf_network.parameters(), deviation_network.parameters(), color_network.parameters()), lr)
writer = SummaryWriter(logdir)
setproctitle.setproctitle(exp_name)

for e in range(last_e, args.epoch):
    # create iteration for training

    if e < args.warm_up:
        learning_factor = (e + 1) / args.warm_up
    else:
        progress = (e - args.warm_up) / (args.epoch - args.warm_up)
        learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - 0.05) + 0.05

    for g in optimizer.param_groups:
        g['lr'] = lr * learning_factor

    if args.anneal == 0.0:
        cos_anneal_ratio = 1.
    else:
        cos_anneal_ratio = np.min([1.0, (e + 1) / args.anneal])

    batch = torch.randperm(N)
    rays = rays[batch, :]
    mks = rays_mask[batch, :]
    ray_iter = iter(torch.split(rays, args.Batch_size, dim=0))
    mask_iter = iter(torch.split(mks, args.Batch_size, dim=0))
    for i in range(iterations):
        train_rays = next(ray_iter)
        train_masks = next(mask_iter)
        assert train_rays.shape == (args.Batch_size, 9)

        rays_o, rays_d, target_rgb = torch.chunk(train_rays, 3, dim=-1)
        rays_od = (rays_o, rays_d)
        rgb, _, eikonal = render_rays(sdf_network, color_network, deviation_network, rays_od, bound=bound,
                                      N_samples=N_samples,
                                      device=device,
                                      use_view=args.use_view, perturb=args.perturb,
                                      cos_anneal_ratio=cos_anneal_ratio
                                      )

        loss = F.mse_loss(train_masks * rgb, train_masks * target_rgb) + eikonal * 0.1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        psnr = -10. * torch.log(torch.mean(((train_masks * rgb-train_masks * target_rgb)**2)[(train_masks>0.).squeeze(dim=-1)]).detach()).item() / torch.log(
            torch.tensor([10.]))
        writer.add_scalar('train/psnr', psnr, i + iterations * e)
        writer.add_scalar('train/inv_s', 1 / torch.exp(deviation_network.variance * 10.0).clone().detach().cpu().item(),
                          i + iterations * e)

    # sample = torch.randint(0, len(test_rays), [1]).item()
    # r = test_rays[sample]
    # m = test_masks[sample]
    # rays_o, rays_d, rays_rgb = torch.chunk(r, 3, dim=-1)
    # rays_od = (rays_o, rays_d)
    # rgb, _, _ = render_rays(sdf_network, color_network, deviation_network, rays_od, bound=bound,
    #                         N_samples=N_samples,
    #                         device=device,
    #                         use_view=args.use_view, perturb=args.perturb,
    #                         cos_anneal_ratio=cos_anneal_ratio
    #                         )
    # loss = torch.mean(((rgb * m - m * rays_rgb) ** 2)[(m > 0.).squeeze(dim=-1)])
    # loss.backward()
    # optimizer.zero_grad()
    # test_psnr = -10. * torch.log(loss.detach().cpu()).item() / torch.log(torch.tensor([10.]))
    # writer.add_scalar('test/psnr', test_psnr, iterations * e)
    state = {'sdf': sdf_network.state_dict(), 'color': color_network.state_dict(), 's': deviation_network.state_dict(),
             'epoch': e + 1}
    torch.save(state, logdir + f'/epoch_latest.pth')

    if e % args.reconstruct_interval == 0 and e:
        with torch.no_grad():
            sdf_network.save_mesh(logdir + f'/{args.things}_{e}.ply', resolution=H, device=device)

    if e % args.save_interval == 0 and e:
        state = {'sdf': sdf_network.state_dict(), 'color': color_network.state_dict(),
                 's': deviation_network.state_dict(), 'epoch': e + 1}
        torch.save(state, logdir + f'/epoch_{e}.pth')
