import torch
import torch.nn.functional as F
from Sample import uniform_sample_point, sample_pdf_point, up_sample, cat_z_vals


def get_rgb_w(sdf_network, color_network, deviation_network, pts, rays_d, z_vals, device, noise_std=.0, use_view=False,
              cos_anneal_ratio=1.0):
    # pts => tensor(Batch_Size, uniform_N, 3)
    # rays_d => tensor(Batch_Size, 3)
    # Run network
    pts_flat = torch.reshape(pts, [-1, 3])
    dir_flat = None
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.tensor([4 / 32], device=device).expand(dists[..., :1].shape)], -1)
    if use_view:
        dir_flat = F.normalize(torch.reshape(rays_d.unsqueeze(-2).expand_as(pts), [-1, 3]), p=2, dim=-1)

    sdf_nn_output = sdf_network(pts_flat)
    sdf = sdf_nn_output[:, :1]
    feature_vector = sdf_nn_output[:, 1:]
    gradients = sdf_network.gradient(pts_flat).squeeze()
    sampled_color = color_network(pts_flat, gradients, dir_flat, feature_vector).reshape(pts.shape)

    inv_s = deviation_network(torch.zeros([1, 3], device=device))[:, :1].clip(1e-6, 1e6)  # Single parameter
    inv_s = inv_s.expand(pts.shape[0] * pts.shape[1], 1)

    true_cos = (dir_flat * gradients).sum(-1, keepdim=True)
    iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                 F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive
    estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
    estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5
    prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
    next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

    p = prev_cdf - next_cdf
    c = prev_cdf
    alpha = ((p + 1e-5) / (c + 1e-5)).reshape(pts.shape[:2]).clip(0.0, 1.0)
    weights = alpha * torch.cumprod(torch.cat([torch.ones([pts.shape[0], 1], device=device), 1. - alpha + 1e-7], -1),
                                    -1)[:, :-1]
    gradient_error = (torch.linalg.norm(gradients.reshape(pts.shape), ord=2,
                                        dim=-1) - 1.0) ** 2

    return sampled_color, weights, gradient_error.mean()


def render_rays(sdf_network, color_network, deviation_network, rays, bound, N_samples, device, noise_std=.0,
                use_view=False, perturb=False,
                cos_anneal_ratio=1.0, ):
    rays_o, rays_d = rays
    near, far = bound
    uniform_N, important_N = N_samples

    z_vals = uniform_sample_point(near, far, uniform_N, device).unsqueeze(0).expand(rays_o.shape[0], uniform_N)

    if perturb:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=device)
        z_vals = lower + (upper - lower) * t_rand
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None]
    # Up sample
    if important_N > 0:
        with torch.no_grad():
            sdf = sdf_network.sdf(pts.reshape(-1, 3)).reshape(-1, uniform_N)
            for i in range(4):
                new_z_vals = up_sample(rays_o, z_vals, sdf, important_N // 4,
                                       32 * 2 ** i, device=device)
                z_vals, sdf = cat_z_vals(sdf_network, rays_o,
                                         rays_d,
                                         z_vals,
                                         new_z_vals,
                                         sdf,
                                         last=(i + 1 == 4))
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None]
    rgb, weights, eikonal = get_rgb_w(sdf_network, color_network, deviation_network, pts, rays_d, z_vals,
                                      device, noise_std=noise_std,
                                      use_view=use_view, cos_anneal_ratio=cos_anneal_ratio)

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = z_vals[torch.arange(z_vals.shape[0], dtype=int, device=device), torch.argmax(weights, dim=-1)]
    acc_map = torch.sum(weights, -1)

    return rgb_map, depth_map, eikonal
