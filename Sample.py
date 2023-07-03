import torch
import torch.nn.functional as F
import numpy as np


def up_sample(rays_o, z_vals, sdf, n_importance, inv_s, device):
    batch_size = rays_o.shape[0]
    prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
    prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
    mid_sdf = (prev_sdf + next_sdf) * 0.5
    cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

    prev_cos_val = torch.cat([torch.zeros([batch_size, 1], device=device), cos_val[:, :-1]], dim=-1)
    cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
    cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
    cos_val = cos_val.clip(-1e3, 0.0)

    dist = (next_z_vals - prev_z_vals)
    prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
    next_esti_sdf = mid_sdf + cos_val * dist * 0.5
    prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
    next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
    alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones([batch_size, 1], device=device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

    z_samples = sample_pdf_point(z_vals, weights, n_importance, device).detach()
    return z_samples


def cat_z_vals(sdf_network, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
    batch_size, n_samples = z_vals.shape
    _, n_importance = new_z_vals.shape
    pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
    z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
    z_vals, index = torch.sort(z_vals, dim=-1)

    if not last:
        new_sdf = sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
        sdf = torch.cat([sdf, new_sdf], dim=-1)
        xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
        index = index.reshape(-1)
        sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

    return z_vals, sdf


def sample_rays_np(H, W, f, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - W * .5) / f, -(j - H * .5) / f, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf_point(bins, weights, N_samples, device):
    pdf = F.normalize(weights + 1e-5, p=1, dim=-1)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # uniform sampling
    u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device).contiguous()

    # invert
    ids = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(ids - 1, device=device), ids - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(ids, device=device), ids)
    ids_g = torch.stack([below, above], -1)
    # ids_g => (batch, N_samples, 2)

    # matched_shape : [batch, N_samples, bins]
    matched_shape = [ids_g.shape[0], ids_g.shape[1], cdf.shape[-1]]
    # gather cdf value
    cdf_val = torch.gather(cdf.unsqueeze(1).expand(matched_shape), -1, ids_g)
    # gather z_val
    bins_val = torch.gather(bins.unsqueeze(1).expand(matched_shape), -1, ids_g)

    # get z_val for the fine sampling
    cdf_d = (cdf_val[..., 1] - cdf_val[..., 0])
    cdf_d = torch.where(cdf_d < 1e-5, torch.ones_like(cdf_d, device=device), cdf_d)
    t = (u - cdf_val[..., 0]) / cdf_d
    samples = bins_val[..., 0] + t * (bins_val[..., 1] - bins_val[..., 0])

    return samples


def uniform_sample_point(tn, tf, N_samples, device):
    k = torch.rand([N_samples], device=device) / float(N_samples)
    pt_value = torch.linspace(0.0, 1.0, N_samples + 1, device=device)[:-1]
    pt_value += k
    return tn + (tf - tn) * pt_value


if __name__ == "__main__":
    a = torch.eye(4)
    b = torch.randn([4, 1])
    c1 = torch.matmul(a, b)
    c2 = b[:, None, :]
    print(c1, c2.shape)
    pass
