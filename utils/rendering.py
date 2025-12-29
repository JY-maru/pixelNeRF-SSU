# utils/rendering.py
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

class VolumeRenderer:
    """Volume rendering utilities for NeRF"""
    
    def __init__(self, white_bkgd: bool = False, noise_std: float = 1.0, config=None):
        """
        Args:
            white_bkgd: Whether to use white background
            noise_std: Standard deviation of noise for density
            config: Config object (kept for compatibility)
        """
        self.white_bkgd = white_bkgd
        self.noise_std = noise_std
        self.config = config
        self._step_counter = 0
    
    def update_step(self, step):
        """Update current step"""
        self._step_counter = step
    
    def sample_along_rays(self, rays_o: torch.Tensor, rays_d: torch.Tensor,
                          z_near: float, z_far: float, n_samples: int,
                          perturb: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample points along rays (Linear in disparity)"""
        N = rays_o.shape[0]
        device = rays_o.device
        
        # Linearly sample in disparity space
        # t_vals = torch.linspace(0., 1., n_samples, device=device)
        # z_vals = 1.0 / (1.0/z_near * (1. - t_vals) + 1.0/z_far * t_vals) 
        
        # [수정 후] Linear in Depth (ShapeNet 같은 객체 중심 데이터셋에 적합)
        # z_near와 z_far 사이를 균등하게 쪼갠다 
        t_vals = torch.linspace(0., 1., n_samples, device=device)
        z_vals = z_near * (1. - t_vals) + z_far * t_vals # (N, n_samples)

        z_vals = z_vals.expand(N, n_samples)
        
        if perturb:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)
            t_rand = torch.rand(z_vals.shape, device=device)
            z_vals = lower + (upper - lower) * t_rand
        
        points = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        
        return points, z_vals
    
    
    def sample_pdf(self, bins: torch.Tensor, weights: torch.Tensor,
                   n_samples: int, perturb: bool = True) -> torch.Tensor:
        """Hierarchical sampling (importance sampling)"""
        weights = weights + 1e-5
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
        
        if perturb:
            u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=cdf.device)
        else:
            u = torch.linspace(0., 1., n_samples, device=cdf.device)
            u = u.expand(list(cdf.shape[:-1]) + [n_samples])
        
        u = u.contiguous()
        indices = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(indices - 1), indices - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(indices), indices)
        indices_g = torch.stack([below, above], dim=-1)
        
        matched_shape = [indices_g.shape[0], indices_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, indices_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, indices_g)
        
        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
        
        return samples
    
    def render_rays(self, raw: torch.Tensor, z_vals: torch.Tensor,
                    rays_d: torch.Tensor, raw_noise_std: float = 0.0) -> Dict[str, torch.Tensor]:
        """
        Volume rendering from raw NeRF output
        
        Args:
            raw: Raw network output (N, n_samples, 4) [rgb, density]
            z_vals: Depth values (N, n_samples)
            rays_d: Ray directions (N, 3)
            raw_noise_std: Noise standard deviation
        """
        
        # 1. Extract RGB and density
        rgb = torch.sigmoid(raw[..., :3])
        raw_density_logit = raw[..., 3]
        
        # 2. Add noise (if regularization enabled)
        if raw_noise_std > 0:
            noise = torch.randn_like(raw_density_logit) * raw_noise_std
            raw_density_logit = raw_density_logit + noise
        
        # 3. Activation (Softplus)
        # Note: Bias (-1.0) is usually handled in the model definition, not here.
        density = F.softplus(raw_density_logit)

        # 4. Compute distances between samples
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        last_delta = z_vals[..., -1:] - z_vals[..., -2:-1] # Use last interval for infinity
        dists = torch.cat([dists, last_delta], dim=-1)
        
        # Scale distances by ray direction to get real world distance
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
        
        # 5. Volume Rendering Equations
        # Alpha: probability of particle existing in this segment
        alpha = 1.0 - torch.exp(-density * dists)
        
        # Transmittance: probability of ray reaching this segment
        # T_i = exp(-sum_{j<i} sigma_j * delta_j)
        T = torch.cumprod(
            torch.cat([
                torch.ones_like(alpha[..., :1]),
                1.0 - alpha + 1e-10
            ], dim=-1),
            dim=-1
        )[..., :-1]
        
        # Weights: contribution of this segment to final color
        weights = alpha * T
        
        # 6. Integrate to get final maps
        rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
        depth_map = torch.sum(weights * z_vals, dim=-1)
        acc_map = torch.sum(weights, dim=-1)
        
        # 7. White background handling
        if self.white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])
        
        return {
            'rgb_map': rgb_map,
            'depth_map': depth_map,
            'acc_map': acc_map,
            'weights': weights,
            'raw_density': density,              # activation 후 (for logging/debug)
            'raw_density_logit': raw_density_logit, # activation 전 (for regularization)
            'raw_rgb': rgb
        }