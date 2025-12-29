# models/pixelnerf.py
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List

import torch.nn.functional as F
from tqdm import tqdm 
from models.encoder import ImageEncoder
from models.nerf import NeRFNetwork
from utils.geometry import ProjectionUtils
from utils.rendering import VolumeRenderer
from utils.debug import DEBUG_MODE, check_tensor 

class PixelNeRF(nn.Module):
    """
    Pixel-NeRF: Neural Radiance Fields from One or Few Images
    """
    
    def __init__(self, encoder_type: str = 'resnet34', encoder_pretrained: bool = True,
                 feature_dim: int = 512, d_hidden: int = 512, n_blocks: int = 5,
                 combine_type: str = 'average', n_coarse: int = 64, n_fine: int = 128,
                 white_bkgd: bool = True, noise_std: float = 0.0,
                 use_pe: bool = True, pe_freq_pos: int = 10, pe_freq_dir: int = 4, density_lambda=0.01,
                 config=None, step=None):
        super().__init__()
        
        # [수정] Feature Dimension 재계산 (Multi-scale Concatenation 반영)
        # RGB(3) + 4개 FPN Layers * 64 channels = 259
        self.actual_feature_dim = 3 + (256 // 4) * 4 

        # Image encoder
        self.encoder = ImageEncoder(
            encoder_type=encoder_type,
            pretrained=encoder_pretrained,
            feature_dim=feature_dim
        )
        
        # Coarse network
        self.nerf_coarse = NeRFNetwork(
            d_in=3,
            d_out=4,
            d_hidden=d_hidden,
            n_blocks=n_blocks,
            d_feature=self.actual_feature_dim, # 수정된 차원 사용
            combine_type=combine_type,
            use_pe=use_pe,  
            pe_freq_pos=pe_freq_pos,  
            pe_freq_dir=pe_freq_dir   
        )
        
        # Fine network
        self.nerf_fine = NeRFNetwork(
            d_in=3,
            d_out=4,
            d_hidden=d_hidden,
            n_blocks=n_blocks,
            d_feature=self.actual_feature_dim, # 수정된 차원 사용
            combine_type=combine_type,
            use_pe=use_pe,  
            pe_freq_pos=pe_freq_pos,  
            pe_freq_dir=pe_freq_dir
        )
        
        # Renderer
        self.renderer = VolumeRenderer(
            white_bkgd=white_bkgd,
            noise_std=noise_std,
            config=config 
        )

        self.n_coarse = n_coarse
        self.n_fine = n_fine
        self.feature_dim = feature_dim
        self.density_lambda = density_lambda
        self.config = config  
        self._step_counter = 0 
        self.global_step = 0
    
    def transform_dirs_to_all_cameras(self, dirs_world: torch.Tensor, src_poses: torch.Tensor) -> torch.Tensor:
        """Target Ray Direction(World)을 각 Source View의 Camera 좌표계로 변환"""
        R_c2w = src_poses[..., :3, :3]       # (B, V, 3, 3)
        R_w2c = R_c2w.transpose(-1, -2)      # (B, V, 3, 3)
        
        if dirs_world.dim() == 3: # (B, N_rays, 3)
            dirs = dirs_world.unsqueeze(-2).unsqueeze(-1)
            R_mat = R_w2c.unsqueeze(1)
            dirs_cam = torch.matmul(R_mat, dirs).squeeze(-1)
            
        elif dirs_world.dim() == 4: # (B, N_rays, N_samples, 3)
            dirs = dirs_world.unsqueeze(-2).unsqueeze(-1)
            R_mat = R_w2c.unsqueeze(1).unsqueeze(1)
            dirs_cam = torch.matmul(R_mat, dirs).squeeze(-1)
            
        return dirs_cam

    def transform_to_camera_space(self, points_world: torch.Tensor, src_poses: torch.Tensor) -> torch.Tensor:
        """Transform world points to source camera space (Canonical Space)"""
        w2c = torch.inverse(src_poses[:, 0]) 
        ones = torch.ones_like(points_world[..., :1])
        points_h = torch.cat([points_world, ones], dim=-1)
        
        if points_h.dim() == 4: 
            w2c_expanded = w2c.unsqueeze(1).unsqueeze(1) 
            points_cam_h = torch.matmul(points_h.unsqueeze(-2), w2c_expanded.transpose(-1, -2)).squeeze(-2)
        else: 
            w2c_expanded = w2c.unsqueeze(1) 
            points_cam_h = torch.matmul(points_h.unsqueeze(-2), w2c_expanded.transpose(-1, -2)).squeeze(-2)

        return points_cam_h[..., :3]

    def transform_dirs_to_camera_space(self, dirs_world: torch.Tensor, src_poses: torch.Tensor) -> torch.Tensor:
        """Transform world directions to source camera space (Rotation only)"""
        R_c2w = src_poses[:, 0, :3, :3] 
        R_w2c = R_c2w.transpose(1, 2)   
        
        if dirs_world.dim() == 4: 
             R_w2c = R_w2c.unsqueeze(1).unsqueeze(1)
             dirs_cam = torch.matmul(dirs_world.unsqueeze(-2), R_w2c.transpose(-1, -2)).squeeze(-2)
        else:
             R_w2c = R_w2c.unsqueeze(1)
             dirs_cam = torch.matmul(dirs_world.unsqueeze(-2), R_w2c.transpose(-1, -2)).squeeze(-2)
             
        return dirs_cam

    # [핵심] 리스트 입력을 처리하도록 수정된 함수
    def extract_features_at_points(self, points: torch.Tensor,
                             src_features_list: List[torch.Tensor],
                             src_intrinsics: torch.Tensor,
                             src_poses: torch.Tensor,
                             image_size: Tuple[int, int]) -> torch.Tensor:
        
        # 1. 정보 추출 (리스트의 첫 번째 요소 사용)
        B, V, _, _, _ = src_features_list[0].shape
        _, N, _ = points.shape
        
        # 2. Projection (한 번만 계산)
        points_exp = points.unsqueeze(1).expand(B, V, N, 3).contiguous()
        points_flat = points_exp.reshape(B * V, N, 3)
        src_intrinsics_flat = src_intrinsics.reshape(B * V, 3, 3)
        src_poses_flat = src_poses.reshape(B * V, 4, 4)

        all_points_2d = []
        all_valid = []
        for i in range(B * V):
            points_2d, valid = ProjectionUtils.project_points(
                points_flat[i], src_intrinsics_flat[i], src_poses_flat[i], image_size
            )
            all_points_2d.append(points_2d)
            all_valid.append(valid)
            
        points_2d = torch.stack(all_points_2d, dim=0) # (B*V, N, 2)
        valid = torch.stack(all_valid, dim=0)         # (B*V, N)
        valid_f = valid.unsqueeze(-1).float()         # (B*V, N, 1)

        # 3. Multi-scale Sampling Loop
        sampled_features_list = []
        
        for feat_map in src_features_list:
            # feat_map: (B, V, C_s, H_s, W_s)
            _, _, C_s, H_s, W_s = feat_map.shape
            
            # (B*V, C_s, H_s, W_s)로 변환
            feat_flat = feat_map.contiguous().reshape(B * V, C_s, H_s, W_s)
            
            # Geometry Utils 사용하여 샘플링
            sampled = ProjectionUtils.sample_features(
                feat_flat, points_2d, image_size
            ) # (B*V, C_s, N)
            
            sampled = sampled.permute(0, 2, 1) # (B*V, N, C_s)
            
            # 마스킹
            sampled = sampled * valid_f
            sampled_features_list.append(sampled)
            
        # 4. Concatenation (채널 방향 결합)
        combined_features = torch.cat(sampled_features_list, dim=-1) # (B*V, N, Total_C)
        
        # 5. Reshape Back
        total_c = combined_features.shape[-1]
        features = combined_features.reshape(B, V, N, total_c)
        
        return features.contiguous()

    
    def forward(self, src_images: torch.Tensor, src_intrinsics: torch.Tensor,
            src_poses: torch.Tensor, tgt_rays_o: torch.Tensor,
            tgt_rays_d: torch.Tensor, z_near: float, z_far: float,
            step=None) -> Dict[str, torch.Tensor]:
        
        if step is not None:
            self.global_step = step
            self.renderer.update_step(step)

        B, V, _, H, W = src_images.shape
        B_rays, N_rays, _ = tgt_rays_o.shape
        
        # Normalize ray directions
        tgt_rays_d = F.normalize(tgt_rays_d, p=2, dim=-1)
        
        # [수정] Encode: 리스트(List[Tensor]) 반환
        src_features_list = self.encoder(src_images)
        
        # Note: 이전의 reshape 로직 삭제 (Encoder가 이미 처리함)
        
        # ==================== Coarse Pass ====================
        rays_o_flat = tgt_rays_o.reshape(B * N_rays, 3)
        rays_d_flat = tgt_rays_d.reshape(B * N_rays, 3)
        
        points_coarse, z_vals_coarse = self.renderer.sample_along_rays(
            rays_o_flat, rays_d_flat, z_near, z_far, self.n_coarse, perturb=self.training
        )
        
        points_coarse = points_coarse.reshape(B, N_rays, self.n_coarse, 3).contiguous() 
        z_vals_coarse = z_vals_coarse.reshape(B, N_rays, self.n_coarse).contiguous() 
        
        # Feature Extraction
        points_coarse_flat = points_coarse.reshape(B, N_rays * self.n_coarse, 3)
        
        # [수정] 리스트를 넘겨줌
        features_coarse = self.extract_features_at_points(
            points_coarse_flat, src_features_list, src_intrinsics, src_poses, (H, W)
        )
        
        # [수정] 차원 처리 (C 대신 계산된 마지막 차원 사용)
        total_C = features_coarse.shape[-1]
        features_coarse = features_coarse.reshape(B, V, N_rays, self.n_coarse, total_C)
        features_coarse = features_coarse.permute(0, 2, 3, 1, 4).contiguous() 

        # 좌표 변환 (World -> Camera)
        points_coarse_cam = self.transform_to_camera_space(points_coarse, src_poses)
        
        # 방향 변환
        viewdirs_coarse = tgt_rays_d.unsqueeze(2).expand(-1, -1, self.n_coarse, -1)
        viewdirs_coarse_cam = self.transform_dirs_to_camera_space(viewdirs_coarse, src_poses)
        relative_dirs_coarse = self.transform_dirs_to_all_cameras(viewdirs_coarse, src_poses)

        # MLP Inference
        raw_coarse = self.nerf_coarse(
            points_coarse_cam, 
            viewdirs_coarse_cam, 
            features_coarse, 
            relative_dirs=relative_dirs_coarse 
        )

        # Rendering
        results_coarse = {}
        for i in range(B):
            result = self.renderer.render_rays(
                raw_coarse[i], z_vals_coarse[i], tgt_rays_d[i],
                raw_noise_std=self.renderer.noise_std if self.training else 0.0
            )
            for key, value in result.items():
                if key not in results_coarse: results_coarse[key] = []
                results_coarse[key].append(value)
        
        for key in results_coarse:
            results_coarse[key] = torch.stack(results_coarse[key], dim=0)

        # ==================== Fine Pass ====================
        z_vals_mid = 0.5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])
        weights_for_sampling = results_coarse['weights'][..., 1:-1]
        
        bins = z_vals_mid.reshape(-1, self.n_coarse - 1)
        weights = weights_for_sampling.reshape(-1, self.n_coarse - 2)

        z_samples_fine = self.renderer.sample_pdf(
            bins, weights, self.n_fine, perturb=self.training
        ).reshape(B, N_rays, self.n_fine).contiguous() 

        z_vals_fine, _ = torch.sort(torch.cat([z_vals_coarse, z_samples_fine], dim=-1), dim=-1)
        z_vals_fine = z_vals_fine.contiguous()
        
        points_fine = tgt_rays_o[..., None, :] + tgt_rays_d[..., None, :] * z_vals_fine[..., :, None]
        points_fine = points_fine.contiguous() 
        
        n_total = self.n_coarse + self.n_fine

        points_fine_flat = points_fine.reshape(B, N_rays * n_total, 3)
        
        # [수정] 리스트 사용
        features_fine = self.extract_features_at_points(
            points_fine_flat, src_features_list, src_intrinsics, src_poses, (H, W)
        )
        features_fine = features_fine.reshape(B, V, N_rays, n_total, total_C)
        features_fine = features_fine.permute(0, 2, 3, 1, 4).contiguous() 

        # 좌표 및 방향 변환
        points_fine_cam = self.transform_to_camera_space(points_fine, src_poses)
        viewdirs_fine = tgt_rays_d.unsqueeze(2).expand(-1, -1, n_total, -1)
        viewdirs_fine_cam = self.transform_dirs_to_camera_space(viewdirs_fine, src_poses)
        relative_dirs_fine = self.transform_dirs_to_all_cameras(viewdirs_fine, src_poses)

        # MLP Inference
        raw_fine = self.nerf_fine(
            points_fine_cam, 
            viewdirs_fine_cam, 
            features_fine,
            relative_dirs=relative_dirs_fine 
        )

        # Rendering
        results_fine = {}
        for i in range(B):
            result = self.renderer.render_rays(
                raw_fine[i], z_vals_fine[i], tgt_rays_d[i],
                raw_noise_std=self.renderer.noise_std if self.training else 0.0,
            )
            for key, value in result.items():
                if key not in results_fine: results_fine[key] = []
                results_fine[key].append(value)

        for key in results_fine:
            results_fine[key] = torch.stack(results_fine[key], dim=0)
                
        # [수정] 캐싱 로직: 리스트를 저장하도록 변경
        if hasattr(self, 'save_features') and self.save_features:
            self.cached_features = src_features_list

        return {
            'coarse': results_coarse,
            'fine': results_fine,
            'density_reg_coarse': self._compute_density_reg(results_coarse),
            'density_reg_fine': self._compute_density_reg(results_fine)
        }

    def _compute_density_reg(self, results):
        density = results.get('raw_density', None)
        if density is None:
            raw_density_logit = results.get('raw_density_logit', None)
            if raw_density_logit is None:
                return torch.tensor(0.0, device=results['rgb_map'].device)
            return torch.nn.functional.softplus(raw_density_logit).mean()
        return density.mean()