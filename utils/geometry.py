# utils/geometry.py 
import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from typing import Tuple, Optional
from utils.debug import DEBUG_MODE, check_tensor


class CameraUtils:
    """Camera-related utility functions"""
    
    @staticmethod
    def parse_intrinsics(intrinsics_path: str) -> torch.Tensor:
        """
        Parse intrinsics file
        
        Args:
            intrinsics_path: Path to intrinsics file
            
        Returns:
            Intrinsics matrix (3x3)
        """
        with open(intrinsics_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) == 4:  # intrinsics.txt format
            fx, cx, cy, _ = map(float, lines[0].split())
            fy = fx  # Assume square pixels
            intrinsics = torch.tensor([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=torch.float32)
        else:  # intrinsics/000000.txt format
            values = list(map(float, lines[0].split()))
            intrinsics = torch.tensor(values, dtype=torch.float32).reshape(3, 3)
        
        return intrinsics
    
    @staticmethod
    def parse_pose(pose_path: str) -> torch.Tensor:
        """
        Parse camera pose file (world-to-camera transformation)
        
        Args:
            pose_path: Path to pose file
            
        Returns:
            Pose matrix (4x4)
        """
        with open(pose_path, 'r') as f:
            values = list(map(float, f.read().split()))
        
        pose = torch.tensor(values, dtype=torch.float32).reshape(4, 4)
        return pose
    
    @staticmethod
    def get_ray_directions(H: int, W: int, intrinsics: torch.Tensor) -> torch.Tensor:
        """
        Get ray directions for all pixels in camera coordinate system
        
        OpenCV convention: +Z forward, Y down (image coordinates)
        FIXED: Now properly normalized!
        
        Args:
            H: Image height
            W: Image width
            intrinsics: Camera intrinsics (3x3)
            
        Returns:
            Ray directions (H, W, 3) - NORMALIZED to unit length
        """
        device = intrinsics.device

        i, j = torch.meshgrid(
            torch.arange(W, dtype=torch.float32, device=device),
            torch.arange(H, dtype=torch.float32, device=device),
            indexing='xy'
        )
        
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        # OpenCV convention: +Z forward, no Y flip needed
        directions = torch.stack([
            (i - cx) / fx,          # X: right
            (j - cy) / fy,          # Y: down (image coordinates)
            torch.ones_like(i)      # Z: forward (+Z)
        ], dim=-1)
        
        # CRITICAL FIX: Normalize ray directions
        # Without this, volume rendering delta calculation is WRONG!
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)
        
        return directions
    
    @staticmethod
    def get_rays(H: int, W: int, intrinsics: torch.Tensor, 
                 c2w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get ray origins and directions in world coordinate system
        
        Args:
            H: Image height
            W: Image width
            intrinsics: Camera intrinsics (3x3)
            c2w: Camera-to-world transformation (4x4)
            
        Returns:
            rays_o: Ray origins (H, W, 3)
            rays_d: Ray directions (H, W, 3) - NORMALIZED
        """
        # Get ray directions in camera coordinate (already normalized)
        directions = CameraUtils.get_ray_directions(H, W, intrinsics)
        
        # Transform to world coordinate
        rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
        rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)
        
        #print(f"Ray origin mean: {rays_o.mean(dim=(0,1))}")
        #print(f"Ray direction mean: {rays_d.mean(dim=(0,1))}")
        #print(f"Projected back to image space? ...")
        # Rays are already normalized from get_ray_directions()
        # No need to normalize again since rotation preserves length
        
        return rays_o, rays_d


class ProjectionUtils:
    @staticmethod
    def project_points(points: torch.Tensor, intrinsics: torch.Tensor,
                c2w: torch.Tensor, image_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project 3D points to 2D image coordinates
        
        OpenCV convention: +Z forward
        
        Args:
            points: (N, 3) world space points
            intrinsics: (3, 3) camera intrinsics
            c2w: (4, 4) camera-to-world transform (OpenCV convention)
            image_size: (H, W)
            
        Returns:
            points_2d: (N, 2) image coordinates in [0, W] x [0, H]
            valid: (N,) boolean mask for valid projections
        """
        
        if DEBUG_MODE:
            tqdm.write(f"\n=== ProjectionUtils.project_points ===")
            check_tensor("Input points", points)
            check_tensor("Input intrinsics", intrinsics)
            check_tensor("Input c2w", c2w)
        
        N = points.shape[0]
        H, W = image_size
        
        # c2w → w2c transformation
        R_c2w = c2w[:3, :3]  # (3, 3)
        t_c2w = c2w[:3, 3]   # (3,)
        
        # Inverse: R_w2c = R_c2w^T, t_w2c = -R_w2c @ t_c2w
        R_w2c = R_c2w.T
        t_w2c = -torch.matmul(R_w2c, t_c2w)
        
        # Transform to camera space: points_cam = R_w2c @ points^T + t_w2c
        points_cam = torch.matmul(R_w2c, points.T).T + t_w2c  # (N, 3)
        
        if DEBUG_MODE:
            check_tensor("points_cam", points_cam)
        
        # OpenCV: +Z is front
        z = points_cam[:, 2]
        valid_depth = z > 0.1  # Positive Z for points in front
        
        if DEBUG_MODE:
            tqdm.write(f"  Valid depth ratio: {valid_depth.float().mean():.3f}")
            tqdm.write(f"  Z range: [{z.min():.3f}, {z.max():.3f}]")
        
        # Safe division
        z_safe = torch.where(
            valid_depth,
            z,
            torch.ones_like(z) * 1.0  # Changed to positive
        )
        z_safe = torch.clamp(z_safe, min=0.1)  # Changed to positive
        
        # Project to image plane
        x_norm = points_cam[:, 0] / z_safe  # No negation
        y_norm = points_cam[:, 1] / z_safe  # No negation
        
        # Apply intrinsics
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        
        u = fx * x_norm + cx
        v = fy * y_norm + cy
        
        # Check bounds
        margin = 0
        valid_x = (u >= -margin) & (u < W + margin)
        valid_y = (v >= -margin) & (v < H + margin)
        valid = valid_depth & valid_x & valid_y
        
        if DEBUG_MODE:
            tqdm.write(f"  Valid projection ratio: {valid.float().mean():.3f}")
        
        points_2d = torch.stack([u, v], dim=-1)  # (N, 2)
        
        if DEBUG_MODE:
            check_tensor("Output points_2d", points_2d)
            tqdm.write(f"=== End project_points ===\n")
        
        return points_2d, valid
    
    @staticmethod
    def sample_features(features: torch.Tensor, points_2d: torch.Tensor,
                       image_size: Tuple[int, int]) -> torch.Tensor:
        """
        Sample features at 2D points using bilinear interpolation
        
        Args:
            features: (B, C, H_feat, W_feat) feature maps
            points_2d: (B, N, 2) image coordinates in [0, W_img] x [0, H_img]
            image_size: (H_img, W_img) original image size
            
        Returns:
            sampled: (B, C, N) sampled features
        """
        
        if DEBUG_MODE:
            tqdm.write(f"\n=== ProjectionUtils.sample_features ===")
            check_tensor("Input features", features)
            check_tensor("Input points_2d", points_2d)
        
        B, C, H_feat, W_feat = features.shape
        _, N, _ = points_2d.shape
        H_img, W_img = image_size
        
        u = points_2d[..., 0]  # (B, N)
        v = points_2d[..., 1]  # (B, N)
        
        # [수정] align_corners=False 논리에 맞춰 '-1' 제거
        # 이유: False 옵션은 픽셀의 중심이 아니라 가장자리(면적)를 기준으로 하므로
        # 좌표 변환 시 너비/높이 그 자체(W, H)로 비율을 계산해야 정확함
        
        # 1. Feature Map 좌표계로 스케일링 (비율 유지)
        # 기존: u * (W_feat - 1) / (W_img - 1)  <-- (X) True용 수식
        # 변경: u * (W_feat / W_img)            <-- (O) False용 수식
        u_scaled = u * (W_feat / W_img)
        v_scaled = v * (H_feat / H_img)

        # 2. [-1, 1]로 정규화
        # 기존: (u_scaled / (W_feat - 1)) * 2 - 1 <-- (X) True용 수식
        # 변경: (u_scaled / W_feat) * 2 - 1      <-- (O) False용 수식
        u_norm = (u_scaled / W_feat) * 2 - 1 
        v_norm = (v_scaled / H_feat) * 2 - 1
        
        # Clamp to valid range
        # u_norm = torch.clamp(u_norm, -1, 1)
        # v_norm = torch.clamp(v_norm, -1, 1)
        
        if DEBUG_MODE:
            tqdm.write(f"  Image size: {H_img} x {W_img}")
            tqdm.write(f"  Feature size: {H_feat} x {W_feat}")
            tqdm.write(f"  u range: [{u.min():.3f}, {u.max():.3f}]")
            tqdm.write(f"  v range: [{v.min():.3f}, {v.max():.3f}]")
            tqdm.write(f"  u_norm range: [{u_norm.min():.3f}, {u_norm.max():.3f}]")
            tqdm.write(f"  v_norm range: [{v_norm.min():.3f}, {v_norm.max():.3f}]")
        
        grid = torch.stack([u_norm, v_norm], dim=-1)  # (B, N, 2)
        grid = grid.unsqueeze(1)  # (B, 1, N, 2)
        
        # [CRITICAL FIX] Force contiguous memory layout
        features = features.contiguous()
        grid = grid.contiguous()

        # Sample features
        sampled = F.grid_sample(
            features,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )  # (B, C, 1, N)
        
        sampled = sampled.squeeze(2)  # (B, C, N)
        
        if DEBUG_MODE:
            check_tensor("Sampled features", sampled)
            tqdm.write(f"=== End sample_features ===\n")
        
        return sampled