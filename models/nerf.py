# models/nerf.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Optional

class NeRFNetwork(nn.Module):
    """
    NeRF Network with Feature Injection & Variance (Stereo Matching)
    """
    def __init__(self, 
                 d_in: int = 3, 
                 d_out: int = 4, 
                 d_hidden: int = 512,
                 n_blocks: int = 5, 
                 combine_layer: int = 3, 
                 d_feature: int = 512, 
                 combine_type: str = 'average',
                 use_pe: bool = True, 
                 pe_freq_pos: int = 6, 
                 pe_freq_dir: int = 2):
        super().__init__()
        
        # [Config]
        self.save_features = False # 캐싱용 플래그
        
        # View-Dependent Feature Processing Layer
        self.feature_processing = nn.Linear(d_feature + 3, d_feature)

        # Skip Connection 위치
        self.skip_layer_idx = n_blocks // 2

        # PE Dimension
        if use_pe:
            self.d_pos_encoded = 3 + 3 * 2 * pe_freq_pos
            self.d_dir_encoded = 3 + 3 * 2 * pe_freq_dir
        else:
            self.d_pos_encoded = 3
            self.d_dir_encoded = 3

        self.use_pe = use_pe
        self.pe_freq_pos = pe_freq_pos
        self.pe_freq_dir = pe_freq_dir
            
        # =================================================================
        # [수정 1] Variance 추가에 따른 입력 차원 확장 (d_feature -> d_feature * 2)
        # =================================================================
        # Mean(평균) Feature + Variance(분산) Feature가 합쳐지므로 크기가 2배가 됨
        total_feature_dim = d_feature * 2  
        
        self.input_dim = self.d_pos_encoded + total_feature_dim 
        self.lin_in = nn.Linear(self.input_dim, d_hidden)
        
        self.blocks = nn.ModuleList([
            ResnetBlock(d_hidden) for _ in range(n_blocks)
        ])
        
        # [수정] Reinjection Layer도 2배 커진 Feature를 받아야 함
        self.feature_reinjection = nn.Linear(d_hidden + total_feature_dim, d_hidden)

        self.density_head = nn.Linear(d_hidden, 1)
        
        # View Attention (Score 계산용)
        self.view_attention = nn.Sequential(
            nn.Linear(d_feature, d_feature // 2),
            nn.SiLU(inplace=True), # ReLU -> SiLU 교체
            nn.Linear(d_feature // 2, 1) 
        )
        
        # [수정] RGB Head 입력 차원 수정
        rgb_input_dim = d_hidden + self.d_dir_encoded + total_feature_dim
        self.rgb_fc1 = nn.Linear(rgb_input_dim, d_hidden // 2)
        self.rgb_head = nn.Linear(d_hidden // 2, 3)
        
        # [수정 2] 활성화 함수 교체 (ReLU -> SiLU)
        self.relu = nn.SiLU(inplace=True) 
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        if self.density_head.bias is not None:
            nn.init.constant_(self.density_head.bias, 0.1) 
        nn.init.normal_(self.density_head.weight, mean=0.0, std=0.01)
        
        nn.init.kaiming_normal_(self.rgb_fc1.weight, a=0, mode='fan_in')
        nn.init.kaiming_normal_(self.rgb_head.weight, a=0, mode='fan_in')

        nn.init.normal_(self.feature_processing.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.feature_processing.bias)
        
        nn.init.normal_(self.feature_reinjection.weight, mean=0.0, std=0.05) 
        nn.init.zeros_(self.feature_reinjection.bias)
        
        if self.rgb_head.bias is not None:
            nn.init.zeros_(self.rgb_head.bias)
        if self.rgb_fc1.bias is not None:
            nn.init.zeros_(self.rgb_fc1.bias)
        
        nn.init.zeros_(self.view_attention[-1].weight)
        nn.init.zeros_(self.view_attention[-1].bias)
    
    @staticmethod
    def positional_encoding(x: torch.Tensor, L: int) -> torch.Tensor:
        encoding = [x]
        for l in range(L):
            for fn in [torch.sin, torch.cos]:
                encoding.append(fn(2.0 ** l * np.pi * x))
        return torch.cat(encoding, dim=-1)
    
    def forward(self, points: torch.Tensor, viewdirs: torch.Tensor, 
                 features: torch.Tensor, relative_dirs: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        original_shape = points.shape[:-1]
        
        # Reshape Logic
        if points.dim() == 4: 
            B, N_rays, N_samples, _ = points.shape
            points = points.reshape(B, -1, 3)
            viewdirs = viewdirs.reshape(B, -1, 3)
            if features.dim() == 5:
                features = features.reshape(B, -1, features.shape[-2], features.shape[-1])
            if relative_dirs is not None and relative_dirs.dim() == 5:
                relative_dirs = relative_dirs.reshape(B, -1, relative_dirs.shape[-2], relative_dirs.shape[-1])
            need_reshape = True
        else:
            need_reshape = False

        # =================================================================
        # [수정 3] Variance Calculation Logic (핵심)
        # =================================================================
        if features.dim() == 4 and relative_dirs is not None:
            # 1. Feature와 방향 결합
            combined = torch.cat([features, relative_dirs], dim=-1)
            processed_features = self.feature_processing(combined) # (B, N, V, C)
            
            # 2. Attention Weight 계산
            scores = self.view_attention(processed_features)
            weights = F.softmax(scores, dim=2) 
            
            # 3. [Mean] 가중 평균 계산
            mean_features = torch.sum(weights * processed_features, dim=2) # (B, N, C)
            
            # 4. [Variance] 가중 분산 계산
            # 수식: sum( w * (x - mean)^2 )
            # (각 뷰의 특징이 평균과 얼마나 다른가? -> 입체감의 핵심 힌트)
            var_features = torch.sum(weights * (processed_features - mean_features.unsqueeze(2))**2, dim=2)
            
            # 5. Concatenation: Mean + Variance
            # MLP 입력 크기 = 2 * C
            aggregated_features = torch.cat([mean_features, var_features], dim=-1)
            
        else:
            # Fallback (단일 뷰)
            if features.dim() == 4:
                mean_feat = torch.mean(features, dim=2)
                # 단일 뷰는 분산이 0임
                var_feat = torch.zeros_like(mean_feat)
                aggregated_features = torch.cat([mean_feat, var_feat], dim=-1)
            else:
                # 뷰 차원이 없는 경우 (이미 처리됨) - 단순히 2배로 복사해서 차원 맞춤
                aggregated_features = torch.cat([features, torch.zeros_like(features)], dim=-1)
        
        # --- Positional Encoding ---
        if self.use_pe:
            points_encoded = self.positional_encoding(points, self.pe_freq_pos)
            viewdirs_encoded = self.positional_encoding(viewdirs, self.pe_freq_dir)
        else:
            points_encoded = points
            viewdirs_encoded = viewdirs
            
        # --- Backbone ---
        # 1. First Layer (좌표 + Feature(Mean+Var))
        x_input = torch.cat([points_encoded, aggregated_features], dim=-1)
        x = self.lin_in(x_input)
        x = self.relu(x)
        
        # 2. ResNet Blocks with Injection
        for i, block in enumerate(self.blocks):
            if i == self.skip_layer_idx:
                # 중간에 Feature(Mean+Var) 다시 주입
                x = torch.cat([x, aggregated_features], dim=-1)
                x = self.feature_reinjection(x)
                x = self.relu(x)
            x = block(x)
            
        # --- Heads ---
        density = self.density_head(x)
        
        # RGB Output (Feature(Mean+Var) 또 주입)
        x_rgb = torch.cat([x, viewdirs_encoded, aggregated_features], dim=-1)
        x_rgb = self.rgb_fc1(x_rgb)
        x_rgb = self.relu(x_rgb)
        rgb = self.rgb_head(x_rgb)
        
        # --- Assembly ---
        raw = torch.cat([rgb, density], dim=-1)
        
        if need_reshape:
            raw = raw.reshape(*original_shape, 4)
            
        return raw


class ResnetBlock(nn.Module):
    def __init__(self, d_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d_hidden, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        # [수정] ReLU -> SiLU 교체
        self.relu = nn.SiLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = out + residual
        return out