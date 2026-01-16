# models/encoder.py
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from tqdm import tqdm  
from typing import List

from utils.debug import DEBUG_MODE, check_tensor


class ResNetEncoder(nn.Module):
    """
    ResNet-based image encoder for Pixel-NeRF
    Extracts multi-scale features from input images AND concatenates raw RGB.
    """
    
    def __init__(self, backbone: str = 'resnet34', pretrained: bool = True,
                 feature_dim: int = 512):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Load backbone
        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            self.latent_size = [64, 64, 128, 256, 512]
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            self.latent_size = [64, 64, 128, 256, 512]
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.latent_size = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Extract layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # [추가] FPN Lateral Layers (채널 수 맞추기 1x1 Conv)
        # 상위 Feature의 채널을 256(또는 적절한 값)으로 통일하여 합치기 쉽게 만듦
        fpn_dim = 256
        self.lat_layer4 = nn.Conv2d(self.latent_size[4], fpn_dim, 1)
        self.lat_layer3 = nn.Conv2d(self.latent_size[3], fpn_dim, 1)
        self.lat_layer2 = nn.Conv2d(self.latent_size[2], fpn_dim, 1)
        self.lat_layer1 = nn.Conv2d(self.latent_size[1], fpn_dim, 1) # layer1 출력용
        
        # [추가] Smoothing Layers (합친 후 3x3 Conv로 앨리어싱 제거)
        self.smooth4 = nn.Conv2d(fpn_dim, fpn_dim // 4, 3, padding=1)
        self.smooth3 = nn.Conv2d(fpn_dim, fpn_dim // 4, 3, padding=1)
        self.smooth2 = nn.Conv2d(fpn_dim, fpn_dim // 4, 3, padding=1)
        self.smooth1 = nn.Conv2d(fpn_dim, fpn_dim // 4, 3, padding=1)
        
        # Final Reduction
        # 입력: (fpn_dim//4 * 4개 레이어) + 3(RGB)
        total_channels = (fpn_dim // 4) * 4 + 3
        
        # Feature dimension reduction
        # 입력 채널에 +3 (원본 RGB 이미지 채널)을 추가
        self.reduce = nn.Sequential(
            nn.Conv2d(total_channels, feature_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 1, bias=False) # Final projection
        )
    
    def _upsample_add(self, x, y):
        """Upsample x and add to y"""
        _, _, H, W = y.shape
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input image (B, 3, H, W) - Normalized
            
        Returns:
            features: Multi-scale features (B, feature_dim, H//4, W//4)
        """
        if DEBUG_MODE:
            tqdm.write(f"\n=== ResNetEncoder in encoder.py forward() ===")
            check_tensor("input_shape", x)
        
        # Save original input for concatenation (clone to be safe)
        # x는 ImageNet Normalized 된 상태 (B, 3, H, W)
        
        # [수정] Denormalization 수행 (approximate)
        # ImageNet Mean/Std를 역산하여 [0, 1] 범위로 근사 복원
        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        
        # 복원된 이미지 (Feature 결합용)
        raw_rgb = (x * std + mean).clamp(0.0, 1.0)

        if DEBUG_MODE: check_tensor("cloned input_img (RGB source)", raw_rgb)
        
        # 2. Bottom-Up Pathway (ResNet Backbone)
        c1 = self.relu(self.bn1(self.conv1(x))) # 1/2 scale (사실 layer1 전이라 64채널)
        c1 = self.maxpool(c1)                   # 1/4 scale
        
        c1 = self.layer1(c1) # 1/4 scale
        c2 = self.layer2(c1) # 1/8 scale
        c3 = self.layer3(c2) # 1/16 scale
        c4 = self.layer4(c3) # 1/32 scale
        
        # 3. Top-Down Pathway (FPN)
        # Semantic 정보(c4)를 아래로 흘려보냄
        p4 = self.lat_layer4(c4)
        p3 = self._upsample_add(p4, self.lat_layer3(c3))
        p2 = self._upsample_add(p3, self.lat_layer2(c2))
        p1 = self._upsample_add(p2, self.lat_layer1(c1))
        
        # 4. Multi-scale Feature Aggregation [수정]
        # 강제로 target_size로 맞추는 로직 삭제
        # target_size = p1.shape[2:] (삭제)

        # 각 해상도 그대로 유지 (Smoothing만 적용)
        f4 = self.smooth4(p4) # 1/32
        f3 = self.smooth3(p3) # 1/16
        f2 = self.smooth2(p2) # 1/8
        f1 = self.smooth1(p1) # 1/4
        
        # 원본 RGB (해상도 1/1 유지 혹은 필요시 1/2)
        # 메모리 절약을 위해 RGB도 1/2 정도로 줄여도 되지만, 디테일을 위해 원본 유지 권장
        # raw_rgb는 이미 위에서 계산됨
        
        # [핵심 변경] 리스트로 반환 (Pyramid)
        # 순서: [Highest Res -> Lowest Res]
        return [raw_rgb, f1, f2, f3, f4]


class ImageEncoder(nn.Module):
    """
    Main image encoder class
    """
    
    def __init__(self, encoder_type: str = 'resnet34', pretrained: bool = True,
                 feature_dim: int = 512):
        super().__init__()
        
        if 'resnet' in encoder_type:
            self.encoder = ResNetEncoder(
                backbone=encoder_type,
                pretrained=pretrained,
                feature_dim=feature_dim
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        self.feature_dim = feature_dim
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to feature maps
        Handles both 4D [B, C, H, W] and 5D [B, V, C, H, W] inputs
        """
        if DEBUG_MODE:
            tqdm.write(f"\n=== ImageEncoder in encoder.py forward() ===")
            check_tensor("input_shape", images)

        # 1. 입력 차원 확인 및 전처리 (5D -> 4D 병합)
        if images.dim() == 5:
            B, V, C, H, W = images.shape
            # Encoder에 넣기 위해 Batch와 View를 합침 (Flatten)
            images_reshaped = images.view(-1, C, H, W) 
            is_5d_input = True
            if DEBUG_MODE: check_tensor("Reshaped for encoder", images_reshaped)
        else:
            # 이미 4D라면 그대로 사용
            images_reshaped = images
            is_5d_input = False

        # 2. Encoder 통과 (리스트 반환됨)
        features_list = self.encoder(images_reshaped)
        
        # 3. 차원 복구 (리스트 내의 모든 텐서에 대해 수행)
        final_features = []
        if is_5d_input:
            for feat in features_list:
                # feat: [B*V, C_f, H_f, W_f]
                _, C_f, H_f, W_f = feat.shape
                # -> [B, V, C_f, H_f, W_f]
                feat_5d = feat.view(B, V, C_f, H_f, W_f)
                final_features.append(feat_5d)
        else:
            final_features = features_list
            
        return final_features