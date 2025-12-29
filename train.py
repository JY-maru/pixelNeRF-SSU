# train.py
"""
┌─────────────────────┐
│          Data Preprocessing              │
├─────────────────────┤
│ Source Images:                           │
│   [0,1] → ImageNet Normalize → [-2,2]  │
│   (For pretrained ResNet encoder)        │
│                                          │
│ Target Images:                           │
│   [0,1] (NO normalization)               │
│   (For supervision/comparison)           │
└─────────────────────┘
                    ↓
┌─────────────────────┐
│              Model Forward               │
├─────────────────────┤
│ Encoder(source_normalized) → features   │
│ NeRF(features) → output [0,1]           │
└─────────────────────┘
                    ↓
┌─────────────────────┐
│         Loss / Metrics / Save            │
├─────────────────────┤
│ Loss: MSE(output[0,1], target[0,1])      │
│ PSNR: -10*log10(MSE)                     │
│ Save: output[0,1] → PNG                 │
└─────────────────────┘
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  


from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse
import time
from datetime import timedelta

from config.config import Config
from models.pixelnerf import PixelNeRF
from data.shapenet import ShapeNetDataset, collate_fn
from utils.geometry import CameraUtils
from utils.debug import DEBUG_MODE, check_tensor, set_debug_mode
from models.encoder import ImageEncoder
from data.shapenet import FilteredShapeNetDataset


def mse2psnr(mse):
    """
    Convert MSE to PSNR
    
    Args:
        mse: Mean squared error
        
    Returns:
        PSNR in dB
    """
    return -10.0 * torch.log10(mse)

def compute_ssim(img1, img2, window_size=11, size_average=True):
        """
        Compute SSIM between two images
        Args:
            img1, img2: (B, 3, H, W) tensors in [0, 1]
        """
        from math import exp

        def create_window(window_size, channel):
            _1D_window = torch.tensor(
                [exp(-(x - window_size // 2)**2 / float(2 * 1.5**2)) for x in range(window_size)]
            )
            _1D_window = _1D_window.unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
            return window

        channel = img1.size(1)
        window = create_window(window_size, channel).to(img1.device)
        
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


class Trainer:
    """
    Trainer class for Pixel-NeRF
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        
        set_debug_mode(config.debug.enabled)
        self.step_debug = config.debug.log_every_n_steps

        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
        
        tqdm.write(f"\n{'='*70}")
        tqdm.write(f"{'Pixel-NeRF Training':^70}")
        tqdm.write(f"{'='*70}")
        tqdm.write(f"Using device: {self.device}")
        
        if config.debug.enabled:
            tqdm.write(f"\n🐛 Debug Configuration:")
            tqdm.write(f"   Mode: ENABLED")
            tqdm.write(f"   Log first batch: {config.debug.log_first_batch}")
            tqdm.write(f"   Log every N steps: {config.debug.log_every_n_steps}")
            tqdm.write(f"   Check NaN/Inf: {config.debug.check_nan_inf}")
            tqdm.write(f"   Save debug images: {config.debug.save_debug_images}")
        else:
            tqdm.write(f"Debug mode: Disabled")

        # Build model
        self.model = PixelNeRF(
            encoder_type=config.model.encoder_type,
            encoder_pretrained=config.model.encoder_pretrained,
            feature_dim=config.model.feature_dim,
            d_hidden=config.model.d_hidden,
            n_blocks=config.model.n_blocks,
            combine_type=config.model.combine_type,
            n_coarse=config.model.n_coarse,
            n_fine=config.model.n_fine,   
            white_bkgd=config.model.white_bkgd,
            noise_std=config.model.noise_std, 
            use_pe=config.model.use_pe,  
            pe_freq_pos=config.model.pe_freq_pos,  
            pe_freq_dir=config.model.pe_freq_dir,
            density_lambda=config.model.density_lambda,
            config=config
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        tqdm.write(f"Total parameters: {total_params:,}")
        tqdm.write(f"Trainable parameters: {trainable_params:,}")
        
        # Build datasets
        tqdm.write(f"\n{'='*70}")
        tqdm.write("Loading datasets...")
        
        self.train_dataset = FilteredShapeNetDataset(
            data_root=self.config.data.data_root,
            split='train',
            num_source_views=self.config.data.num_source_views,
            image_size=self.config.data.image_size,
            z_near=self.config.data.z_near,
            z_far=self.config.data.z_far,
            use_imagenet_normalize=self.config.data.use_imagenet_normalize,
            # Filtering parameters 
            enable_filtering=getattr(self.config.data, 'enable_filtering', True),
            min_view_angle=getattr(self.config.data, 'min_view_angle', 10),           
            max_view_angle=getattr(self.config.data, 'max_view_angle', 60),
            min_camera_distance=getattr(self.config.data, 'min_camera_distance', 0.1), 
            max_camera_distance=getattr(self.config.data, 'max_camera_distance', 3.0),
	          min_elevation=getattr(self.config.data, 'min_elevation', 5.0), 
            max_elevation=getattr(self.config.data, 'max_elevation', 85.0),
            num_instances=getattr(self.config.data, 'num_train_instances', None),
            repeat_factor=getattr(self.config.data, 'repeat_factor', 1)
        )

        self.val_dataset = FilteredShapeNetDataset(
            data_root=self.config.data.data_root,
            split='val',  
            num_source_views=self.config.data.num_source_views,
            image_size=self.config.data.image_size,
            z_near=self.config.data.z_near,
            z_far=self.config.data.z_far,
            use_imagenet_normalize=self.config.data.use_imagenet_normalize,
            # Filtering parameters 
            enable_filtering=getattr(self.config.data, 'enable_filtering', True),
            min_view_angle=getattr(self.config.data, 'min_view_angle', 10),           
            max_view_angle=getattr(self.config.data, 'max_view_angle', 120),
            min_camera_distance=getattr(self.config.data, 'min_camera_distance', 0.1), 
            max_camera_distance=getattr(self.config.data, 'max_camera_distance', 3.0),
            num_instances=getattr(self.config.data, 'num_val_instances', None),
            repeat_factor=1 # Validation은 반복 안 함
        )

        tqdm.write(f"\n{'='*70}")
        tqdm.write("Dataset Loading Complete")
        tqdm.write(f"{'='*70}")
        tqdm.write(f"Training samples:   {len(self.train_dataset):>6}")
        tqdm.write(f"Validation samples: {len(self.val_dataset):>6}")
        tqdm.write(f"{'='*70}\n")
        
        # Build dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        tqdm.write(f"Batches per epoch: {len(self.train_loader)}")
        
        # Logging strategy
        tqdm.write(f"\n{'='*70}")
        tqdm.write("Logging Strategy:")
        if config.training.log_every_n_steps > 0:
            tqdm.write(f"  - Training metrics logged every {config.training.log_every_n_steps} steps")
        if config.training.log_every_n_epochs > 0:
            tqdm.write(f"  - Training metrics logged every {config.training.log_every_n_epochs} epochs")
        if config.training.val_every_n_steps > 0:
            tqdm.write(f"  - Validation every {config.training.val_every_n_steps} steps")
        if config.training.val_every_n_epochs > 0:
            tqdm.write(f"  - Validation every {config.training.val_every_n_epochs} epochs")
        if config.training.save_every_n_steps > 0:
            tqdm.write(f"  - Checkpoints saved every {config.training.save_every_n_steps} steps")
        if config.training.save_every_n_epochs > 0:
            tqdm.write(f"  - Checkpoints saved every {config.training.save_every_n_epochs} epochs")
        
        # Build optimizer
        self.optimizer = optim.Adam([
            {
                'params': self.model.nerf_coarse.parameters(), 
                'lr': config.training.learning_rate * 1.0, 
                'weight_decay': config.training.weight_decay
            },
            {
                'params': self.model.nerf_fine.parameters(), 
                'lr': config.training.learning_rate,  # Fine 기본
                'weight_decay': config.training.weight_decay
            },
            {
                'params': self.model.encoder.parameters(), 
                'lr': config.training.learning_rate * 0.1,  # Encoder 천천히
                'weight_decay': 0.0  # Encoder는 weight decay 안 함 (pretrained)
            }
        ])
        
        # Learning rate scheduler
        scheduler_type = getattr(config.scheduler, 'type', 'cosine')
        
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.scheduler.T_max,
                eta_min=config.scheduler.eta_min
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=getattr(config.scheduler, 'step_size', 50),
                gamma=getattr(config.scheduler, 'gamma', 0.5)
            )
        elif scheduler_type == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=getattr(config.scheduler, 'gamma', 0.98)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        # Loss function (MSE as per NeRF paper - Equation 2)
        self.criterion = nn.MSELoss()
        
        # Checkpoint directory
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(
            log_dir=os.path.join(config.training.checkpoint_dir, 'logs')
        )
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_psnr = 0.0
        self.best_val_ssim = 0.0
        self.start_time = None
        
        # Resume from checkpoint if specified
        if config.training.resume_from:
            self.load_checkpoint(
                config.training.resume_from, 
                load_optimizer=True
            )
        tqdm.write(f"{'='*70}\n")
    
    def should_log_step(self):
        """Check if should log at current step"""
        if self.config.training.log_every_n_steps <= 0:
            return False
        return self.global_step % self.config.training.log_every_n_steps == 0
    
    def should_validate_step(self):
        """Check if should validate at current step"""
        if self.config.training.val_every_n_steps <= 0:
            return False
        return self.global_step % self.config.training.val_every_n_steps == 0 and self.global_step > 0
    
    def should_save_step(self):
        """Check if should save checkpoint at current step"""
        if self.config.training.save_every_n_steps <= 0:
            return False
        return self.global_step % self.config.training.save_every_n_steps == 0 and self.global_step > 0
    
    def should_validate_epoch(self):
        """Check if should validate at current epoch"""
        if self.config.training.val_every_n_epochs <= 0:
            return False
        return (self.epoch + 1) % self.config.training.val_every_n_epochs == 0
    
    def should_save_epoch(self):
        """Check if should save checkpoint at current epoch"""
        if self.config.training.save_every_n_epochs <= 0:
            return False
        return (self.epoch + 1) % self.config.training.save_every_n_epochs == 0
    
    def _print_dataset_statistics(self):
        """Print detailed dataset statistics at the start of training"""
        print(f"\n{'='*70}")
        print(f"📊 DATASET STATISTICS")
        print(f"{'='*70}")
        
        # Helper function
        def print_view_stats(dataset, dataset_name):
            # ✅ 여러 속성 체크
            view_counts = []
            
            if hasattr(dataset, 'sample_view_counts'):
                view_counts = list(dataset.sample_view_counts.values())
                print(f"\n{dataset_name}:")
                print(f"   📌 Using sample_view_counts")
            elif hasattr(dataset, 'instance_view_counts'):
                view_counts = list(dataset.instance_view_counts.values())
                print(f"\n{dataset_name}:")
                print(f"   📌 Using instance_view_counts")
            else:
                print(f"\n{dataset_name}:")
                print(f"   ⚠️  No view count data available")
                print(f"   Total samples: {len(dataset)}")
                return
            
            # ✅ Empty check
            if len(view_counts) == 0:
                print(f"   ⚠️  No valid instances found!")
                print(f"   Total samples: {len(dataset)}")
                if hasattr(dataset, 'valid_indices'):
                    print(f"   Valid indices: {len(dataset.valid_indices)}")
                return
            
            # Print statistics
            print(f"   Total instances:     {len(view_counts)}")
            print(f"   Avg views/instance:  {sum(view_counts)/len(view_counts):.1f}")
            print(f"   View range:          [{min(view_counts)}, {max(view_counts)}]")
            print(f"   Std deviation:       {np.std(view_counts):.1f}")
            
            if hasattr(dataset, 'repeat_factor'):
                print(f"   Repeat factor:       {dataset.repeat_factor}")
                print(f"   Total samples:       {len(dataset)} ({len(view_counts)} × {dataset.repeat_factor})")
            else:
                print(f"   Total samples:       {len(dataset)}")
            
            # Distribution
            if len(view_counts) > 0:
                print(f"\n   📊 View Count Distribution:")
                bins = [0, 3, 6, 9, 12, 15, 100]
                bin_labels = ['0-2', '3-5', '6-8', '9-11', '12-14', '15+']
                hist, _ = np.histogram(view_counts, bins=bins)
                
                for label, count in zip(bin_labels, hist):
                    if count > 0:
                        pct = count / len(view_counts) * 100
                        bar = '█' * int(pct / 5)
                        print(f"      {label:>6} views: {count:>3} instances ({pct:>5.1f}%) {bar}")
        
        # Train dataset
        print_view_stats(self.train_dataset, "🚂 Training Dataset")
        if hasattr(self, 'train_loader'):
            print(f"   Batches per epoch:   {len(self.train_loader)}")
        
        # Val dataset
        print_view_stats(self.val_dataset, "✅ Validation Dataset")
        
        print(f"{'='*70}\n")

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        # Epoch 시작 시 dataset 통계 출력 (첫 epoch만)
        if self.epoch == 0:
            self._print_dataset_statistics()

        # Metrics tracking
        metrics = {
            'loss': 0.0,
            'loss_coarse': 0.0,
            'loss_fine': 0.0,
            'psnr_coarse': 0.0,
            'psnr_fine': 0.0
        }
        
        pbar = tqdm(
            self.train_loader, 
            desc=f'Epoch [{self.epoch+1}/{self.config.training.num_epochs}]',
            ncols=140
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Debug mode for first batch
            is_debug_batch = DEBUG_MODE and batch_idx == 0
            
            if is_debug_batch:
                self._print_debug_header()
            
            # 1. Prepare data
            data = self._prepare_batch(batch, is_debug_batch)
            
            # 2. Sample rays
            rays_data = self._sample_rays(data, is_debug_batch)
            
            # 3. Forward pass (No more denormalization)
            outputs = self._forward_pass(data, rays_data, is_debug_batch)
            
            # 4. Compute loss (Use rays_data['tgt_rgb'] directly!)
            losses = self._compute_losses(outputs, rays_data['tgt_rgb'], is_debug_batch)
            
            # 5. Check for NaN/Inf (즉시 중단)
            if self._check_nan_inf(losses, outputs, rays_data, batch_idx):
                raise RuntimeError("Training stopped due to NaN/Inf")
            
            # 6. Backward pass
            self._backward_step(losses, is_debug_batch)
            
            # 7. Update metrics
            self._update_metrics(metrics, losses, batch_idx)
            
            # 8. Update progress bar
            self._update_progress_bar(pbar, metrics, losses, batch_idx)
            
            # 9. Logging and validation
            self._handle_logging_and_validation(data, losses)
            
            self.global_step += 1
        
        # Epoch summary
        return self._summarize_epoch(metrics)


    def _print_debug_header(self):
        """tqdm.write debug header"""
        tqdm.write(f"\n{'='*70}")
        tqdm.write(f"🐛 DEBUG MODE - First Batch of Epoch {self.epoch+1}")
        tqdm.write(f"{'='*70}")


    def _prepare_batch(self, batch, debug=False):
        """Prepare batch data and move to device"""
        data = {
            'src_images': batch['src_images'].to(self.device),
            'src_poses': batch['src_poses'].to(self.device),
            'src_intrinsics': batch['src_intrinsics'].to(self.device),
            'tgt_image': batch['tgt_image'].to(self.device),
            'tgt_pose': batch['tgt_pose'].to(self.device),
            'tgt_intrinsics': batch['tgt_intrinsics'].to(self.device),
            'B': batch['src_images'].shape[0],
            'H': self.config.data.image_size[0],
            'W': self.config.data.image_size[1]
        }
        
        if debug:
            check_tensor("src_images", data['src_images'])
            check_tensor("tgt_image", data['tgt_image'])
            check_tensor("src_intrinsics", data['src_intrinsics'])
        
        return data


    def _sample_rays(self, data, debug=False):
        """
        Sample rays with foreground bias
        
        IMPORTANT: Rays are in WORLD space!
        """
        # ✅ Config에서 fg_sampling_ratio 가져오기
        fg_ratio = getattr(self.config.training, 'fg_sampling_ratio', 0.8)
        fg_threshold = getattr(self.config.training, 'fg_threshold', 0.05)
        num_rays = self.config.training.num_rays_per_batch
        H, W = data['H'], data['W']
        
        rays_o_list, rays_d_list, tgt_rgb_list = [], [], []
        
        # Statistics for debug
        total_fg_pixels = 0
        total_bg_pixels = 0
        total_fg_sampled = 0
        total_bg_sampled = 0
        
        for i in range(data['B']):
            # Generate rays in WORLD space
            tgt_pose_c2w = data['tgt_pose'][i]
            
            rays_o, rays_d = CameraUtils.get_rays(
                H, W,
                data['tgt_intrinsics'][i],
                tgt_pose_c2w
            )
            
            # cuda로 맞추기
            device = rays_o.device

            if debug and i == 0:
                tqdm.write(f"\n🔍 Ray Sampling Debug (Batch {i}):")
                tqdm.write(f"  Target pose (w2c):\n{data['tgt_pose'][i]}")
                tqdm.write(f"  Target pose (c2w):\n{tgt_pose_c2w}")
                rays_o_flat = rays_o.reshape(-1, 3)
                check_tensor("rays_o (WORLD space)", rays_o_flat[:100])
                check_tensor("rays_d (WORLD space)", rays_d)
            
            # Detect foreground with config threshold
            tgt_img = data['tgt_image'][i]  # (3, H, W)
            fg_mask = self._detect_foreground(tgt_img, threshold=fg_threshold)
            
            # cuda로 맞추기
            if tgt_img.device != device:
                tgt_img = tgt_img.to(device)

            # Statistics
            n_fg_pixels = fg_mask.sum().item()
            n_bg_pixels = (~fg_mask).sum().item()
            total_fg_pixels += n_fg_pixels
            total_bg_pixels += n_bg_pixels
            
            
            fg_coords = torch.nonzero(fg_mask, as_tuple=False)
            bg_coords = torch.nonzero(~fg_mask, as_tuple=False)
            
            num_fg = int(num_rays * fg_ratio)
            num_bg = num_rays - num_fg
            
            # Sample foreground
            if len(fg_coords) > 0 and num_fg > 0:
                fg_idx = np.random.choice(len(fg_coords), size=min(num_fg, len(fg_coords)), replace=False)
                fg_sample_coords = fg_coords[fg_idx]
            else:
                fg_sample_coords = torch.empty(0, 2, dtype=torch.long)
            
            # Sample background
            if len(bg_coords) > 0 and num_bg > 0:
                bg_idx = np.random.choice(len(bg_coords), size=min(num_bg, len(bg_coords)), replace=False)
                bg_sample_coords = bg_coords[bg_idx]
            else:
                bg_sample_coords = torch.empty(0, 2, dtype=torch.long)
            
            # Statistics
            n_fg_sampled = len(fg_sample_coords)
            n_bg_sampled = len(bg_sample_coords)
            total_fg_sampled += n_fg_sampled
            total_bg_sampled += n_bg_sampled
            
            if debug and i == 0:
                tqdm.write(f"\n  🎲 Ray Sampling (Image {i}):")
                tqdm.write(f"    Target: {num_fg} FG rays + {num_bg} BG rays = {num_rays} total")
                tqdm.write(f"    Sampled: {n_fg_sampled} FG rays + {n_bg_sampled} BG rays")
                if n_fg_sampled < num_fg:
                    tqdm.write(f"    ⚠️  FG shortage: {num_fg - n_fg_sampled} rays (not enough FG pixels)")
                if n_bg_sampled < num_bg:
                    tqdm.write(f"    ⚠️  BG shortage: {num_bg - n_bg_sampled} rays (not enough BG pixels)")
            
            # Combine
            select_coords = torch.cat([fg_sample_coords, bg_sample_coords], dim=0)
            
            # Fill if needed
            if len(select_coords) < num_rays:
                shortage = num_rays - len(select_coords)
                if debug and i == 0:
                    tqdm.write(f"    📦 Filling {shortage} rays with random sampling...")
                
                all_coords = torch.stack(
                    torch.meshgrid(
                        torch.arange(H, device=device),
                        torch.arange(W, device=device),
                        indexing='ij'
                    ), dim=-1
                ).reshape(-1, 2)

                extra_idx_np = np.random.choice(len(all_coords), size=shortage, replace=False)
                extra_idx = torch.from_numpy(extra_idx_np).to(device=device, dtype=torch.long)
                extra_coords = all_coords[extra_idx]

                select_coords = torch.cat([select_coords, extra_coords], dim=0)
            
            # Extract rays and RGB
            rays_o_list.append(rays_o[select_coords[:, 0], select_coords[:, 1]])
            rays_d_list.append(rays_d[select_coords[:, 0], select_coords[:, 1]])
            tgt_rgb_list.append(tgt_img[:, select_coords[:, 0], select_coords[:, 1]].T)
        
        if debug:
            tqdm.write(f"\n  📈 Overall Statistics:")
            tqdm.write(f"    Total FG pixels: {total_fg_pixels} ({total_fg_pixels/(data['B']*H*W)*100:.1f}%)")
            tqdm.write(f"    Total BG pixels: {total_bg_pixels} ({total_bg_pixels/(data['B']*H*W)*100:.1f}%)")
            tqdm.write(f"    Total FG rays sampled: {total_fg_sampled} ({total_fg_sampled/(data['B']*num_rays)*100:.1f}%)")
            tqdm.write(f"    Total BG rays sampled: {total_bg_sampled} ({total_bg_sampled/(data['B']*num_rays)*100:.1f}%)")
            tqdm.write(f"{'='*70}\n")
        
        rays_data = {
            'rays_o': torch.stack(rays_o_list, dim=0),
            'rays_d': torch.stack(rays_d_list, dim=0),
            'tgt_rgb': torch.stack(tgt_rgb_list, dim=0)
        }
        
        return rays_data


    def _detect_foreground(self, image: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
        """
        Detect foreground in image
        
        Args:
            image: (3, H, W) in [0, 1]
            threshold: Color difference threshold (from config)
            
        Returns:
            fg_mask: (H, W) boolean mask
        """
        _, H, W = image.shape
        image_hw = image.permute(1, 2, 0)  # (H, W, 3)
        
        # Method 1: Edge-based background detection
        edge_pixels = []
        edge_pixels.append(image_hw[0, :, :])  # top
        edge_pixels.append(image_hw[-1, :, :])  # bottom
        edge_pixels.append(image_hw[:, 0, :])  # left
        edge_pixels.append(image_hw[:, -1, :])  # right
        
        edge_colors = torch.cat([p.reshape(-1, 3) for p in edge_pixels], dim=0)
        bg_color = edge_colors.mean(dim=0)  # (3,)
        
        # Compute color difference
        color_diff = (image_hw - bg_color).norm(dim=-1)  # (H, W)
        
        # Foreground = pixels different from background
        fg_mask = color_diff > threshold  # Use config threshold
        
        return fg_mask
        
    def _forward_pass(self, data, rays_data, debug=False):
        """Forward pass through model with comprehensive debugging"""
        
        # Forward pass
        try:
            outputs = self.model(
                data['src_images'],
                data['src_intrinsics'],
                data['src_poses'],
                rays_data['rays_o'],
                rays_data['rays_d'],
                self.config.data.z_near,
                self.config.data.z_far,   
                step=self.global_step  
            )
        except Exception as e:
            tqdm.write(f"\n❌ Exception during forward pass: {e}")
            if DEBUG_MODE:
                import traceback
                traceback.print_exc()
            raise
        
        # ✅ COMPREHENSIVE OUTPUT ANALYSIS
        if self.global_step % self.step_debug == 0:
            with torch.no_grad():
                tqdm.write(f"\n2️⃣ Network Output Analysis:")
                
                # Coarse network
                coarse_rgb = outputs['coarse']['rgb_map'][0]  # (N_rays, 3)
                coarse_acc = outputs['coarse']['acc_map'][0]  # (N_rays,)
                coarse_depth = outputs['coarse']['depth_map'][0]  # (N_rays,)
                
                tqdm.write(f"\n   📊 Coarse Network:")
                tqdm.write(f"   RGB output:")
                tqdm.write(f"     R: mean={coarse_rgb[:, 0].mean():.3f}, std={coarse_rgb[:, 0].std():.3f}")
                tqdm.write(f"     G: mean={coarse_rgb[:, 1].mean():.3f}, std={coarse_rgb[:, 1].std():.3f}")
                tqdm.write(f"     B: mean={coarse_rgb[:, 2].mean():.3f}, std={coarse_rgb[:, 2].std():.3f}")
                
                # Check for grayscale
                r_g_diff = (coarse_rgb[:, 0] - coarse_rgb[:, 1]).abs().mean()
                g_b_diff = (coarse_rgb[:, 1] - coarse_rgb[:, 2]).abs().mean()
                r_b_diff = (coarse_rgb[:, 0] - coarse_rgb[:, 2]).abs().mean()
                
                if r_g_diff < 0.01 and g_b_diff < 0.01 and r_b_diff < 0.01:
                    tqdm.write(f"   ❌ RGB channels IDENTICAL! (grayscale/uniform)")
                    tqdm.write(f"      R-G: {r_g_diff:.4f}, G-B: {g_b_diff:.4f}, R-B: {r_b_diff:.4f}")
                    tqdm.write(f"      → Network not learning color!")
                else:
                    tqdm.write(f"   ✅ RGB channels have variation")
                    tqdm.write(f"      R-G: {r_g_diff:.4f}, G-B: {g_b_diff:.4f}, R-B: {r_b_diff:.4f}")
                
                # Opacity analysis
                tqdm.write(f"\n   Accumulated Opacity:")
                tqdm.write(f"     Mean: {coarse_acc.mean():.3f}")
                tqdm.write(f"     Std: {coarse_acc.std():.3f}")
                tqdm.write(f"     Range: [{coarse_acc.min():.3f}, {coarse_acc.max():.3f}]")
                
                # Interpret opacity
                if coarse_acc.mean() < 0.1:
                    tqdm.write(f"   ❌ VERY LOW opacity! Scene is transparent!")
                    tqdm.write(f"      → Density bias too negative OR density not learned")
                elif coarse_acc.mean() < 0.3:
                    tqdm.write(f"   ⚠️  Low opacity. Still learning geometry")
                elif coarse_acc.mean() > 0.9:
                    tqdm.write(f"   ⚠️  VERY HIGH opacity! Scene is opaque everywhere")
                    tqdm.write(f"      → Might be overfitting or density bias too high")
                else:
                    tqdm.write(f"   ✅ Opacity in reasonable range")
                
                # Depth analysis
                tqdm.write(f"\n   Depth Map:")
                tqdm.write(f"     Mean: {coarse_depth.mean():.3f}")
                tqdm.write(f"     Std: {coarse_depth.std():.3f}")
                tqdm.write(f"     Range: [{coarse_depth.min():.3f}, {coarse_depth.max():.3f}]")
                
                scene_mid = (self.config.data.z_near + self.config.data.z_far) / 2
                if abs(coarse_depth.mean() - scene_mid) < 0.1:
                    tqdm.write(f"   ⚠️  Depth ≈ scene midpoint ({scene_mid:.2f})")
                    tqdm.write(f"      → Uniform depth (not learning geometry)")
                else:
                    tqdm.write(f"   ✅ Depth varies from midpoint")
                
                # Fine network
                fine_rgb = outputs['fine']['rgb_map'][0]
                fine_acc = outputs['fine']['acc_map'][0]
                
                tqdm.write(f"\n   📊 Fine Network:")
                tqdm.write(f"   RGB output:")
                tqdm.write(f"     R: mean={fine_rgb[:, 0].mean():.3f}, std={fine_rgb[:, 0].std():.3f}")
                tqdm.write(f"     G: mean={fine_rgb[:, 1].mean():.3f}, std={fine_rgb[:, 1].std():.3f}")
                tqdm.write(f"     B: mean={fine_rgb[:, 2].mean():.3f}, std={fine_rgb[:, 2].std():.3f}")
                
                r_g_diff_f = (fine_rgb[:, 0] - fine_rgb[:, 1]).abs().mean()
                g_b_diff_f = (fine_rgb[:, 1] - fine_rgb[:, 2]).abs().mean()
                r_b_diff_f = (fine_rgb[:, 0] - fine_rgb[:, 2]).abs().mean()
                
                if r_g_diff_f < 0.01 and g_b_diff_f < 0.01:
                    tqdm.write(f"   ❌ RGB channels IDENTICAL!")
                else:
                    tqdm.write(f"   ✅ RGB channels have variation")
                
                tqdm.write(f"\n   Opacity: mean={fine_acc.mean():.3f}, range=[{fine_acc.min():.3f}, {fine_acc.max():.3f}]")
                
                # Target comparison
                tgt_rgb = rays_data['tgt_rgb'][0]
                tqdm.write(f"\n3️⃣ Target vs Output:")
                tqdm.write(f"   Target RGB range: [{tgt_rgb.min():.3f}, {tgt_rgb.max():.3f}]")
                tqdm.write(f"   Output RGB range: [{fine_rgb.min():.3f}, {fine_rgb.max():.3f}]")
                tqdm.write(f"   Target mean: {tgt_rgb.mean():.3f}")
                tqdm.write(f"   Output mean: {fine_rgb.mean():.3f}")
                
                # Check if output stuck at extremes
                if fine_rgb.mean() < 0.1:
                    tqdm.write(f"   ❌ Output very dark! (mean={fine_rgb.mean():.3f})")
                elif fine_rgb.mean() > 0.9:
                    tqdm.write(f"   ⚠️  Output very bright! (mean={fine_rgb.mean():.3f})")
                
                tqdm.write(f"{'='*70}\n")
        
        return outputs

    def _visualize_output(self, data, debug=False):
        """
        Visualize comprehensive training progress (4x2 Layout)
        Row 1: Source | Target
        Row 2: Coarse RGB | Fine RGB
        Row 3: Coarse Depth | Fine Depth
        Row 4: Coarse Acc | Fine Acc
        """
        if debug:
            tqdm.write(f"\n{'='*70}")
            tqdm.write(f"🎨 VISUALIZATION (Step {self.global_step})")
            tqdm.write(f"{'='*70}")
        
        with torch.no_grad():
            # Use smaller resolution for speed
            H_viz, W_viz = 128, 128
            
            # Get full image rays
            rays_o, rays_d = CameraUtils.get_rays(
                H_viz, W_viz,
                data['tgt_intrinsics'][0],
                data['tgt_pose'][0]
            )
            
            rays_o_flat = rays_o.reshape(-1, 3).unsqueeze(0)
            rays_d_flat = rays_d.reshape(-1, 3).unsqueeze(0)
            
            # Render in chunks to avoid OOM
            chunk_size = 512
            
            # 리스트 초기화 (Acc 추가)
            coarse_chunks = []
            fine_chunks = []
            depth_coarse_chunks = []
            depth_fine_chunks = []
            acc_coarse_chunks = []  # ✅ 추가됨
            acc_fine_chunks = []    # ✅ 추가됨
            
            for i in range(0, rays_o_flat.shape[1], chunk_size):
                rays_o_chunk = rays_o_flat[:, i:i+chunk_size]
                rays_d_chunk = rays_d_flat[:, i:i+chunk_size]
                
                outputs_viz = self.model(
                    data['src_images'][0:1],
                    data['src_intrinsics'][0:1],
                    data['src_poses'][0:1],
                    rays_o_chunk,
                    rays_d_chunk,
                    self.config.data.z_near,
                    self.config.data.z_far
                )
                
                # RGB & Depth 수집
                coarse_chunks.append(outputs_viz['coarse']['rgb_map'])
                fine_chunks.append(outputs_viz['fine']['rgb_map'])
                depth_coarse_chunks.append(outputs_viz['coarse']['depth_map'])
                depth_fine_chunks.append(outputs_viz['fine']['depth_map'])
                
                # ✅ Acc Map (Opacity) 수집
                acc_coarse_chunks.append(outputs_viz['coarse']['acc_map'])
                acc_fine_chunks.append(outputs_viz['fine']['acc_map'])
            
            # Combine chunks & Reshape
            coarse_rgb = torch.cat(coarse_chunks, dim=1)[0].reshape(H_viz, W_viz, 3).cpu().numpy()
            fine_rgb = torch.cat(fine_chunks, dim=1)[0].reshape(H_viz, W_viz, 3).cpu().numpy()
            
            depth_coarse = torch.cat(depth_coarse_chunks, dim=1)[0].reshape(H_viz, W_viz).cpu().numpy()
            depth_fine = torch.cat(depth_fine_chunks, dim=1)[0].reshape(H_viz, W_viz).cpu().numpy()
            
            # ✅ Acc Map 결합
            acc_coarse = torch.cat(acc_coarse_chunks, dim=1)[0].reshape(H_viz, W_viz).cpu().numpy()
            acc_fine = torch.cat(acc_fine_chunks, dim=1)[0].reshape(H_viz, W_viz).cpu().numpy()
            
            # Clip RGB to [0, 1]
            coarse_rgb = np.clip(coarse_rgb, 0, 1)
            fine_rgb = np.clip(fine_rgb, 0, 1)
            
            # Denormalize source image
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(data['src_images'].device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(data['src_images'].device)
            source_vis = (data['src_images'][0, 0] * std + mean).clamp(0, 1)
            
            # Interpolate Source & Target to viz size
            source_img = F.interpolate(
                source_vis.unsqueeze(0),
                size=(H_viz, W_viz),
                mode='bilinear',
                align_corners=False
            )[0].permute(1, 2, 0).cpu().numpy()
            
            target_img = F.interpolate(
                data['tgt_image'][0:1],
                size=(H_viz, W_viz),
                mode='bilinear',
                align_corners=False
            )[0].permute(1, 2, 0).cpu().numpy()
            target_img = np.clip(target_img, 0, 1)
            
            # PSNR Calculation
            coarse_psnr = -10 * np.log10(np.mean((coarse_rgb - target_img) ** 2) + 1e-8)
            fine_psnr = -10 * np.log10(np.mean((fine_rgb - target_img) ** 2) + 1e-8)

            if debug:
                tqdm.write(f"  Source shape: {source_img.shape}")
                tqdm.write(f"  Target shape: {target_img.shape}")
                tqdm.write(f"  Depth range: [{depth_fine.min():.3f}, {depth_fine.max():.3f}]")
                tqdm.write(f"  Acc range: [{acc_fine.min():.3f}, {acc_fine.max():.3f}]")
        
        # --- 4x2 Plotting ---
        fig, axes = plt.subplots(4, 2, figsize=(10, 20))
        
        # Row 1: Source | Target
        axes[0, 0].imshow(source_img)
        axes[0, 0].set_title(f'Source', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(target_img)
        axes[0, 1].set_title(f'Target GT', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Row 2: Coarse RGB | Fine RGB
        axes[1, 0].imshow(coarse_rgb)
        axes[1, 0].set_title(f'Coarse RGB ({coarse_psnr:.2f} dB)', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(fine_rgb)
        axes[1, 1].set_title(f'Fine RGB ({fine_psnr:.2f} dB)', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Row 3: Coarse Depth | Fine Depth
        im_cd = axes[2, 0].imshow(depth_coarse, cmap='plasma')
        axes[2, 0].set_title(f'Coarse Depth', fontsize=12, fontweight='bold')
        axes[2, 0].axis('off')
        plt.colorbar(im_cd, ax=axes[2, 0], fraction=0.046, pad=0.04)
        
        im_fd = axes[2, 1].imshow(depth_fine, cmap='plasma')
        axes[2, 1].set_title(f'Fine Depth', fontsize=12, fontweight='bold')
        axes[2, 1].axis('off')
        plt.colorbar(im_fd, ax=axes[2, 1], fraction=0.046, pad=0.04)
        
        # Row 4: Coarse Acc | Fine Acc (불투명도)
        im_ca = axes[3, 0].imshow(acc_coarse, cmap='gray', vmin=0, vmax=1)
        axes[3, 0].set_title(f'Coarse Acc (Opacity)', fontsize=12, fontweight='bold')
        axes[3, 0].axis('off')
        plt.colorbar(im_ca, ax=axes[3, 0], fraction=0.046, pad=0.04)
        
        im_fa = axes[3, 1].imshow(acc_fine, cmap='gray', vmin=0, vmax=1)
        axes[3, 1].set_title(f'Fine Acc (Opacity)', fontsize=12, fontweight='bold')
        axes[3, 1].axis('off')
        plt.colorbar(im_fa, ax=axes[3, 1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Save
        viz_dir = os.path.join(self.config.training.checkpoint_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        save_path = os.path.join(viz_dir, f'step_{self.global_step:06d}.png')
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.close(fig)
        
        if debug:
            tqdm.write(f"  💾 Saved: {save_path}")
            tqdm.write(f"  📊 Coarse PSNR: {coarse_psnr:.2f} dB")
            tqdm.write(f"  📊 Fine PSNR: {fine_psnr:.2f} dB")
            tqdm.write(f"{'='*70}\n")

    def _compute_losses(self, outputs, tgt_rgb, debug=False):
        """
        Compute coarse and fine losses with Entropy Regularization
        
        Args:
            outputs: Model outputs
            tgt_rgb: Target RGB (already in [0,1])
        """

        # 1. Color Loss (MSE)
        loss_coarse = self.criterion(outputs['coarse']['rgb_map'], tgt_rgb) 
        loss_fine = self.criterion(outputs['fine']['rgb_map'], tgt_rgb) 

        # --- 가중치 스케줄링 --- : 1.0이 최적 
        coarse_weight = 1.0
        #coarse_weight = 0.1 # coarse의 학습 비중을 낮추고 fine쪽에 높여 선명도 확보 시도 

        # 2. Density Regularization (기존 코드 유지)
        density_reg_coarse = outputs.get('density_reg_coarse', torch.tensor(0.0, device=tgt_rgb.device))
        density_reg_fine = outputs.get('density_reg_fine', torch.tensor(0.0, device=tgt_rgb.device))

        density_reg_total = density_reg_coarse + density_reg_fine

       
        current_density_lambda = self.config.model.density_lambda 
        
        # 3. [핵심 수정] Background Transparency Regularization (배경 투명화 강제)
        # 입력에 알파 채널이 없으므로, Target RGB가 "거의 흰색"인 부분을 배경으로 간주합니다.
        
        # (1) 배경 마스크 생성: RGB가 모두 0.95 이상인 픽셀을 배경으로 추정
        # 실제 데이터가 완벽한 (1,1,1)이 아닐 수 있으므로 0.95~0.98 정도의 임계값 사용
        bg_mask = (tgt_rgb > 0.95).all(dim=-1).float() 
        
        # (2) 배경에서의 Opacity Penalty
        # 배경이라고 추정된 픽셀들(bg_mask=1)의 acc_map(불투명도) 합계를 Loss로 추가
        # 즉, "흰색 배경 부분에서는 불투명도가 0이 되어야 한다"고 강제함
        acc_coarse = outputs['coarse']['acc_map']
        acc_fine = outputs['fine']['acc_map']
        
        
        loss_transparency_coarse = (acc_coarse * bg_mask).mean()
        loss_transparency_fine = (acc_fine * bg_mask).mean()
        
        # 배경 날리기 가중치 
        transparency_lambda = 0.01  
        
        # Fine weights 가져오기 (B, N_rays, N_samples)
        weights = outputs['fine']['weights'] 

        # Entropy 계산: P * log(P) 형태
        # weights가 퍼져있으면(안개) Entropy가 높고, 한곳에 쏠려있으면(선명) 낮음
        entropy_loss = -torch.sum(weights * torch.log(weights + 1e-10), dim=-1).mean()

        # 가중치 설정 (보통 0.001 ~ 0.01 사이 사용)
        entropy_lambda = 0.001

        # --- 최종 Loss 합산 ---
        loss_total = (coarse_weight * loss_coarse) + loss_fine \
                   + (current_density_lambda * density_reg_total) \
                   + (entropy_lambda * entropy_loss) \
                   + (transparency_lambda * (loss_transparency_coarse + loss_transparency_fine))

        # Compute PSNR
        with torch.no_grad():
            psnr_coarse = mse2psnr(loss_coarse)
            psnr_fine = mse2psnr(loss_fine)
        
        losses = {
            'coarse': loss_coarse,
            'fine': loss_fine,
            'density_reg': density_reg_coarse + density_reg_fine,
            'transparency': 0, # loss_transparency_fine, # 로깅용 추가
            'total': loss_total,
            'psnr_coarse': psnr_coarse,
            'psnr_fine': psnr_fine
        }
        
        if self.global_step % self.step_debug == 0:
            # [수정] 여러 줄로 나눠진 로그를 한 줄로 요약
            tqdm.write(
                f"[Step {self.global_step}] "
                f"Loss: {loss_total.item():.4f} (C: {loss_coarse.item():.4f}, F: {loss_fine.item():.4f}) | "
                f"PSNR: {psnr_fine.item():.2f} dB"
            )
            
            # Density 통계는 중요하므로 간략하게 유지 (옵션)
            if 'raw_density' in outputs['fine']:
                density_mean = outputs['fine']['raw_density'].mean().item()
                if density_mean > 0.0: # 이상 징후일 때만 출력하도록 조건 추가 가능
                     tqdm.write(f"  🌫️ Avg Density: {density_mean:.3f}")

        return losses
    

    def _check_nan_inf(self, losses, outputs, rays_data, batch_idx):
        """Check for NaN/Inf in losses"""
        loss = losses['total']
        
        if torch.isnan(loss) or torch.isinf(loss):
            tqdm.write(f"\n{'='*70}")
            tqdm.write(f"❌ NaN/Inf DETECTED at Epoch {self.epoch+1}, Batch {batch_idx}!")
            tqdm.write(f"{'='*70}")
            tqdm.write(f"  Loss coarse: {losses['coarse'].item()}")
            tqdm.write(f"  Loss fine: {losses['fine'].item()}")
            tqdm.write(f"  Total loss: {loss.item()}")
            
            # Detailed debugging
            tqdm.write(f"\n📊 Detailed Debug Info:")
            check_tensor("Coarse RGB output", outputs['coarse']['rgb_map'])
            check_tensor("Fine RGB output", outputs['fine']['rgb_map'])
            check_tensor("Target RGB", rays_data['tgt_rgb'])
            check_tensor("Rays origin", rays_data['rays_o'])
            check_tensor("Rays direction", rays_data['rays_d'])
            
            tqdm.write(f"\n💡 Possible causes:")
            tqdm.write(f"  1. Learning rate too high")
            tqdm.write(f"  2. Gradient explosion")
            tqdm.write(f"  3. Invalid camera parameters")
            tqdm.write(f"  4. Numerical instability in rendering")
            
            return True
        
        return False


    def _backward_step(self, losses, debug=False):
        """Backward pass and optimizer step"""
        self.optimizer.zero_grad()
        losses['total'].backward()

        # [수정] 레이어별 Gradient 출력 루프 삭제 -> Total Norm만 계산
        if self.global_step % self.step_debug == 0:
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            # [수정] Gradient 요약 정보만 간결하게 출력
            tqdm.write(f"\n[Step {self.global_step}] 📉 Grad Norm: {total_norm:.4f}")
            if total_norm > 10.0:
                tqdm.write("  ⚠️ High Gradient detected (will be clipped)")

        # Gradient clipping
        # [수정] Config에서 설정한 grad_clip 값 사용 (기본값 1.0)
        clip_val = getattr(self.config.optimizer, 'grad_clip', 1.0)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            max_norm=clip_val 
        )
        
        if self.global_step % self.step_debug == 0:
            tqdm.write(f"\n  Clipped grad norm: {grad_norm:.6f}")
            tqdm.write(f"{'='*70}\n")

        self.optimizer.step()

    def _update_metrics(self, metrics, losses, batch_idx):
        """Update running metrics"""
        metrics['loss'] += losses['total'].item()
        metrics['loss_coarse'] += losses['coarse'].item()
        metrics['loss_fine'] += losses['fine'].item()
        metrics['psnr_coarse'] += losses['psnr_coarse'].item()
        metrics['psnr_fine'] += losses['psnr_fine'].item()


    def _update_progress_bar(self, pbar, metrics, losses, batch_idx):
        """Update progress bar with instance tracking"""
        n = batch_idx + 1
        
        # Calculate ETA
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            total_steps = self.config.training.num_epochs * len(self.train_loader)
            steps_done = self.epoch * len(self.train_loader) + n
            eta_seconds = (elapsed / steps_done) * (total_steps - steps_done)
            eta = str(timedelta(seconds=int(eta_seconds)))
        else:
            eta = "calculating..."
        
        # ✅ 현재 배치의 첫 번째 샘플 인덱스
        batch_size = self.config.training.batch_size
        samples_done = (self.epoch * len(self.train_loader) + batch_idx) * batch_size
        current_sample_idx = samples_done % len(self.train_dataset)
        
        # ✅ Dataset에서 instance 정보 가져오기
        if hasattr(self.train_dataset, 'sample_to_instance_info'):
            info = self.train_dataset.sample_to_instance_info.get(current_sample_idx)
            
            if info is not None:
                inst_idx, repeat_num, total_views = info
                num_instances = len(self.train_dataset.valid_indices)
                
                # ✅ 수정: 실제 view 개수 표시
                desc = (
                    f'E[{self.epoch+1}/{self.config.training.num_epochs}] '
                    f'I {inst_idx+1}/{num_instances} '  # ✅ 1-based index
                    f'R{repeat_num+1}/{self.train_dataset.repeat_factor} '  # ✅ 1-based
                    f'V{total_views}'  # ✅ Total valid views
                )
            else:
                desc = f'E[{self.epoch+1}/{self.config.training.num_epochs}]'
        else:
            desc = f'E[{self.epoch+1}/{self.config.training.num_epochs}]'

        pbar.set_description(desc)

        current_lr = self.optimizer.param_groups[0]['lr']
        
        pbar.set_postfix({
            'lr': f'{current_lr:.2e}',  # ✅ LR 추가
            'loss': f'{metrics["loss"] / n:.4f}',
            'c_psnr': f'{metrics["psnr_coarse"] / n:.2f}',
            'f_psnr': f'{metrics["psnr_fine"] / n:.2f}',
            'ETA': eta
        })


    def _handle_logging_and_validation(self, data, losses):
        """Handle logging and validation"""
        # Step-based logging
        if self.should_log_step():
            self.writer.add_scalar('train_step/loss', losses['total'].item(), self.global_step)
            self.writer.add_scalar('train_step/loss_coarse', losses['coarse'].item(), self.global_step)
            self.writer.add_scalar('train_step/loss_fine', losses['fine'].item(), self.global_step)
            self.writer.add_scalar('train_step/psnr_coarse', losses['psnr_coarse'].item(), self.global_step)
            self.writer.add_scalar('train_step/psnr_fine', losses['psnr_fine'].item(), self.global_step)
            self.writer.add_scalar('train_step/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            # Log sample images (first time only)
            if self.global_step == 0 or (self.global_step == self.epoch * len(self.train_loader)):
                with torch.no_grad():
                    self.writer.add_images('train/source_image', 
                                          data['src_images'][0, 0:1], self.global_step)
                    self.writer.add_images('train/target_image', 
                                          data['tgt_image'][0:1], self.global_step)
        
        # Step-based validation
        if self.should_validate_step():
            # [수정] 3개 값(Loss, PSNR, SSIM) 받아오기
            val_loss, val_psnr, val_ssim = self.validate()
            
            self.writer.add_scalar('val_step/loss', val_loss, self.global_step)
            self.writer.add_scalar('val_step/psnr', val_psnr, self.global_step)
            self.writer.add_scalar('val_step/ssim', val_ssim, self.global_step) # ✅ [추가] TensorBoard 기록
            
            # Save best model
            if val_psnr > self.best_val_psnr:
                self.best_val_psnr = val_psnr
                self.best_val_loss = val_loss
                self.best_val_ssim = val_ssim # [추가] 최신 기록 갱신
                self.save_checkpoint('best_model.pth')
                tqdm.write(f"\n✓ New best model! Val PSNR: {val_psnr:.2f} dB, SSIM: {val_ssim:.3f}") # ✅ 로그 수정
            
            self.model.train()
        
        # Step-based checkpoint saving
        if self.should_save_step():
            self.save_checkpoint(f'checkpoint_step_{self.global_step}.pth')


    def _summarize_epoch(self, metrics):
        """Summarize epoch results"""
        n = len(self.train_loader)
        
        avg_loss = metrics['loss'] / n
        avg_loss_coarse = metrics['loss_coarse'] / n
        avg_loss_fine = metrics['loss_fine'] / n
        avg_psnr_coarse = metrics['psnr_coarse'] / n
        avg_psnr_fine = metrics['psnr_fine'] / n
        
        tqdm.write(f"\nEpoch {self.epoch+1} Summary:")
        tqdm.write(f"  Avg Loss: {avg_loss:.4f} (Coarse: {avg_loss_coarse:.4f}, Fine: {avg_loss_fine:.4f})")
        tqdm.write(f"  Avg PSNR: {avg_psnr_fine:.2f} dB (Coarse: {avg_psnr_coarse:.2f} dB, Fine: {avg_psnr_fine:.2f} dB)")
        
        return avg_loss, avg_psnr_fine
    
    @torch.no_grad()
    def validate(self):
        """Validate the model with SSIM and Multi-sample Visualization"""
        self.model.eval()
        
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        num_batches = 0
        
        # 시각화할 샘플들을 담을 리스트
        vis_samples = []
        
        # 시각화할 배치 인덱스 (고정된 차들을 비교하기 위함)
        b_size = getattr(self.config.training,"batch_size", 4)
        val_len = getattr(self.config.data, "num_val_instances", 20)
        vis_batch_indices =  [i for i in range(int(val_len//b_size))]
        
        val_pbar = tqdm(self.val_loader, desc='Validating', leave=False, ncols=100)
        
        for batch_idx, batch in enumerate(val_pbar):
            # Move to device
            src_images = batch['src_images'].to(self.device)     # (B, V, 3, H, W)
            src_poses = batch['src_poses'].to(self.device)
            src_intrinsics = batch['src_intrinsics'].to(self.device)
            tgt_image = batch['tgt_image'].to(self.device)       # (B, 3, H, W)
            tgt_pose = batch['tgt_pose'].to(self.device)
            tgt_intrinsics = batch['tgt_intrinsics'].to(self.device)
            
            B = src_images.shape[0]
            H, W = self.config.data.image_size
            
            # Render full image (Chunking)
            rendered_images = []
            
            for i in range(B):
                rays_o, rays_d = CameraUtils.get_rays(
                    H, W, tgt_intrinsics[i], tgt_pose[i]
                )
                
                rays_o_flat = rays_o.reshape(-1, 3)
                rays_d_flat = rays_d.reshape(-1, 3)
                
                chunk_size = self.config.training.chunk_size
                rgb_chunks = []
                
                for j in range(0, rays_o_flat.shape[0], chunk_size):
                    rays_o_chunk = rays_o_flat[j:j+chunk_size].unsqueeze(0)
                    rays_d_chunk = rays_d_flat[j:j+chunk_size].unsqueeze(0)
                    
                    outputs = self.model(
                        src_images[i:i+1],
                        src_intrinsics[i:i+1],
                        src_poses[i:i+1],
                        rays_o_chunk,
                        rays_d_chunk,
                        self.config.data.z_near,
                        self.config.data.z_far
                    )
                    rgb_chunks.append(outputs['fine']['rgb_map'])
                
                rgb_pred = torch.cat(rgb_chunks, dim=1).reshape(H, W, 3)
                rendered_images.append(rgb_pred)
            
            rendered_images = torch.stack(rendered_images, dim=0) # (B, H, W, 3)
            tgt_image_hwc = tgt_image.permute(0, 2, 3, 1)         # (B, H, W, 3)
            
            # --- Metrics Calculation ---
            loss = self.criterion(rendered_images, tgt_image_hwc)
            psnr = mse2psnr(loss)
            
            # SSIM (B, 3, H, W) 형태로 변환 후 계산
            pred_permute = rendered_images.permute(0, 3, 1, 2).clamp(0, 1)
            tgt_permute = tgt_image.clamp(0, 1)
            ssim_val = compute_ssim(pred_permute, tgt_permute)

            total_loss += loss.item()
            total_psnr += psnr.item()
            total_ssim += ssim_val.item()
            num_batches += 1
            
            # [수정] 시각화 샘플 수집 (Source 전체 뷰 저장 + 개별 Metric 저장)
            if batch_idx in vis_batch_indices:
                # 현재 배치의 첫 번째 샘플(index 0)에 대한 Metric 따로 계산 (정확한 표기를 위해)
                # (배치 전체 평균이 아니라, 보여지는 저 이미지 한 장의 점수)
                sample_loss = F.mse_loss(rendered_images[0], tgt_image_hwc[0])
                sample_psnr = mse2psnr(sample_loss).item()
                sample_ssim = compute_ssim(pred_permute[0:1], tgt_permute[0:1]).item()
                
                vis_samples.append({
                    'batch_idx': batch_idx,
                    'source': src_images[0], # (V, 3, H, W) 모든 소스 뷰
                    'target': tgt_image[0],  # (3, H, W)
                    'rendered': rendered_images[0].permute(2, 0, 1).clamp(0, 1), # (3, H, W)
                    'psnr': sample_psnr,     # 이 이미지의 PSNR
                    'ssim': sample_ssim      # 이 이미지의 SSIM
                })

            val_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'psnr': f'{psnr.item():.2f}',
                'ssim': f'{ssim_val.item():.3f}'
            })
            
        avg_loss = total_loss / num_batches
        avg_psnr = total_psnr / num_batches
        avg_ssim = total_ssim / num_batches
        
        tqdm.write(f'\n  Validation - Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.3f}')
        
        if len(vis_samples) > 0:
            self._log_validation_images(vis_samples)
        
        return avg_loss, avg_psnr, avg_ssim


    def _log_validation_images(self, samples_list):
        """
        Log validation images in a Single Grid Image
        Layout:
        Row 1: [Source 1] [Source 2] ... [Target] [Rendered]  (Car A)
        Row 2: [Source 1] [Source 2] ... [Target] [Rendered]  (Car B)
        ...
        """
        import matplotlib.pyplot as plt
        
        save_dir = os.path.join(self.config.training.checkpoint_dir, 'val_images')
        os.makedirs(save_dir, exist_ok=True)
        
        if len(samples_list) == 0:
            return

        # Denormalization constants
        device = samples_list[0]['target'].device
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

        # --- Grid 설정 ---
        num_samples = len(samples_list)
        num_source_views = samples_list[0]['source'].shape[0]
        
        # 컬럼 수: 소스 뷰 개수 + 타겟(1) + 렌더링(1)
        num_cols = num_source_views + 2 
        
        # 이미지 크기 설정 (샘플 수에 따라 세로로 길어짐)
        fig_width = 3 * num_cols
        fig_height = 3.5 * num_samples 
        
        # squeeze=False: 샘플이 1개여도 2차원 배열로 접근하기 위함
        fig, axes = plt.subplots(num_samples, num_cols, figsize=(fig_width, fig_height), squeeze=False)
        
        # 전체 제목
        fig.suptitle(f"Epoch: {self.epoch+1} Validation Summary", fontsize=16, fontweight='bold', y=0.99)

        for row_idx, sample in enumerate(samples_list):
            sources = sample['source'] # (V, 3, H, W)
            target = sample['target']  # (3, H, W)
            rendered = sample['rendered'] # (3, H, W)
            
            # 1. Source Views 그리기
            for v in range(num_source_views):
                src_vis = (sources[v] * std + mean).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
                ax = axes[row_idx, v]
                ax.imshow(src_vis)
                if row_idx == 0: # 첫 번째 행에만 컬럼 제목 표시
                    ax.set_title(f"Source View {v+1}", fontsize=12)
                ax.axis('off')
                
                # 왼쪽 첫 열에 배치 정보 표시 (세로 텍스트)
                if v == 0:
                    ax.text(-0.1, 0.5, f"Batch {sample['batch_idx']}", 
                            transform=ax.transAxes, va='center', ha='right', rotation=90, fontsize=12, fontweight='bold')

            # 2. Target GT 그리기
            tgt_vis = target.cpu().permute(1, 2, 0).numpy()
            ax_tgt = axes[row_idx, num_source_views] # 뒤에서 두 번째
            ax_tgt.imshow(tgt_vis)
            if row_idx == 0:
                ax_tgt.set_title("Target GT", fontsize=12, fontweight='bold')
            ax_tgt.axis('off')
            
            # 3. Rendered Output 그리기
            pred_vis = rendered.cpu().permute(1, 2, 0).numpy()
            ax_pred = axes[row_idx, num_source_views + 1] # 마지막
            ax_pred.imshow(pred_vis)
            
            # 결과 이미지 위에 점수 표시
            title_text = "Rendered" if row_idx == 0 else ""
            score_text = f"PSNR: {sample['psnr']:.2f}\nSSIM: {sample['ssim']:.3f}"
            
            if row_idx == 0:
                ax_pred.set_title(f"{title_text}\n{score_text}", fontsize=10, fontweight='bold', color='blue')
            else:
                ax_pred.set_title(score_text, fontsize=10, fontweight='bold', color='blue')
                
            ax_pred.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.98]) # Title 공간 확보
        
        # --- Save to File ---
        # 파일명 하나로 통일 (Epoch 번호만 사용)
        file_name = f'epoch_{self.epoch+1:03d}_summary.png'
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path, dpi=100)
        
        plt.close(fig)
        
        tqdm.write(f"  📸 Validation summary saved: {file_name}")
    
    def train(self):
        """Main training loop"""
        tqdm.write(f"{'='*70}")
        tqdm.write(f"Starting training for {self.config.training.num_epochs} epochs")
        tqdm.write(f"Total steps: {self.config.training.num_epochs * len(self.train_loader):,}")
        tqdm.write(f"{'='*70}\n")
        
        # Training Configuration
        tqdm.write("\n📋 Training Configuration:")
        tqdm.write("─" * 70)
        tqdm.write(f"  {'Batch size:':<25} {self.config.training.batch_size}")
        tqdm.write(f"  {'Number of epochs:':<25} {self.config.training.num_epochs}")
        tqdm.write(f"  {'Learning rate:':<25} {self.config.training.learning_rate}")
        tqdm.write(f"  {'Weight decay:':<25} {self.config.training.weight_decay}")
        tqdm.write(f"  {'Rays per batch:':<25} {self.config.training.num_rays_per_batch}")
        tqdm.write("─" * 70)
        tqdm.write(f"  {'Coarse samples:':<25} {self.config.model.n_coarse}")
        tqdm.write(f"  {'Fine samples:':<25} {self.config.model.n_fine}")
        tqdm.write("─" * 70)
        
        # Optimizer info
        tqdm.write(f"\n🔧 Optimizer: {self.optimizer.__class__.__name__}")
        tqdm.write(f"  {'Total parameters:':<25} {sum(p.numel() for p in self.model.parameters()):,}")
        tqdm.write(f"  {'Trainable parameters:':<25} {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        self.start_time = time.time()
        
        for epoch in range(self.epoch, self.config.training.num_epochs):
            self.epoch = epoch
            
            epoch_loss, epoch_psnr = self.train_epoch()
            self.scheduler.step()
            
            # Epoch-based logging
            if self.config.training.log_every_n_epochs > 0:
                if (epoch + 1) % self.config.training.log_every_n_epochs == 0:
                    self.writer.add_scalar('train_epoch/loss', epoch_loss, epoch)
                    self.writer.add_scalar('train_epoch/psnr', epoch_psnr, epoch)
                    self.writer.add_scalar('train_epoch/lr', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Epoch-based validation
            if self.should_validate_epoch():
                val_loss, val_psnr, val_ssim = self.validate()
                
                self.writer.add_scalar('val_epoch/loss', val_loss, epoch)
                self.writer.add_scalar('val_epoch/psnr', val_psnr, epoch)
                self.writer.add_scalar('val_epoch/ssim', val_ssim, epoch) 
                
                # Save best model
                if val_psnr > self.best_val_psnr:
                    self.best_val_psnr = val_psnr
                    self.best_val_loss = val_loss
                    self.best_val_ssim = val_ssim # ✅ [추가]
                    self.save_checkpoint('best_model.pth')
                    tqdm.write(f"\n✓ New best model! Val PSNR: {val_psnr:.2f} dB, SSIM: {val_ssim:.3f}")
            
            # Epoch-based checkpoint saving
            if self.should_save_epoch():
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
            
            # tqdm.write progress
            elapsed = time.time() - self.start_time
            tqdm.write(f"  Time elapsed: {str(timedelta(seconds=int(elapsed)))}")
            # [수정] 로그에 SSIM 표시 추가
            tqdm.write(f"  Best val PSNR: {self.best_val_psnr:.2f} dB, SSIM: {self.best_val_ssim:.3f}")
            tqdm.write(f"{'-'*70}\n")
        
        total_time = time.time() - self.start_time
        tqdm.write(f"\n{'='*70}")
        tqdm.write(f"Training completed!")
        tqdm.write(f"Total time: {str(timedelta(seconds=int(total_time)))}")
        tqdm.write(f"Best validation PSNR: {self.best_val_psnr:.2f} dB")
        tqdm.write(f"Best validation loss: {self.best_val_loss:.4f}")
        tqdm.write(f"Best validation SSIM: {self.best_val_ssim:.3f}")
        tqdm.write(f"{'='*70}\n")
    

    def save_checkpoint(self, filename: str):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_psnr': self.best_val_psnr,
            'config': self.config
        }
        
        path = os.path.join(self.config.training.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        if 'best' in filename or 'epoch' in filename:
            tqdm.write(f'  Checkpoint saved: {filename}')
    
    def load_checkpoint(self, path: str, load_optimizer: bool = False):
        """
        Load checkpoint with LR Precedence Logic:
        1. --lr Option (Override)
        2. Checkpoint State (Resume)
        3. Config Default (Fresh Start - handled in __init__)
        """
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except Exception as e:
            tqdm.write(f"❌ Failed to load checkpoint: {e}")
            return

        # 1. 모델 가중치 로드 (항상 실행)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 2. 학습 진행 상황 복원
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_psnr = checkpoint.get('best_val_psnr', 0.0)
        self.best_val_ssim = checkpoint.get('best_val_ssim', 0.0) # SSIM이 없을 경우 대비
        
        tqdm.write(f'Checkpoint loaded from {path}')
        tqdm.write(f'Resuming from epoch {self.epoch}, step {self.global_step}')

        # ------------------------------------------------------------------
        # 학습률 및 옵티마이저 로드 우선순위 로직
        # ------------------------------------------------------------------
        override_lr = getattr(self.config.training, 'override_lr', None)

        if override_lr is not None:
            # Case 1: --lr 옵션이 있는 경우 (최우선)
            tqdm.write(f"\n{'='*45}")
            tqdm.write(f"   LR OVERRIDE ACTIVATED: {override_lr}")
            tqdm.write(f"{'='*45}")

            # (선택) Optimizer 상태는 로드하되 LR만 바꿀 것인가, 아예 초기화할 것인가?
            # 여기서는 'momentum' 정보를 살리기 위해 로드는 하되 LR을 강제 변경합니다.
            if load_optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 파라미터 그룹별로 LR 강제 적용 (PixelNeRF 비율 유지)
            # Group 0: Coarse (1.0x)
            # Group 1: Fine (1.0x)
            # Group 2: Encoder (0.1x)
            self.optimizer.param_groups[0]['lr'] = override_lr * 1.0
            self.optimizer.param_groups[1]['lr'] = override_lr * 1.0
            if len(self.optimizer.param_groups) > 2:
                self.optimizer.param_groups[2]['lr'] = override_lr * 0.1
            
            tqdm.write(f"     Optimizer LR updated to: {override_lr} (Encoder: {override_lr*0.1:.2e})")

            # Scheduler는 무조건 리셋 (새로운 LR에 맞춰서 다시 시작)
            scheduler_type = getattr(self.config.scheduler, 'type', 'cosine')
            if scheduler_type == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, 
                    T_max=self.config.scheduler.T_max, 
                    eta_min=self.config.scheduler.eta_min
                )
            elif scheduler_type == 'exponential':
                self.scheduler = optim.lr_scheduler.ExponentialLR(
                    self.optimizer, 
                    gamma=getattr(self.config.scheduler, 'gamma', 0.98)
                )
            elif scheduler_type == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=getattr(self.config.scheduler, 'step_size', 50),
                    gamma=getattr(self.config.scheduler, 'gamma', 0.5)
                )
            
            tqdm.write("     Scheduler has been RESET for the new learning rate.")

        elif load_optimizer:
            # Case 2: --lr 옵션 없고, Resume인 경우 (체크포인트 상태 복원)
            tqdm.write(f"  Loading optimizer & scheduler state from checkpoint...")
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            current_lr = self.optimizer.param_groups[0]['lr']
            tqdm.write(f"    Resumed LR from checkpoint: {current_lr:.2e}")
        
        else:
            # Case 3: Resume도 아니고 Override도 아님 (기본값 유지)
            # Trainer.__init__에서 이미 Config 값으로 초기화되었으므로 아무것도 안 함
            tqdm.write(f"  Using default config LR: {self.config.training.learning_rate:.2e}")
            tqdm.write(f"  Optimizer state initialized fresh.")

        tqdm.write(f'Best val PSNR: {self.best_val_psnr:.2f} dB')
    
    def diagnose_training(self):
        """학습 상태 진단 (리스트 반환 대응 수정됨)"""
        self.model.eval()
        
        tqdm.write(f"\n{'='*70}")
        tqdm.write("🔍 TRAINING DIAGNOSIS")
        tqdm.write(f"{'='*70}\n")
        
        # 배치 준비
        batch = next(iter(self.train_loader))
        data = self._prepare_batch(batch, debug=False)
        rays_data = self._sample_rays(data, debug=False)
        
        with torch.no_grad():
            # ============================================================
            # ✅ 1. Feature Check (리스트 대응 수정)
            # ============================================================
            features_raw = self.model.encoder(data['src_images'])
            tqdm.write("1️⃣ Feature Extraction:")
            
            # 리스트인지 확인하여 분기 처리
            if isinstance(features_raw, list):
                tqdm.write(f"   ✅ Multi-scale Features detected (Count: {len(features_raw)})")
                for i, feat in enumerate(features_raw):
                    # 5D(B, V, C, H, W) -> 4D(B*V, C, H, W) View 변환 (통계 계산용)
                    if feat.dim() == 5: 
                        feat = feat.view(-1, *feat.shape[2:])
                    
                    std_val = feat.std()
                    mean_val = feat.mean()
                    tqdm.write(f"     - Level {i} shape={tuple(feat.shape)}: Mean={mean_val:.3f}, Std={std_val:.3f} {'✅' if std_val > 0.01 else '⚠️ Low var'}")
            else:
                # 기존 단일 텐서 로직 (Fallback)
                features = features_raw
                if features.dim() == 5:
                     features = features.view(-1, *features.shape[2:])
                tqdm.write(f"   Std: {features.std():.3f} {'✅' if features.std() > 0.1 else '❌ Too uniform!'}")
            
            # ============================================================
            # ✅ 2. RGB & Density Analysis
            # ============================================================
            tqdm.write("\n2️⃣ Network Output Analysis:")
            
            # pixelnerf.py의 forward 호출
            outputs = self.model(
                data['src_images'][0:1],
                data['src_intrinsics'][0:1],
                data['src_poses'][0:1],
                rays_data['rays_o'][0:1],
                rays_data['rays_d'][0:1],
                self.config.data.z_near,
                self.config.data.z_far,
                
            )
            
            # Coarse network
            coarse_rgb = outputs['coarse']['rgb_map'][0]
            coarse_acc = outputs['coarse']['acc_map']
            
            r_mean, g_mean, b_mean = coarse_rgb.mean(dim=0)
            rgb_diff = max(abs(r_mean - g_mean), abs(g_mean - b_mean), abs(r_mean - b_mean))
            
            tqdm.write(f"\n   📊 Coarse Network:")
            tqdm.write(f"      RGB: R={r_mean:.3f}, G={g_mean:.3f}, B={b_mean:.3f}")
            tqdm.write(f"      Color Diff: {rgb_diff:.4f} {'✅ Learning color!' if rgb_diff > 0.01 else '❌ Grayscale'}")
            tqdm.write(f"      Opacity: {coarse_acc.mean():.3f} {'✅' if 0.1 < coarse_acc.mean() < 0.9 else '⚠️ Extreme!'}")
            
            # Fine network
            fine_rgb = outputs['fine']['rgb_map'][0]
            fine_acc = outputs['fine']['acc_map']
            
            r_mean, g_mean, b_mean = fine_rgb.mean(dim=0)
            rgb_diff = max(abs(r_mean - g_mean), abs(g_mean - b_mean), abs(r_mean - b_mean))
            
            tqdm.write(f"\n   📊 Fine Network:")
            tqdm.write(f"      RGB: R={r_mean:.3f}, G={g_mean:.3f}, B={b_mean:.3f}")
            tqdm.write(f"      Color Diff: {rgb_diff:.4f} {'✅ Learning color!' if rgb_diff > 0.01 else '❌ Grayscale'}")
            tqdm.write(f"      Opacity: {fine_acc.mean():.3f} {'✅' if 0.1 < fine_acc.mean() < 0.9 else '⚠️ Extreme!'}")
            
            # Raw density check
            if 'raw_density' in outputs['coarse']:
                density = outputs['coarse']['raw_density']
                density_max = density.max()
                tqdm.write(f"\n   🧱 Density Peak: {density_max:.3f} {'✅ Solid' if density_max > 3.0 else '⚠️ Weak'}")
            
            # ============================================================
            # ✅ 3. Loss & PSNR
            # ============================================================
            tqdm.write("\n3️⃣ Loss Check:")
            tgt_rgb = rays_data['tgt_rgb']
            loss = self.criterion(fine_rgb, tgt_rgb[0])
            psnr = mse2psnr(loss)
            tqdm.write(f"   PSNR: {psnr.item():.2f} dB")
            
            tqdm.write(f"\n{'='*70}\n")
        
        self.model.train()

def main():
    parser = argparse.ArgumentParser(description='Train Pixel-NeRF')
    parser.add_argument('--config', type=str, default='config/default_config.yaml', 
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, 
                       help='Path to checkpoint to resume from')
    
    # [추가] 수동 LR 설정을 위한 인자
    parser.add_argument('--lr', type=float, default=None, 
                       help='Override learning rate (priority: High)')

    args = parser.parse_args()
     
    # Load config
    if args.config and os.path.exists(args.config):
        config = Config.from_yaml(args.config)
        tqdm.write(f"Loaded config from: {args.config}")
    else:
        tqdm.write(f"Config file not found at {args.config}, using default config")
        config = Config()
    
    # Save config
    config.to_yaml(os.path.join(config.training.checkpoint_dir, 'config.yaml'))
    
    # Resume 경로 설정
    if args.resume:
        config.training.resume_from = args.resume
        
    # args.lr이 있으면 config 객체에 임시 저장하여 Trainer에서 확인 가능하게 함
    if args.lr is not None:
        config.training.override_lr = args.lr
        
    # Train
    trainer = Trainer(config)

    trainer.diagnose_training()
    trainer.train()

if __name__ == '__main__':
    main()