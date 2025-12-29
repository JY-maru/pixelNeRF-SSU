import os
import sys
import glob
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import imageio
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

# 프로젝트 경로 설정
sys.path.append(os.getcwd())

from config.config import Config
from models.pixelnerf import PixelNeRF
from utils.geometry import CameraUtils

# ==============================================================================
# 1. 데이터 로드 함수 (Inference.py 로직 재사용 + TTO용 변형)
# ==============================================================================
ROTATION_MATRIX = torch.tensor([
    [ 0,  1,  0,  0],   # New X = Old Y
    [-1,  0,  0,  0],   # New Y = -Old X
    [ 0,  0,  1,  0],   # Z 유지
    [ 0,  0,  0,  1]
], dtype=torch.float32)

def parse_pose_file(path):
    with open(path, 'r') as f:
        values = [float(x) for x in f.read().split()]
    return torch.tensor(values, dtype=torch.float32).reshape(4, 4)

def load_single_instance_data(folder_path, target_size=(128, 128), device='cuda'):
    """
    특정 폴더(Instance)의 데이터를 로드하여 TTO 학습용 텐서로 반환
    """
    rgb_dir = os.path.join(folder_path, 'rgb')
    pose_dir = os.path.join(folder_path, 'pose')
    
    img_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")) + glob.glob(os.path.join(rgb_dir, "*.jpg")))
    pose_paths = sorted(glob.glob(os.path.join(pose_dir, "*.txt")))
    
    if not img_paths: return None

    # Transform: TTO시에는 Augmentation 없이 리사이즈만
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    images, poses = [], []
    
    for i in range(len(img_paths)):
        # 이미지 로드
        img = Image.open(img_paths[i]).convert("RGB")
        images.append(transform(img))
        
        # 포즈 로드 및 좌표계 변환
        pose = parse_pose_file(pose_paths[i])
        pose = torch.matmul(ROTATION_MATRIX, pose)
        poses.append(pose)
    
    # Intrinsics (ShapeNet 기본값 가정 또는 파일 로드)
    # 여기서는 간소화를 위해 기본값 사용 (inference.py 로직 참고)
    focal = 0.5 * target_size[0] / np.tan(0.5 * np.deg2rad(50)) # FOV 50 assumption
    intrinsic = torch.tensor([
        [focal, 0, target_size[1]/2],
        [0, focal, target_size[0]/2],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    intrinsics = intrinsic.unsqueeze(0).repeat(len(images), 1, 1) # (N, 3, 3)

    return {
        'images': torch.stack(images).to(device),       # (N, 3, H, W)
        'poses': torch.stack(poses).to(device),         # (N, 4, 4)
        'intrinsics': intrinsics.to(device)             # (N, 3, 3)
    }

# ==============================================================================
# 2. TTO 및 렌더링 클래스
# ==============================================================================
class TTOHandler:
    def __init__(self, config, checkpoint_path):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path
        self.base_model_state = None
        
        # 모델 초기화
        self.model = PixelNeRF(
            encoder_type=config.model.encoder_type,
            encoder_pretrained=False, 
            feature_dim=config.model.feature_dim,
            d_hidden=config.model.d_hidden,
            n_blocks=config.model.n_blocks,
            combine_type=config.model.combine_type,
            n_coarse=config.model.n_coarse,
            n_fine=config.model.n_fine,
            white_bkgd=config.model.white_bkgd,
            use_pe=config.model.use_pe,
            pe_freq_pos=config.model.pe_freq_pos,
            pe_freq_dir=config.model.pe_freq_dir
        ).to(self.device)
        
        # 원본 체크포인트 로드 및 저장 (매 인스턴스마다 리셋하기 위해)
        self._load_checkpoint(checkpoint_path)
        self.base_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        print(f"✅ Base model loaded from {checkpoint_path}")

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)

    def reset_model(self):
        """다음 자동차를 위해 모델을 원본 상태로 되돌림"""
        self.model.load_state_dict(self.base_model_state)

    def optimize_instance(self, src_data, steps=500, lr=1e-5):
        """
        [핵심] 단일 인스턴스에 대해 모델을 파인튜닝 (TTO)
        """
        self.model.train()
        
        # 전체 파라미터 학습 (또는 인코더 제외 가능)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        images = src_data['images']      # (N, 3, H, W)
        poses = src_data['poses']        # (N, 4, 4)
        intrinsics = src_data['intrinsics'] # (N, 3, 3)
        
        N, _, H, W = images.shape
        batch_rays = 1024 # 메모리 안전을 위해 적당히 설정
        
        pbar = tqdm(range(steps), desc="⚡ TTO Optimizing", leave=False)
        
        for _ in pbar:
            # 1. Random View & Pixel Sampling
            img_idx = np.random.randint(0, N)
            
            # 픽셀 좌표 랜덤 생성
            coords = torch.stack(torch.meshgrid(
                torch.arange(H, device=self.device),
                torch.arange(W, device=self.device)
            ), -1).reshape(-1, 2)
            
            select_inds = np.random.choice(coords.shape[0], size=[batch_rays], replace=False)
            select_coords = coords[select_inds] # (B, 2) -> (y, x) 순서 주의
            
            # Ground Truth Pixel Value 가져오기
            # grid 좌표는 (y, x) 순서이므로 indexing 주의
            target_rgb = images[img_idx, :, select_coords[:, 0], select_coords[:, 1]].T # (B, 3)
            
            # 2. Ray Generation (CameraUtils 사용)
            # select_coords는 (y, x) -> (row, col)
            # get_rays_at_coords 같은 함수가 없으면 수동 계산 또는 get_rays 후 인덱싱
            # 여기서는 효율을 위해 전체 Ray 생성 후 인덱싱 (H, W가 크지 않으므로 가능)
            
            rays_o, rays_d = CameraUtils.get_rays(H, W, intrinsics[img_idx], poses[img_idx])
            # rays: (H, W, 3) -> reshape -> (H*W, 3)
            rays_o = rays_o.reshape(-1, 3)[select_inds] # (B, 3)
            rays_d = rays_d.reshape(-1, 3)[select_inds]
            
            # 차원 추가 (Batch size 1 간주)
            rays_o = rays_o.unsqueeze(0) # (1, B, 3)
            rays_d = rays_d.unsqueeze(0)
            
            # 3. Model Forward
            # PixelNeRF는 conditioning을 위해 src info를 받음
            # 여기서 src는 '자기 자신'이 됨 (Few-shot learning)
            out = self.model(
                images.unsqueeze(0), 
                intrinsics.unsqueeze(0), 
                poses.unsqueeze(0),
                rays_o, 
                rays_d,
                z_near=self.config.data.z_near,
                z_far=self.config.data.z_far
            )
            
            rgb_pred = out['fine']['rgb_map'] # (1, B, 3)
            
            # 4. Loss & Backward
            loss = F.mse_loss(rgb_pred.squeeze(0), target_rgb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_description(f"Loss: {loss.item():.4f}")

    def render_orbit(self, src_data, output_path, render_size=512):
        """학습된 모델로 고해상도 오르빗 영상 렌더링"""
        self.model.eval()
        
        # 렌더링용 Intrinsic (해상도 변경 반영)
        H_src, W_src = src_data['images'].shape[-2:]
        scale = render_size / H_src
        
        # Source 0번 뷰의 Intrinsic 가져와서 스케일링
        tgt_intrinsic = src_data['intrinsics'][0].clone()
        tgt_intrinsic[:2] *= scale
        
        # Orbit Pose 생성
        center = np.array([0., 0., 0.])
        radius = 1.5 # 적절한 거리 설정
        poses = self._get_orbit_poses(40, radius, center, elevation=30)
        
        frames = []
        chunk_size = 1024 # 렌더링 시 OOM 방지
        
        print(f"🎥 Rendering video ({render_size}x{render_size})...")
        with torch.no_grad():
            for pose in tqdm(poses, desc="Rendering", leave=False):
                rays_o, rays_d = CameraUtils.get_rays(render_size, render_size, tgt_intrinsic, pose)
                rays_o = rays_o.reshape(-1, 3).unsqueeze(0)
                rays_d = rays_d.reshape(-1, 3).unsqueeze(0)
                
                rgb_chunks = []
                for i in range(0, rays_o.shape[1], chunk_size):
                    chunk_o = rays_o[:, i:i+chunk_size]
                    chunk_d = rays_d[:, i:i+chunk_size]
                    
                    out = self.model(
                        src_data['images'].unsqueeze(0), # Source Condition
                        src_data['intrinsics'].unsqueeze(0),
                        src_data['poses'].unsqueeze(0),
                        chunk_o, chunk_d,
                        self.config.data.z_near,
                        self.config.data.z_far
                    )
                    rgb_chunks.append(out['fine']['rgb_map'].cpu())
                
                img = torch.cat(rgb_chunks, dim=1).reshape(render_size, render_size, 3)
                img = torch.clamp(img, 0, 1).numpy()
                frames.append((img * 255).astype(np.uint8))
                
        imageio.mimsave(output_path, frames, fps=30)
        print(f"✨ Saved to {output_path}")

    def _get_orbit_poses(self, num_frames, radius, center, elevation):
        # Orbit Pose 생성 로직 (inference.py와 동일한 방식 사용 권장)
        # 간소화를 위해 간단한 로직 구현
        poses = []
        phi = np.deg2rad(90 - elevation)
        for i in range(num_frames):
            theta = 2 * np.pi * i / num_frames
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)
            
            cam_pos = np.array([x, y, z]) + center
            forward = center - cam_pos
            forward /= np.linalg.norm(forward)
            up = np.array([0, 0, 1])
            right = np.cross(forward, up)
            right /= np.linalg.norm(right)
            down = np.cross(forward, right)
            
            pose = np.eye(4)
            pose[:3, 0] = right
            pose[:3, 1] = down
            pose[:3, 2] = forward
            pose[:3, 3] = cam_pos
            
            pose_tensor = torch.from_numpy(pose).float()
            # 학습 때 Rotation Matrix 적용했다면 여기서도 필요
            pose_tensor = torch.matmul(ROTATION_MATRIX, pose_tensor)
            poses.append(pose_tensor)
            
        return torch.stack(poses).to(self.device)

# ==============================================================================
# 3. 메인 실행부
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default_config.yaml')
    parser.add_argument('--test_data_dir', type=str, default='/content/pixNeRF_shapeNet_v2_data/cars_test')
    parser.add_argument('--tto_steps', type=int, default=500, help="Instance당 학습 Step 수")
    parser.add_argument('--tto_size', type=int, default=256, help="TTO 학습 시 이미지 해상도 (높을수록 디테일 유리)")
    parser.add_argument('--render_size', type=int, default=512, help="최종 렌더링 해상도")
    args = parser.parse_args()

    # 1. Config 로드
    config = Config.from_yaml(args.config)
    
    # 2. TTO 핸들러 초기화 (모델 로드)
    handler = TTOHandler(config, config.inference.checkpoint_path)
    
    # 3. Test 데이터 폴더 검색 (Config 무시하고 직접 검색)
    if not os.path.exists(args.test_data_dir):
        print(f"❌ Error: Test directory not found: {args.test_data_dir}")
        return

    instance_folders = sorted(glob.glob(os.path.join(args.test_data_dir, "*")))
    # 하위 폴더인지 확인 (파일 제외)
    instance_folders = [f for f in instance_folders if os.path.isdir(f)]
    
    print(f"Found {len(instance_folders)} test instances.")
    print(f"ℹ️  TTO Settings: Steps={args.tto_steps}, TrainSize={args.tto_size}, RenderSize={args.render_size}")

    os.makedirs(config.inference.output_dir, exist_ok=True)

    # 4. 전체 루프 실행
    for idx, folder in enumerate(instance_folders):
        instance_name = os.path.basename(folder)
        print(f"\n[{idx+1}/{len(instance_folders)}] Processing: {instance_name}")
        
        # A. 모델 리셋 (이전 자동차 학습 내용 삭제)
        handler.reset_model()
        
        # B. 데이터 로드 (TTO용 해상도로 로드)
        # 여기서 tto_size(예: 256)를 줘서 학습 때보다 더 크게 보게 만듦
        data = load_single_instance_data(folder, target_size=(args.tto_size, args.tto_size))
        
        if data is None:
            print("   ⚠️ No images found, skipping...")
            continue
            
        # C. Test-Time Optimization 수행
        handler.optimize_instance(data, steps=args.tto_steps, lr=1e-5)
        
        # D. 결과 렌더링
        output_filename = os.path.join(config.inference.output_dir, f"TTO_{instance_name}.mp4")
        handler.render_orbit(data, output_filename, render_size=args.render_size)

if __name__ == "__main__":
    main()