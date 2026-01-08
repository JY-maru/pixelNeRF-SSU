import os
import sys
import glob
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

# 지표 계산용 라이브러리
try:
    import lpips
    from skimage.metrics import structural_similarity as ssim_func
except ImportError:
    print("❌ 필수 라이브러리 설치 필요: pip install lpips scikit-image")
    sys.exit(1)

sys.path.append(os.getcwd())

from config.config import Config
from models.pixelnerf import PixelNeRF
from utils.geometry import CameraUtils
from data.shapenet import SHAPENET_ROTATION_MATRIX # 변환행렬

# ==============================================================================
# 1. 데이터 로드
# ==============================================================================
def parse_pose_file(path):
    with open(path, 'r') as f:
        values = [float(x) for x in f.read().split()]
    return torch.tensor(values, dtype=torch.float32).reshape(4, 4)

def load_single_instance_data(folder_path, target_size=(128, 128), device='cuda'):
    rgb_dir = os.path.join(folder_path, 'rgb')
    pose_dir = os.path.join(folder_path, 'pose')
    
    img_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")) + glob.glob(os.path.join(rgb_dir, "*.jpg")))
    pose_paths = sorted(glob.glob(os.path.join(pose_dir, "*.txt")))
    
    if not img_paths: return None

    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    images, poses = [], []
    for i in range(len(img_paths)):
        img = Image.open(img_paths[i]).convert("RGB")
        images.append(transform(img))
        pose = parse_pose_file(pose_paths[i])
        pose = torch.matmul(SHAPENET_ROTATION_MATRIX, pose) # 행렬 적용
        poses.append(pose)
    
    focal = 0.5 * target_size[0] / np.tan(0.5 * np.deg2rad(50))
    intrinsic = torch.tensor([
        [focal, 0, target_size[1]/2],
        [0, focal, target_size[0]/2],
        [0, 0, 1]
    ], dtype=torch.float32)
    intrinsics = intrinsic.unsqueeze(0).repeat(len(images), 1, 1)

    return {
        'images': torch.stack(images).to(device),       
        'poses': torch.stack(poses).to(device),         
        'intrinsics': intrinsics.to(device)
    }

# ==============================================================================
# 2. TTO 및 평가 클래스 (영상 렌더링 함수 삭제됨)
# ==============================================================================
class TTOHandler:
    def __init__(self, config, checkpoint_path):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = PixelNeRF(
            encoder_type=config.model.encoder_type, encoder_pretrained=False, 
            feature_dim=config.model.feature_dim, d_hidden=config.model.d_hidden,
            n_blocks=config.model.n_blocks, combine_type=config.model.combine_type,
            n_coarse=config.model.n_coarse, n_fine=config.model.n_fine,
            white_bkgd=config.model.white_bkgd, use_pe=config.model.use_pe,
            pe_freq_pos=config.model.pe_freq_pos, pe_freq_dir=config.model.pe_freq_dir
        ).to(self.device)
        
        self._load_checkpoint(checkpoint_path)
        self.base_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(self.device)

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)

    def reset_model(self):
        self.model.load_state_dict(self.base_model_state)

    def optimize_instance(self, src_data, steps=200, lr=1e-5):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        images = src_data['images']
        poses = src_data['poses']
        intrinsics = src_data['intrinsics']
        N, _, H, W = images.shape
        batch_rays = 1024 
        
        # TTO Progress bar 제거 (너무 빨라서 로그만 지저분해질 수 있음, 원하면 주석 해제)
        for _ in range(steps):
            img_idx = np.random.randint(0, N)
            coords = torch.stack(torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device)), -1).reshape(-1, 2)
            select_inds = np.random.choice(coords.shape[0], size=[batch_rays], replace=False)
            select_coords = coords[select_inds]
            target_rgb = images[img_idx, :, select_coords[:, 0], select_coords[:, 1]].T
            
            rays_o, rays_d = CameraUtils.get_rays(H, W, intrinsics[img_idx], poses[img_idx])
            rays_o = rays_o.reshape(-1, 3)[select_inds].unsqueeze(0).unsqueeze(0)
            rays_d = rays_d.reshape(-1, 3)[select_inds].unsqueeze(0).unsqueeze(0)
            
            out = self.model(images.unsqueeze(0), intrinsics.unsqueeze(0), poses.unsqueeze(0), rays_o, rays_d, z_near=self.config.data.z_near, z_far=self.config.data.z_far)
            loss = F.mse_loss(out['fine']['rgb_map'].squeeze(0).squeeze(0), target_rgb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def evaluate_metrics(self, data):
        self.model.eval()
        psnr_list, ssim_list, lpips_list = [], [], []
        
        images = data['images']
        poses = data['poses']
        intrinsics = data['intrinsics']
        N, _, H, W = images.shape
        
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        chunk_size = 2048 # 메모리 부족하면 줄이세요
        
        with torch.no_grad():
            for i in range(N): # tqdm 제거 (메인 루프에서 진행상황 확인)
                gt_tensor = inv_normalize(images[i]) 
                gt_tensor = torch.clamp(gt_tensor, 0, 1)
                gt_img_np = gt_tensor.permute(1, 2, 0).cpu().numpy()

                pose = poses[i]
                intrinsic = intrinsics[i]
                rays_o, rays_d = CameraUtils.get_rays(H, W, intrinsic, pose)
                rays_o = rays_o.reshape(-1, 3).unsqueeze(0)
                rays_d = rays_d.reshape(-1, 3).unsqueeze(0)
                
                rgb_chunks, acc_chunks = [], []
                for k in range(0, rays_o.shape[1], chunk_size):
                    chunk_o = rays_o[:, k:k+chunk_size]
                    chunk_d = rays_d[:, k:k+chunk_size]
                    out = self.model(
                        data['images'].unsqueeze(0), data['intrinsics'].unsqueeze(0), data['poses'].unsqueeze(0), 
                        chunk_o, chunk_d, self.config.data.z_near, self.config.data.z_far
                    )
                    rgb_chunks.append(out['fine']['rgb_map'].cpu())
                    acc_chunks.append(out['fine']['acc_map'].cpu())
                
                pred_rgb = torch.cat(rgb_chunks, dim=1).reshape(H, W, 3) 
                pred_acc = torch.cat(acc_chunks, dim=1).reshape(H, W, 1)
                
                # 배경 합성 (흰색 배경)
                pred_img = pred_rgb + (1.0 - pred_acc) * 1.0 
                pred_img = torch.clamp(pred_img, 0, 1)
                pred_img_np = pred_img.numpy()

                mse = np.mean((gt_img_np - pred_img_np) ** 2)
                psnr_list.append(-10.0 * np.log10(mse))
                
                ssim_list.append(ssim_func(gt_img_np, pred_img_np, data_range=1.0, channel_axis=-1, win_size=3))
                
                pred_tensor_lpips = pred_img.permute(2, 0, 1).unsqueeze(0).to(self.device) * 2.0 - 1.0
                gt_tensor_lpips = gt_tensor.unsqueeze(0).to(self.device) * 2.0 - 1.0
                lpips_list.append(self.loss_fn_vgg(pred_tensor_lpips, gt_tensor_lpips).item())

        return {
            'PSNR': np.mean(psnr_list),
            'SSIM': np.mean(ssim_list),
            'LPIPS': np.mean(lpips_list)
        }

# ==============================================================================
# 3. 메인
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default_config.yaml')
    parser.add_argument('--test_data_dir', type=str, default='/content/pixNeRF_shapeNet_v2_data/cars_test')
    parser.add_argument('--tto_steps', type=int, default=100) # 속도를 위해 100으로 낮춤
    parser.add_argument('--tto_size', type=int, default=256)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    handler = TTOHandler(config, config.inference.checkpoint_path)
    
    if not os.path.exists(args.test_data_dir):
        print(f"[Error] 경로 없음: {args.test_data_dir}")
        return

    instance_folders = sorted(glob.glob(os.path.join(args.test_data_dir, "*")))
    instance_folders = [f for f in instance_folders if os.path.isdir(f)]
    
    output_dir = config.inference.output_dir
    os.makedirs(output_dir, exist_ok=True)
    metrics_file = os.path.join(output_dir, "test_metrics.txt")
    
    with open(metrics_file, "w") as f:
        f.write("Instance_ID, PSNR, SSIM, LPIPS\n")

    print(f" {len(instance_folders)}개 객체 평가 시작 (영상 생성 X)")
    
    total_psnr, total_ssim, total_lpips = [], [], []

    # 전체 진행 상황바 (tqdm)
    for folder in tqdm(instance_folders, desc="Processing Objects"):
        instance_name = os.path.basename(folder)
        handler.reset_model()
        data = load_single_instance_data(folder, target_size=(args.tto_size, args.tto_size))
        
        if data is None: continue
            
        handler.optimize_instance(data, steps=args.tto_steps) # TTO 수행
        metrics = handler.evaluate_metrics(data)          # 지표 측정
        
        with open(metrics_file, "a") as f:
            f.write(f"{instance_name}, {metrics['PSNR']:.4f}, {metrics['SSIM']:.4f}, {metrics['LPIPS']:.4f}\n")
        
        total_psnr.append(metrics['PSNR'])
        total_ssim.append(metrics['SSIM'])
        total_lpips.append(metrics['LPIPS'])

    # 최종 결과 출력
    avg_psnr = np.mean(total_psnr) if total_psnr else 0
    avg_ssim = np.mean(total_ssim) if total_ssim else 0
    avg_lpips = np.mean(total_lpips) if total_lpips else 0
    
    print("\n" + "="*40)
    print(f"✅ 최종 평균 점수 ({len(total_psnr)}개 완료)")
    print(f"   PSNR : {avg_psnr:.4f}")
    print(f"   SSIM : {avg_ssim:.4f}")
    print(f"   LPIPS: {avg_lpips:.4f}")
    print("="*40)
    
    with open(metrics_file, "a") as f:
        f.write(f"AVERAGE, {avg_psnr:.4f}, {avg_ssim:.4f}, {avg_lpips:.4f}\n")

if __name__ == "__main__":
    main()